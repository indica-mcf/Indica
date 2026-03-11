from typing import List
from typing import Optional
from typing import Union

import numpy as np
from xarray import DataArray

from indica import Plasma
from indica.operators.beam_utils import adas_nbi_utils
from indica.operators.beam_utils import analytic_nbi_utils
from indica.operators.beam_utils import fidasim_utils
from .abstractoperator import Operator


class NBIOperator(Operator):

    """This operator should be operating on a standard plasma+profiles, and spit out
    fast neutral density and fast particle pressure. I believe it does.
    """

    def __init__(
        self,
        name: str,
        einj: float,
        pinj: float,
        current_fractions: List[float],
        ab: float,
        plasma_ion_amu: float = 2.014,
        file_name: str = "nbiop",
    ):
        # Initialized with beam related info; transform and plasma are set later.
        self.transform = None
        self.plasma = None

        self.name = name
        self.einj = einj
        self.pinj = pinj
        self.current_fractions = current_fractions
        self.ab = ab

        self.plasma_ion_amu = plasma_ion_amu
        self.file_name = file_name

    def __call__(
        self,
        nbi_model: str = "FIDASIM",
        ion_temperature: Optional[DataArray] = None,
        electron_temperature: Optional[DataArray] = None,
        electron_density: Optional[DataArray] = None,
        neutral_density: Optional[DataArray] = None,
        toroidal_rotation: Optional[DataArray] = None,
        zeff: Optional[DataArray] = None,
        *,
        t: Union[DataArray, float, int],
        file_name: Optional[str] = None,
        plasma: Optional[Plasma] = None,
    ) -> dict:

        self.nbi_model = nbi_model
        self.ion_temperature = ion_temperature
        self.electron_temperature = electron_temperature
        self.electron_density = electron_density
        self.neutral_density = neutral_density
        self.toroidal_rotation = toroidal_rotation
        self.zeff = zeff
        self.t = t
        if plasma is not None:
            self.plasma = plasma
        if file_name is not None:
            self.file_name = file_name

        if self.plasma is not None:
            if self.ion_temperature is None:
                self.ion_temperature = self.plasma.ion_temperature
            if self.electron_temperature is None:
                self.electron_temperature = self.plasma.electron_temperature
            if self.electron_density is None:
                self.electron_density = self.plasma.electron_density
            if self.neutral_density is None:
                self.neutral_density = self.plasma.neutral_density
            if self.toroidal_rotation is None:
                self.toroidal_rotation = self.plasma.toroidal_rotation
            if self.zeff is None:
                self.zeff = self.plasma.zeff

        if (
            self.ion_temperature is None
            or self.electron_temperature is None
            or self.electron_density is None
            or self.neutral_density is None
            or self.toroidal_rotation is None
            or self.zeff is None
        ):
            raise ValueError("Give inputs or assign plasma class!")

        if self.t is None:
            raise ValueError("t is required (pass to __call__)")
        if not self.file_name:
            raise ValueError("file_name is required (set it on init or pass to __call__)")
        if self.transform is None:
            raise ValueError("transform is required (set it before calling)")
        if (
            not hasattr(self.transform, "equilibrium")
            or self.transform.equilibrium is None
        ):
            raise ValueError("transform is missing equilibrium data")

        # Resolve which NBI model runner to use for this call.
        model = nbi_model
        model_key = str(model).strip().upper()
        model_handler = self._get_model_handler(model_key)

        # Build per-time context dictionaries (profiles + equilibrium geometry).
        contexts = self._build_nbi_contexts()

        # TODO: sequential time stepping
        # If multiple times are provided, we should run them in order and allow
        # the plasma to be updated between steps (the simulation may modify
        # plasma parameters). Suggested future API:
        #   __call__(..., plasma_updater: Callable[[Plasma, dict, Any], Plasma] = None)
        # where plasma_updater receives (plasma, ctx, result) and returns the
        # updated plasma to use for the next step. Default behavior is stateless.

        # Execute the selected model for each time slice and collect results.
        neutrals_by_time = {}
        for ctx in contexts:
            result = model_handler(self, ctx)
            if isinstance(result, dict):
                neutrals_by_time.update(result)
            elif result is not None:
                neutrals_by_time[float(ctx["time"])] = result

        # Return all neutrals indexed by time.
        return neutrals_by_time

    def _build_nbi_contexts(self):
        t = self.t

        # Normalize time input into a 1D array.
        t_values = np.atleast_1d(getattr(t, "data", t))

        # Resolve equilibrium from the transform.
        eq = self.transform.equilibrium

        # Build a per-time context dict with profiles and equilibrium geometry.
        contexts = []
        for time in t_values:
            rho_1d = self.ion_temperature.rhop.values
            ion_temperature_t = self.ion_temperature.sel(t=time).values
            electron_temperature_t = self.electron_temperature.sel(t=time).values
            electron_density_t = self.electron_density.sel(t=time).values
            neutral_density_t = self.neutral_density.sel(t=time).values
            toroidal_rotation_t = self.toroidal_rotation.sel(t=time).values
            zeffective = self.zeff.sum("element").sel(t=time).values

            # rho poloidal
            rho_2d = eq.rhop.interp(
                t=time,
                method="nearest",
            )

            # rho toroidal
            rho_tor = eq.convert_flux_coords(rho_2d, t=time)
            rho_tor = rho_tor[0].values

            # radius
            R = eq.rhop.R.values
            z = eq.rhop.z.values
            R_2d, z_2d = np.meshgrid(R, z)

            # B field components
            br, bz, bt, _ = eq.Bfield(
                eq.rhop.R,
                eq.rhop.z,
                t=time,
            )
            br = br.values
            bz = bz.values
            bt = bt.transpose("z", "R").values

            # TODO - remove this. Currently using this bt estimate
            #  as otherwise it wont run due to nan errors.
            irod = 3.0 * 1e6
            bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

            rho = rho_2d.values
            ctx = {
                "file_name": self.file_name,
                "time": time,
                "rho_1d": rho_1d,
                "rho": rho,
                "rho_tor": rho_tor,
                "R_2d": R_2d,
                "z_2d": z_2d,
                "br": br,
                "bz": bz,
                "bt": bt,
                "ion_temperature": ion_temperature_t,
                "electron_temperature": electron_temperature_t,
                "electron_density": electron_density_t,
                "neutral_density": neutral_density_t,
                "toroidal_rotation": toroidal_rotation_t,
                "zeffective": zeffective,
            }
            contexts.append(ctx)

        return contexts

    @staticmethod
    def _get_model_handler(model_key: str):
        handlers = {
            "FIDASIM": fidasim_utils._run_fidasim,
            "ANALYTIC": analytic_nbi_utils._run_analytic,
            "ADAS": adas_nbi_utils._run_adas,
        }
        if model_key not in handlers:
            supported = ", ".join(handlers.keys())
            raise ValueError(
                f"Unknown nbi_model '{model_key}'. Supported models: {supported}"
            )
        return handlers[model_key]
