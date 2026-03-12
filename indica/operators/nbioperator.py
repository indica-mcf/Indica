from typing import List
from typing import Optional
from typing import Union

import numpy as np
from xarray import DataArray

from indica.operators.beam_utils import adas_nbi_utils
from indica.operators.beam_utils import analytic_nbi_utils
from indica.operators.beam_utils import fidasim_utils
from .abstractoperator import Operator


class NBIOperator(Operator):

    """This operator should operate on profile inputs and spit out
    fast neutral density and fast particle pressure.
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
        nbi_model: str = "FIDASIM",
    ):
        # Initialized with beam related info; transform is set later.
        self.transform = None

        self.name = name
        self.einj = einj
        self.pinj = pinj
        self.current_fractions = current_fractions
        self.ab = ab

        self.plasma_ion_amu = plasma_ion_amu
        self.file_name = file_name
        self.nbi_model = nbi_model

    def __call__(
        self,
        ion_temperature: DataArray,
        electron_temperature: DataArray,
        electron_density: DataArray,
        neutral_density: DataArray,
        toroidal_rotation: DataArray,
        zeff: DataArray,
        *,
        t: Union[DataArray, float, int],
        file_name: Optional[str] = None,
    ) -> dict:

        self.ion_temperature = ion_temperature
        self.electron_temperature = electron_temperature
        self.electron_density = electron_density
        self.neutral_density = neutral_density
        self.toroidal_rotation = toroidal_rotation
        self.zeff = zeff
        self.t = t
        if file_name is not None:
            self.file_name = file_name

        if (
            self.ion_temperature is None
            or self.electron_temperature is None
            or self.electron_density is None
            or self.neutral_density is None
            or self.toroidal_rotation is None
            or self.zeff is None
        ):
            raise ValueError("All profile inputs are required.")

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
        model = self.nbi_model
        model_key = str(model).strip().upper()
        model_handler = self._get_model_handler(model_key)

        # Build per-time context dictionaries (profiles + equilibrium geometry).
        contexts = self._build_nbi_contexts()

        # TODO: sequential time stepping
        # If multiple times are provided, we should run them in order and allow
        # the profile state to be updated between steps (the simulation may modify
        # profile parameters). Suggested future API:
        #   __call__(..., profile_updater: Callable[[dict, Any], dict] = None)
        # where profile_updater receives (ctx, result) and returns updated inputs
        # for the next step. Default behavior is stateless.

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
