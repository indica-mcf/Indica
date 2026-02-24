import os
from typing import List

import numpy as np

from indica.operators import adas_nbi_utils
from indica.operators import analytic_nbi_utils
from indica.operators import fidasim_utils



PATH_TO_TE_FIDASIM = os.path.dirname(os.path.realpath(__file__))
print(f'PATH_TO_TE_FIDASIM = {PATH_TO_TE_FIDASIM}')


from .abstractoperator import Operator

# Flow/Config map:
# 1) NBIOperator takes a transform + nbispecs (=beam specs).
# 2) nbispecs (from nbi_configs.DEFAULT_NBI_SPECS or test overrides) supplies
#    beam operating params (einj/pinj/current_fractions/ab).
# 3) fidasim_utils.prepare_fidasim builds FIDASIM inputs by combining:
#    - nbispecs (beam params; also picks beam name for geometry),
#    - plasmaconfig (equilibrium + profiles),
#    - global settings in nbi_configs.py (paths, MC settings, grids, switches),
#    - beam geometry from get_hnbi_geo/get_rfx_geo via create_st40_beam_grid.
# 4) Resulting inputs are written to FIDASIM_OUTPUT_DIR and run.



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
        selected_model: str = "FIDASIM",
        pulse: int = None,
        plasma_ion_amu: float = 2.014,
    ):
        # Initialized with beam related info; transform is set later.
        self.transform = None
        self.selected_model = selected_model
        self.pulse = pulse

        # NBI config
        self.name = name
        self.einj = einj
        self.pinj = pinj
        self.current_fractions = current_fractions
        self.ab = ab
        self.nbispecs = {
            "name": self.name,
            "einj": self.einj,
            "pinj": self.pinj,
            "current_fractions": self.current_fractions,
            "ab": self.ab,
        }

        self.plasma_ion_amu = plasma_ion_amu

    def __call__(
        self,
        nbi_model="FIDASIM",
        ion_temperature=None,
        electron_temperature=None,
        electron_density=None,
        neutral_density=None,
        toroidal_rotation=None,
        zeff=None,
        t=None,
        pulse: int = None,
        plasma=None,
    ) -> dict:
        model = nbi_model or self.selected_model or "FIDASIM"
        model_key = str(model).strip().upper()
        model_handler = self._get_model_handler(model_key)

        if plasma is not None:
            self.plasma = plasma
        if self.plasma is not None:
            if ion_temperature is None:
                ion_temperature = self.plasma.ion_temperature
            if electron_temperature is None:
                electron_temperature = self.plasma.electron_temperature
            if electron_density is None:
                electron_density = self.plasma.electron_density
            if neutral_density is None:
                neutral_density = self.plasma.neutral_density
            if toroidal_rotation is None:
                toroidal_rotation = self.plasma.toroidal_rotation
            if zeff is None:
                zeff = self.plasma.zeff
            if t is None:
                t = getattr(self.plasma, "time_to_calculate", None)
                if t is None:
                    t = self.plasma.t

        if (
            ion_temperature is None
            or electron_temperature is None
            or electron_density is None
            or neutral_density is None
            or toroidal_rotation is None
            or zeff is None
        ):
            raise ValueError("Give inputs or assign plasma class!")

        if t is None:
            t = ion_temperature.t
        t_values = np.atleast_1d(getattr(t, "data", t))

        if pulse is None:
            pulse = self.pulse
        if pulse is None:
            raise ValueError("pulse is required (set it on init or pass to __call__)")

        if self.transform is None:
            raise ValueError("transform is required (set it before calling)")
        if not hasattr(self.transform, "equilibrium") or self.transform.equilibrium is None:
            raise ValueError("transform is missing equilibrium data")
        eq = self.transform.equilibrium

        profiles = {
            "t": t_values,
            "ion_temperature": ion_temperature,
            "electron_temperature": electron_temperature,
            "electron_density": electron_density,
            "neutral_density": neutral_density,
            "toroidal_rotation": toroidal_rotation,
            "zeff": zeff,
        }
        eqdata = {
            "rhop": eq.rhop,
            "convert_flux_coords": eq.convert_flux_coords,
            "Br": eq.Br,
            "Bz": eq.Bz,
        }

        neutrals_by_time = {}
        for i_time, time in enumerate(profiles["t"]):
            rho_1d = profiles["ion_temperature"].rhop.values
            ion_temperature = profiles["ion_temperature"].sel(t=time).values
            electron_temperature = profiles["electron_temperature"].sel(t=time).values
            electron_density = profiles["electron_density"].sel(t=time).values
            neutral_density = profiles["neutral_density"].sel(t=time).values
            toroidal_rotation = profiles["toroidal_rotation"].sel(t=time).values
            zeffective = profiles["zeff"].sum("element").sel(t=time).values




            # rho poloidal
            rho_2d = eqdata["rhop"].interp(
                t=time,
                method="nearest"
            )

            # rho toroidal
            # equilibrium too (convert_flux_coordinates func)
            rho_tor = eqdata["convert_flux_coords"](rho_2d, t=time)
            rho_tor = rho_tor[0].values

            # radius
            R = eqdata["rhop"].R.values
            z = eqdata["rhop"].z.values
            R_2d, z_2d = np.meshgrid(R, z)

            # Br
            br, _ = eqdata["Br"](
                eqdata["rhop"].R,
                eqdata["rhop"].z,
                t=time
            )
            br = br.values

            # Bz
            bz, _ = eqdata["Bz"](
                eqdata["rhop"].R,
                eqdata["rhop"].z,
                t=time
            )
            bz = bz.values

            # Bt
            # bt, _ = plasma.equilibrium.Bt(
            #     plasma.equilibrium.rhop.R,
            #     plasma.equilibrium.rhop.z,
            #     t=time
            # )
            # bt = bt.values  # NaN values an issue??

            # this comes from eq inside transform. transform.eq.bfield


            irod = 3.0 * 1e6
            bt = irod * (4 * np.pi * 1e-7) / (2 * np.pi * R_2d)

            # rho
            # comes from eq too
            rho = rho_2d.values
            ctx = {
                "pulse": pulse,
                "time": time,
                "rho_1d": rho_1d,
                "rho": rho,
                "rho_tor": rho_tor,
                "R_2d": R_2d,
                "z_2d": z_2d,
                "br": br,
                "bz": bz,
                "bt": bt,
                "ion_temperature": ion_temperature,
                "electron_temperature": electron_temperature,
                "electron_density": electron_density,
                "neutral_density": neutral_density,
                "toroidal_rotation": toroidal_rotation,
                "zeffective": zeffective,
                "neutrals_by_time": neutrals_by_time,
            }
            result = model_handler(self, ctx)
            if isinstance(result, dict):
                neutrals_by_time = result
            elif result is not None:
                neutrals_by_time[float(time)] = result

        return neutrals_by_time

    def _get_model_handler(self, model_key: str):
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
