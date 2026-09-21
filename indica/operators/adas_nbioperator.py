from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.operators import NbiOperator
from indica.readers import ADASReader
from indica.utilities import get_element_info


class NbiADAS(NbiOperator):
    """Calculate neutral beam attenuation from beam stopping coefficients, typically
    ADAS ADF21 (bms) data.

    """

    bms: dict[str, DataArray]

    def prepare(
        self,
        bms: dict[str, DataArray] | None = None,
        reader: ADASReader | None = None,
        year: str = "97",
        bms_quantity: str = "bms",
    ):
        """Calculate neutral attenuation factors

        Parameters
        ----------
        **kwargs : [type]
            [description]
        """
        if bms is None:
            if reader is None:
                reader = ADASReader()
            beam_species = str(self.nbi_element_info["symbol"]).lower()
            if beam_species == "d" or beam_species == "t":
                beam_species = "h"
            self.bms = {
                str(elem): reader.get_adf21(
                    element=str(elem),
                    charge=str(get_element_info(elem)[0]),
                    year=year,
                    beam=beam_species,
                    quantity=bms_quantity,
                )
                for elem in self.Ni.element.values
            }
        else:
            self.bms = bms

        current_fractions = xr.DataArray(
            np.asarray(self.current_fractions),
            dims=("fraction",),
            coords={
                "fraction": ("fraction", np.arange(1, len(self.current_fractions) + 1))
            },
        )
        amu = float(self.nbi_element_info["A"])
        e_amu = self.energy / amu
        self.source_neutral_flux = (
            self.power * (constants.e**-1.5) * np.sqrt(constants.m_p / 2)
        ) / (amu * (e_amu / current_fractions) ** 1.5)

        bms = xr.concat(
            [val.assign_coords({"element": key}) for key, val in self.bms.items()],
            dim="element",
        )
        bms_mapped = self.transform.map_profile_to_los(
            bms.interp(
                target_temperature=self.Te, target_density=self.Ne, beam_energy=e_amu
            ),
            t=np.asarray(self.t),
        ).assign_coords({"element": bms.element})
        ni_mapped = self.transform.map_profile_to_los(self.Ni, np.asarray(self.t))
        meanz_mapped = self.transform.map_profile_to_los(
            self.MeanZ, np.asarray(self.t)
        ).assign_coords(element=self.MeanZ.element)
        self.vbeam = np.sqrt(2 * self.energy * constants.e / constants.m_u)
        self.zeta = np.exp(
            -(
                bms_mapped
                * meanz_mapped.sel(element=bms_mapped.element)
                * ni_mapped
                / self.vbeam
            ).cumsum("los_position")
        )

    def run(self, **kwargs):
        self.neutral_density = self.source_neutral_flux * self.zeta

    def refactor_output(self):
        result = {"neutral_density": self.neutral_density}
        return result

    def __call__(
        self,
        Ti: DataArray,
        Te: DataArray,
        Ni: DataArray,
        Ne: DataArray,
        Nn: DataArray,
        Vtor: DataArray,
        Zeff: DataArray,
        MeanZ: DataArray,
        ImpurityCharge: int,
        target_element: str,
        t: float | ArrayLike,
        file_name: Optional[str] = "",
        pulse: int = 0,
        machine: str = "tokamak",
        prepare_kwargs: dict = {},
        run_kwargs: dict = {},
    ) -> dict:
        """
        Run NBI code for specified time-point (one only!)

        target_element - plasma main ion element symbol (e.g. "d" for deuterium)
        file_name - first part of the file name to save the NBI model data to
        """
        if not hasattr(self, "transform"):
            raise ValueError("transform is required (set it before calling)")

        if not hasattr(self.transform, "equilibrium"):
            raise ValueError("transform is missing equilibrium data")

        _element_info = get_element_info(target_element)
        self.target_element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }

        self.t = t
        self.file_name = file_name
        self.pulse = pulse
        self.machine = machine

        # self.Ti = Ti.interp(t=t)
        self.Te = Te
        self.Ni = Ni
        self.Ne = Ne
        # self.Nn = Nn.interp(t=t)
        # self.Vtor = Vtor.interp(t=t)
        self.Zeff = Zeff.sum("element")
        self.MeanZ = MeanZ
        self.impurity_charge = ImpurityCharge

        """Prepare input data structure for NBI code"""
        self.prepare(**prepare_kwargs)

        """Run NBI code"""
        self.run(**run_kwargs)

        """Reorganise NBI code output to return Indica-native results"""
        result = self.refactor_output()

        return result
