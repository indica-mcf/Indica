import numpy as np
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.operators import NbiOperator
from indica.readers import ADASReader
from indica.utilities import get_element_info


class NbiAnalytic(NbiOperator):
    """Calculate neutral beam attenuation from beam stopping coefficients, typically
    ADAS ADF21 (bms) data.

    """

    bms: dict[str, DataArray]

    def prepare(
        self,
        bms: dict[str, DataArray] | None,
        reader: ADASReader | None,
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

        z = DataArray(
            [get_element_info(val)[0] for val in self.bms.keys()],
            dims=("element",),
            coords={"element": ("element", list(self.bms.keys()))},
        )
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
        conc_mapped = self.transform.map_profile_to_los(
            self.Ni / self.Ne, np.asarray(self.t)
        )
        self.vbeam = np.sqrt(2 * self.energy * constants.e / constants.m_u)
        self.stopping_cross_section = bms_mapped * z * conc_mapped / self.vbeam  # m^2

    def run(self, **kwargs):
        ...

    def refactor_output(self):
        ...


if __name__ == "__main__":
    from indica import Plasma
    from indica.configs.operators.nbi_configs import get_default_nbi_transform_config
    from indica.converters.line_of_sight import LineOfSightTransform
    from indica.defaults.load_defaults import load_default_objects

    machine = "st40"
    nbi_cfg = get_default_nbi_transform_config()
    nbi_transform = LineOfSightTransform(**nbi_cfg)

    equilibrium = load_default_objects(machine, "equilibrium")
    plasma: Plasma = load_default_objects(machine, "plasma")
    plasma.set_equilibrium(equilibrium)
    nbi_transform.set_equilibrium(equilibrium)

    nbi_op = NbiAnalytic(
        name="hnbi",
        energy=1.15e05 * 2.01410177784,  # eV
        power=2.09e06,  # W
        nbi_element="d",
        current_fractions=(0.5, 0.35, 0.15),
    )
    nbi_op.set_transform(nbi_transform)
    nbi_op.t = plasma.electron_density.t[0].values
    nbi_op.Ne = plasma.electron_density.isel(t=0)
    nbi_op.Ni = plasma.ion_density.sel(element=["h", "c"]).isel(t=0)
    nbi_op.Te = plasma.electron_temperature.isel(t=0)
    nbi_op.prepare(None, None)
