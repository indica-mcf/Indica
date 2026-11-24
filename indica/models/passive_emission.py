import numpy as np
import xarray as xr
from xarray import DataArray

from indica import physics as ph
from indica.configs.readers.adasconf import ADF15
from indica.models.beam_spectrometer import LocalEmissionModel
from indica.numpy_typing import LabeledArray
from indica.readers import ADASReader
from indica.utilities import get_element_info


def read_adf15s(
    elements: list,
) -> dict[str, dict[str, xr.DataArray]]:

    reader = ADASReader()
    pecs = {}
    for element in elements:
        pecs[element] = {}
        for charge in ADF15[element].keys():
            file_type = ADF15[element][charge]["file_type"]
            year = ADF15[element][charge]["year"]
            _pec = reader.get_adf15(
                element=element, charge=charge, filetype=file_type, year=year
            )
            _pec = _pec.assign_coords(
                {"wavelength": _pec.wavelength * 0.1}
            )  # Angstrom -> nm
            pecs[element][charge] = _pec
    return pecs


def format_pecs(
    pecs: dict[str, dict[str, xr.DataArray]], wavelength_bounds: slice = None
) -> dict[str, xr.DataArray]:
    """
    DataArrays have dimensions of:
    * electron temperature
    * electron density
    * type of transition (excit, recom, chexc)
    * ion charge of parent ion
    * wavelength of transition
    """

    formatted_pecs = {}
    for element, pec in pecs.items():
        _pecs = []
        for charge, pec_z in pec.items():
            pec_z = pec_z.swap_dims({"index": "wavelength"}).drop_vars("index")
            types = np.unique(pec_z.type)
            pec_list = []
            for _type in types:
                pec_type = pec_z.where(pec_z.type == _type, drop=True).drop_vars("type")
                pec_type = pec_type.expand_dims(
                    {"type": (_type,)},
                )
                pec_type = pec_type.drop_duplicates("wavelength")

                pec_list.append(pec_type)
            _pec = xr.concat(pec_list, "type")

            _pec = _pec.expand_dims(
                {"ion_charge": (int(charge),)},
            )
            if wavelength_bounds is not None:
                _pec = _pec.where(
                    (_pec.wavelength > wavelength_bounds.start)
                    & (_pec.wavelength < wavelength_bounds.stop),
                    drop=True,
                )
            _pecs.append(_pec)

        pec_element = xr.concat(_pecs, "ion_charge")
        pec_element = pec_element.interpolate_na("electron_temperature").interpolate_na(
            "electron_density"
        )
        formatted_pecs[element] = pec_element
    return formatted_pecs


class PassiveEmission(LocalEmissionModel):
    """Define a local passive emission model using only flux surface
    quantity inputs"""

    def __init__(
        self,
        name: str,
        pecs: dict,
        wavelength: np.array,
    ):

        self.name = name
        self.pecs = pecs
        self.wavelength = xr.DataArray(wavelength, {"wavelength": wavelength})
        self.intensity: dict[str, xr.DataArray] = None

    def _transition_matrix(self, element: str = "ar") -> xr.DataArray:
        """Returns transition matrix used to convert
        PECs to emissivity"""
        # fmt: off
        _Nimp = self.Nimp.sel(element=element, ).drop_vars("element")
        _Fz = self.Fz[element]
        transition_matrix = xr.concat([
            self.Ne * _Nimp * _Fz,
            self.Ne * _Nimp * _Fz,
            self.Nn * _Nimp * _Fz,
        ], "type").assign_coords(
            type=["excit", "recom", "chexc", ])
        # fmt: on
        return transition_matrix

    def calculate_emission(
        self,
    ) -> dict[str, xr.DataArray]:
        """
        Returns emissivity with dims:
        * time
        * rho poloidal
        * wavelength of transition
        """
        self.intensity = {}
        self.emission = []
        for element, pec in self.pecs.items():
            e_info = get_element_info(element)
            mult = self._transition_matrix(
                element=element,
            )
            pec = pec.interp(electron_temperature=self.Te, electron_density=self.Ne)
            self.intensity[element] = (pec * mult).sum("type").sum("ion_charge")

            _emission = ph.doppler_broaden(
                self.wavelength,
                self.intensity[element],
                self.intensity[element].wavelength,
                e_info[1],  # atomic mass
                self.Ti,  # + self.instrumental_broadening,
            )
            self.emission.append(
                _emission.sum("wavelength")
                .expand_dims("element")
                .rename({"window": "wavelength"})
            )

    def __call__(
        self,
        Te: DataArray = None,
        Ti: DataArray = None,
        Ne: DataArray = None,
        Nimp: DataArray = None,
        Fz: dict = None,
        Nn: DataArray = None,
        t: LabeledArray = None,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Te - electron temperature (eV)
        Ti - ion temperature (eV)
        Ne - electron density (m**-3)
        Nimp - impurity density (m**-3)
        fractional_abundance - fractional abundance
        Nn - neutral density (m**-3)
        t - time (s)

        Returns
        -------

        """

        if hasattr(self, "plasma"):
            if t is None:
                t = self.plasma.time_to_calculate
            Te = self.plasma.electron_temperature.sel(
                t=t,
            )
            Ne = self.plasma.electron_density.sel(
                t=t,
            )
            Nn = self.plasma.neutral_density.sel(
                t=t,
            )
            Fz = {}
            _Fz = self.plasma.fz
            for elem in _Fz.keys():
                Fz[elem] = _Fz[elem].sel(t=t)

            Ti = self.plasma.ion_temperature.sel(t=t)
            Nimp = self.plasma.impurity_density.sel(t=t)
        else:
            if (
                Ne is None
                or Te is None
                or Nn is None
                or Fz is None
                or Ti is None
                or Nimp is None
            ):
                raise ValueError("Give inputs or assign plasma class!")

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nn = Nn
        self.Fz = Fz
        self.Ti = Ti
        self.Nimp = Nimp

        self.calculate_emission()

        return self.emission
