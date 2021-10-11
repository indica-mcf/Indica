import scipy.constants as constants
from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.readers import ADASReader
from indica.operators.atomic_data import FractionalAbundance


from hda.profiles import Profiles

from indica.numpy_typing import ArrayLike

ADF11 = {"c": {"scd": "96", "acd": "96", "ccd": "96"}}
ADF15 = {
    "529": {
        "element": "c",
        "file": ("5", "bnd", "96"),
        "charge": 5,
        "transition": "n=8-n=7",
        "wavelength": 5292.7,
    }
}

class PISpectrometer:
    """
    Data and methods to model XRCS spectrometer measurements

    Parameters
    ----------
    reader
        ADASreader class to read atomic data
    name
        String identifier for the measurement type / spectrometer

    """

    def __init__(
        self,
        name="",
        recom=False,
        adf11:dict=None,
        adf15:dict=None,
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            Identifier for the spectrometer
        recom
            Set to True if
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)

        Returns
        -------

        """

        self.ADASReader = ADASReader()
        self.name = name
        self.recom = recom
        self.set_ion_data(adf11=adf11)
        self.set_pec_data(adf15=adf15)

    def test_flow(self):
        """
        Test module with standard inputs
        """

        kline = list(self.adf15.keys())[0]

        self.set_ion_data()
        self.set_pec_data()
        Ne = Profiles(datatype=("density", "electron")).yspl
        Te = Profiles(datatype=("temperature", "electron")).yspl
        Ti = Profiles(datatype=("temperature", "electron")).yspl
        Ti /= 2.0

        Nh_1 = 1.0e15
        Nh_0 = Nh_1 / 10
        Nh = Ne.rho_poloidal ** 5 * (Nh_1 - Nh_0) + Nh_0

        fz0, emiss0 = self.radiation_characteristics(Te, Ne)

        fz1, emiss1 = self.radiation_characteristics(Te, Ne, Nh=Nh)

        plt.figure()
        emiss0[kline].plot(label="Nh = 0", marker="o")
        emiss1[kline].plot(label="Nh != 0", marker="x")
        plt.title("PEC * fz")
        plt.legend()

    def set_ion_data(self, adf11: dict = None):
        """
        Read adf11 data and build fractional abundance objects for all elements
        whose lines are to included in the modelled spectra

        Parameters
        ----------
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)

        """

        fract_abu = {}
        if adf11 is None:
            adf11 = ADF11

        scd, acd, ccd = {}, {}, {}
        for elem in adf11.keys():
            scd[elem] = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd[elem] = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd[elem] = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])

            fract_abu[elem] = FractionalAbundance(
                scd[elem],
                acd[elem],
                CCD=ccd[elem],
            )

        self.adf11 = adf11
        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        self.fract_abu = fract_abu

    def set_pec_data(self, adf15: dict = None):
        """
        Read adf15 data and extract PECs for all lines to be included
        in the modelled spectra

        Parameters
        ----------
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)

        """

        if adf15 is None:
            adf15 = ADF15

        adf15_data = {}
        pec = deepcopy(adf15)
        elements = []
        for line in adf15.keys():
            element = adf15[line]["element"]
            transition = adf15[line]["transition"]
            wavelength = adf15[line]["wavelength"]
            charge, filetype, year = adf15[line]["file"]

            identifier = f"{element}_{charge}_{filetype}_{year}"
            if identifier not in adf15_data.keys():
                adf15_data[identifier] = self.ADASReader.get_adf15(
                    element, charge, filetype, year=year
                )

            pec[line]["emiss_coeff"] = select_transition(
                adf15_data[identifier], transition, wavelength
            )

            elements.append(element)

        self.adf15 = adf15
        self.pec = pec
        self.elements = elements

    def radiation_characteristics(
        self,
        Te,
        Ne,
        Nh=None,
        tau=None,
        recom=False,
    ):
        """

        Parameters
        ----------
        Te
            Electron temperature for interpolation of atomic data
        Ne
            Electron density for interpolation of atomic data
        Nh
            Neutral (thermal) hydrogen density for interpolation of atomic data
        recom
            Set to True if recombination emission is to be included

        Returns
        -------

        """

        if Nh is None:
            Nh = xr.full_like(Ne, 0.0)
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te

        fz = {}
        emiss = {}
        for line, pec in self.pec.items():
            elem = pec["element"]
            charge = pec["charge"]
            coords = pec["emiss_coeff"].coords

            if elem not in fz.keys():
                _fz = self.fract_abu[elem](Ne, Te, Nh, tau=tau)
                fz[elem] = xr.where(_fz >= 0, _fz, 0)

            if "index" in coords:
                emiss_coeff = interp_pec(select_type(pec["emiss_coeff"], type="excit"), Ne, Te)
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

                if recom is True and "recom" in pec["emiss_coeff"].type:
                    emiss_coeff = interp_pec(select_type(pec["emiss_coeff"], type="recom"), Ne, Te)
                    _emiss += emiss_coeff * fz[elem].sel(ion_charges=charge + 1)
            else:
                emiss_coeff = interp_pec(pec["emiss_coeff"], Ne, Te)
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

            emiss[line] = xr.where(_emiss >= 0, _emiss, 0)

        self.fz = fz
        self.emiss = emiss

        return fz, emiss

def select_type(pec, type="excit"):

    if "index" in pec.dims:
        pec = pec.swap_dims({"index": "type"})
    if "transition" in pec.dims:
        pec = pec.swap_dims({"transition": "type"})

    return pec.sel(type=type)


def select_transition(adf15_data, transition: str, wavelength: float):

    """
    Given adf15 data in input, select pec for specified spectral line, given
    transition and wavelength identifiers

    Parameters
    ----------
    adf15_data
        adf15 data
    transition
        transition for spectral line as specified in adf15
    wavelength
        wavelength of spectral line as specified in adf15

    Returns
    -------
    pec data of desired spectral line

    """

    pec = deepcopy(adf15_data)

    dim = [
        d for d in pec.dims if d != "electron_temperature" and d != "electron_density"
    ][0]
    if dim != "transition":
        pec = pec.swap_dims({dim: "transition"})
    pec = pec.sel(transition=transition, drop=True)

    if len(np.unique(pec.coords["wavelength"].values)) > 1:
        pec = pec.swap_dims({"transition": "wavelength"})
        try:
            pec = pec.sel(wavelength=wavelength, drop=True)
        except KeyError:
            pec = pec.sel(wavelength=wavelength, method="nearest", drop=True)

    return pec
