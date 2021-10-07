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

ADF11 = {"ar": {"scd": "89", "acd": "89", "ccd": "89"}}
ADF15 = {
    "w": {
        "element": "ar",
        "file": ("16", "llu", "transport"),
        "charge": 16,
        "transition": "(1)1(1.0)-(1)0(0.0)",
        "wavelength": 4.0,
    }
}


class XRCSpectrometer:
    """
    Data and methods to model XRCS spectrometer measurements

    Parameters
    ----------
    reader
        ADASreader class to read atomic data
    name
        String identifier for the measurement type / spectrometer

    Examples
    ---------
    For passive C5+ measurements:
        spectrometer("c", "5",
                    transition="n=8-n=7", wavelength=5292.7)

    For passive he-like Ar measurements:
        spectrometer("ar", "16",
                    transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0)

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
        te_kw0, ti_w0 = self.moment_analysis(Ti)

        fz1, emiss1 = self.radiation_characteristics(Te, Ne, Nh=Nh)
        te_kw1, ti_w1 = self.moment_analysis(Ti)

        plt.figure()
        emiss0["w"].plot(label="Nh = 0", marker="o")
        emiss1["w"].plot(label="Nh != 0", marker="x")
        plt.title("w-line PEC * fz")
        plt.legend()

        plt.figure()
        self.Te.plot(label="Te", color="red")
        plt.plot(te_kw0.rho_poloidal, te_kw0.values, marker="o", color="red")
        plt.plot(te_kw1.rho_poloidal, te_kw1.values, marker="x", color="red")
        plt.hlines(te_kw0.values,
                   te_kw0.rho_poloidal - te_kw0.rho_poloidal_err["in"],
                   te_kw0.rho_poloidal + te_kw0.rho_poloidal_err["out"],
                   color="red")
        plt.hlines(te_kw1.values,
                   te_kw1.rho_poloidal - te_kw1.rho_poloidal_err["in"],
                   te_kw1.rho_poloidal + te_kw1.rho_poloidal_err["out"],
                   color="red")
        self.Ti.plot(label="Ti", color="black")
        plt.plot(ti_w0.rho_poloidal, ti_w0.values, marker="o", color="black")
        plt.plot(ti_w1.rho_poloidal, ti_w1.values, marker="x", color="black")
        plt.hlines(ti_w0.values,
                   ti_w0.rho_poloidal - ti_w0.rho_poloidal_err["in"],
                   ti_w0.rho_poloidal + ti_w0.rho_poloidal_err["out"],
                   color="black")
        plt.hlines(ti_w1.values,
                   ti_w1.rho_poloidal - ti_w1.rho_poloidal_err["in"],
                   ti_w1.rho_poloidal + ti_w1.rho_poloidal_err["out"],
                   color="black")
        plt.title("w-line PEC * fz")
        plt.legend()

        # return (fz0, fz1), (emiss0, emiss1)

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
                emiss_coeff = interp_pec(select_type(pec, type="excit"), Ne, Te)
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

                if recom is True and "recom" in pec.type:
                    emiss_coeff = interp_pec(select_type(pec, type="recom"), Ne, Te)
                    _emiss += emiss_coeff * fz[elem].sel(ion_charges=charge + 1)
            else:
                emiss_coeff = interp_pec(pec["emiss_coeff"], Ne, Te)
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

            emiss[line] = xr.where(_emiss >= 0, _emiss, 0)

        self.fz = fz
        self.emiss = emiss

        return fz, emiss

    def moment_analysis(
        self,
        Ti: ArrayLike,
        rho_los: ArrayLike = None,
        Cimp: dict = None,
    ):
        """
        Infer the spectrometer measurement of electron and ion temperatures

        Parameters
        ----------
        Ti
            Ion temperature profile
        Ne
            Electron density profiles (close to the values used for the interpolation
            of the atomic data)
        rho_los
            rho_poloidal for interpolation along the LOS
        Cimp
            impurity concentration profile (dictionary)

        Returns
        -------

        """
        coord = "rho_poloidal"

        Te = self.Te
        Ne = self.Ne
        elements = self.elements
        w_emiss = xr.where(np.isnan(self.emiss["w"]), 0, self.emiss["w"])
        # k_emiss = xr.where(np.isnan(self.emiss["k"]), 0, self.emiss["k"])

        if rho_los is None:
            rho_los = Te.rho_poloidal
        rho_tmp = rho_los.values
        rho_min = np.min(rho_tmp)

        if Cimp is None:
            Cimp = {}
            for elem in elements:
                Cimp[elem] = xr.full_like(Ne, 1.0)

        # TODO: currently using emission shape for both Te and Ti, change to account for shape of k-like
        w_emiss *= Cimp["ar"] * Ne ** 2
        w_emiss = xr.where((rho_los <= 1) * np.isfinite(w_emiss), w_emiss, 0)

        # Position of w-line emissivity and its ion temperature
        x = np.array(range(len(w_emiss)))
        y = w_emiss
        avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)

        pos = rho_tmp[int(avrg)]
        pos_err_in = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
        if pos <= rho_min:
            pos = rho_min
            pos_err_in = 0.0
        if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
            pos_err_in = pos - rho_min
        pos_err_out = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)])

        # Ion temperature
        x = w_emiss
        y = Ti
        ti_w, err_in, err_out, _, _ = calc_moments(
            x, y, ind_in=ind_in, ind_out=ind_out, simmetry=False
        )
        datatype = ("temperature", "ion")
        attrs = {
            "datatype": datatype,
            "err":{"in": err_in, "out": err_out},
            f"{coord}_err": {"in": pos_err_in, "out": pos_err_out},
        }
        ti_w = DataArray([ti_w], coords=[(coord, [pos])], attrs=attrs)

        # Position of w/k-lines emissivity and the measured electron temperature
        # x = np.array(range(len(w_emiss)))
        # y = w_emiss
        # avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)
        #
        # pos = rho_tmp[int(avrg)]
        # pos_err_in = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
        # if pos <= rho_min:
        #     pos = rho_min
        #     pos_err_in = 0.0
        # if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
        #     pos_err_in = pos - rho_min
        # pos_err_out = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)])

        # Electron temperature
        x = w_emiss
        y = Te
        te_kw, err_in, err_out, _, _ = calc_moments(
            x, y, ind_in=ind_in, ind_out=ind_out, simmetry=False
        )
        datatype = ("temperature", "electron")
        attrs = {
            "datatype": datatype,
            "err":{"in": err_in, "out": err_out},
            f"{coord}_err": {"in": pos_err_in, "out": pos_err_out},
        }
        te_kw = DataArray([te_kw], coords=[(coord, [pos])], attrs=attrs)

        self.rho_los = rho_los
        self.Cimp = Cimp
        self.Ti = Ti

        self.te_kw = te_kw
        self.ti_w = ti_w

        return te_kw, ti_w


def bremsstrahlung(
    Te,
    Ne,
    wavelength,
    zeff,
    gaunt_approx="callahan",
):
    """
    Calculate Bremsstrahlung along LOS

    Parameters
    ----------
    Te
        electron temperature (eV) for calculation
    wavelength
        wavelength (nm) at which Bremsstrahlung should be calculated
    zeff
        effective charge
    gaunt_approx
        approximation for free-free gaunt factors:
            "callahan" see citation in KJ Callahan 2019 JINST 14 C10002

    Returns
    -------
    Bremsstrahlung emissing per unit time and volume
    --> to be integrated along the LOS and multiplied by spectrometer t_exp

    """

    gaunt_funct = {"callahan": lambda Te: 1.35 * Te ** 0.15}

    const = constants.e ** 6 / (
        np.sqrt(2)
        * (3 * np.pi * constants.m_e) ** 1.5
        * constants.epsilon_0 ** 3
        * constants.c ** 2
    )
    gaunt = gaunt_funct[gaunt_approx](Te)
    ev_to_k = constants.physical_constants["electron volt-kelvin relationship"][0]
    wlenght = wavelength * 1.0e-9  # nm to m
    exponent = np.exp(
        -(constants.h * constants.c) / (wlenght * constants.k * ev_to_k * Te)
    )
    bremss = (
        const
        * (Ne ** 2 * zeff / np.sqrt(constants.k * Te))
        * (exponent / wlenght ** 2)
        * gaunt
    ) * wlenght

    return bremss


def calc_moments(y: ArrayLike, x: ArrayLike, ind_in=None, ind_out=None, simmetry=False):
    x_avrg = np.nansum(y * x) / np.nansum(y)

    if (ind_in is None) and (ind_out is None):
        ind_in = x <= x_avrg
        ind_out = x >= x_avrg
        if simmetry is True:
            ind_in = ind_in + ind_out
            ind_out = ind_in

    x_err_in = np.sqrt(
        np.nansum(y[ind_in] * (x[ind_in] - x_avrg) ** 2) / np.nansum(y[ind_in])
    )

    x_err_out = np.sqrt(
        np.nansum(y[ind_out] * (x[ind_out] - x_avrg) ** 2) / np.nansum(y[ind_out])
    )

    return x_avrg, x_err_in, x_err_out, ind_in, ind_out


def interp_pec(pec, Ne, Te):
    pec_interp = pec.indica.interp2d(
        electron_temperature=Te,
        electron_density=Ne,
        method="cubic",
        assume_sorted=True,
    )
    return pec_interp


def select_type(pec, type="excit"):
    if "index" in pec.dims:
        pec = pec.swap_dims({"index": "type"})
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
