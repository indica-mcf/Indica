import scipy.constants as constants
from copy import deepcopy
import re
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
        defaults=False,
        fract_abu: FractionalAbundance = None,
    ):
        self.ADASReader = ADASReader()
        self.name = name
        self.recom = recom
        self.fract_abu = fract_abu


    def run_default(self):
        self.get_ion_data()
        self.get_pec_data()

        rho = np.linspace(0 1, 100)
        yspl = DataArray(yspl, coords=[(coord, self.xspl)])
        attrs = {"datatype": self.datatype}

        Te = Profiles(datatype=("temperature", "electron"))
        Te.build_profile(5.e3, 50.)
        Ti = Profiles(datatype=("temperature", "ion")).yspl
        Ne = Profiles(datatype=("density", "electron")).yspl
        Nh = Profiles(datatype=("density", "neutral_h")).yspl
        Nh_0 = 1.e13
        Nh_1 = 1.e15
        Nh.values = (Nh.rho_poloidal ** 5 * (Nh_1 - Nh_0) + Nh_0)

        self.radiation_characteristics(Te, Ne, Nh=Nh)

    def get_ion_data(self, adf11=None):
        """
        Read adf11 data and build fractional abundance objects

        Parameters
        ----------
        adf11

        Returns
        -------

        """
        fract_abu = {}
        if adf11 is None:
            adf11 = ADF11

        scd, acd, ccd = {}, {}, {}
        for elem in adf11.keys():
            scd[elem] = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd[elem] = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd[elem] = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])

            if self.fract_abu is None:
                fract_abu[elem] = FractionalAbundance(
                    scd[elem], acd[elem], CCD=ccd[elem],
                )

        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        if self.fract_abu is None:
            self.fract_abu = fract_abu

    def get_pec_data(self, adf15=None):
        """
        Read af15 data and extract PECs of desired lines

        Parameters
        ----------
        adf15

        Returns
        -------

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
        self, Te, Ne, Nh=None, Nimp=None, tau=None, recom=False,
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
            Nh = xr.full_like(Ne, 1.0)

        fz = {}
        emiss = {}
        for line, pec in self.pec.items():
            if pec["element"] not in fz.keys():
                _fz = self.fract_abu[pec["element"]](Ne, Te, Nh, tau=tau)
                _fz = xr.where(_fz >= 0, _fz, 0)
                _fz = _fz.assign_coords(electron_temperature=("rho_poloidal", Te))
                _fz = _fz.assign_coords(electron_density=("rho_poloidal", Ne))
                _fz = _fz.assign_coords(neutral_density=("rho_poloidal", Nh))
                fz[pec["element"]] = _fz

            if "index" in pec["emiss_coeff"].coords:
                if "index" in pec.dims:
                    pec = pec.swap_dims({"index": "type"})
                emiss_coeff = pec.sel(type="excit").indica.interp2d(
                    electron_temperature=Te,
                    electron_density=Ne,
                    method="cubic",
                    assume_sorted=True,
                )
                _emiss = emiss_coeff * fz[pec["element"]].sel(ion_charges=pec["charge"])

                if recom is True and "recom" in pec.type:
                    emiss_coeff = pec.sel(type="recom").indica.interp2d(
                        electron_temperature=Te,
                        electron_density=Ne,
                        method="cubic",
                        assume_sorted=True,
                    )
                    _emiss += emiss_coeff * fz[pec["element"]].sel(ion_charges=pec["charge"] + 1)
            else:
                emiss_coeff = pec["emiss_coeff"].indica.interp2d(
                    electron_temperature=Te,
                    electron_density=Ne,
                    method="cubic",
                    assume_sorted=True,
                )
                _emiss = emiss_coeff * fz[pec["element"]].sel(ion_charges=pec["charge"])

            if Nimp is not None:
                _emiss *= Nimp[pec["element"]]

            _emiss = xr.where(_emiss >= 0, _emiss, 0)
            _emiss = _emiss.assign_coords(electron_temperature=("rho_poloidal", Te))
            _emiss = _emiss.assign_coords(electron_density=("rho_poloidal", Ne))
            _emiss = _emiss.assign_coords(neutral_density=("rho_poloidal", Nh))

            emiss[line] = _emiss

        self.fz = fz
        self.emiss = emiss

        return fz, emiss

    def simulate_measurements(
        self, rho_los, Ne, Te, Ti, Nh=None, recom=False, calc_emiss=True,
    ):
        """
        Initialize data variables given time and radial coordinates, perform
        forward model of measurement given input plasma profiles

        Returns
        -------

        """

        if Nh is None:
            Nh = xr.full_like(Ne, 1.0)

        vals = np.nan
        attrs = {"err_in": deepcopy(vals), "err_out": deepcopy(vals)}
        self.pos = DataArray(deepcopy(vals), attrs=deepcopy(attrs))
        self.el_temp = DataArray(deepcopy(vals), attrs=deepcopy(attrs))
        self.ion_temp = DataArray(deepcopy(vals), attrs=deepcopy(attrs))

        if calc_emiss:
            fz, emiss = self.radiation_characteristics(Te, Ne=Ne, Nh=Nh, recom=recom,)
            self.fz = fz
            self.emiss = emiss
        else:
            fz = self.fz
            emiss = self.emiss

        emiss = xr.where(np.isnan(emiss), 0, emiss) * Ne ** 2
        emiss = xr.where((rho_los <= 1) * np.isfinite(emiss), emiss, 0)

        rho_tmp = rho_los.values
        rho_min = np.min(rho_tmp)

        x = np.array(range(len(emiss)))
        y = emiss

        avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)

        self.pos.values = rho_tmp[int(avrg)]
        self.pos.attrs["err_in"] = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
        if self.pos.values == rho_min:
            self.pos.attrs["err_in"] = 0.0
        if self.pos.attrs["err_in"] > self.pos:
            self.pos.attrs["err_in"] = self.pos - rho_min
        self.pos.attrs["err_out"] = np.abs(
            rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)]
        )

        x = emiss
        y = Te
        te_avrg, te_err_in, te_err_out, _, _ = calc_moments(
            x, y, ind_in=ind_in, ind_out=ind_out, simmetry=True
        )
        self.el_temp.values = te_avrg
        self.el_temp.attrs["err_in"] = te_err_in
        self.el_temp.attrs["err_out"] = te_err_out

        y = Ti
        ti_avrg, ti_err_in, ti_err_out, _, _ = calc_moments(
            x, y, ind_in=ind_in, ind_out=ind_out, simmetry=True
        )
        self.ion_temp.values = ti_avrg
        self.ion_temp.attrs["err_in"] = ti_err_in
        self.ion_temp.attrs["err_out"] = ti_err_out

    def bremsstrahlung(
        self, Te, Ne, wavelength, zeff, gaunt_approx="callahan",
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


def select_transition(df15_data, transition: str, wavelength: float):

    """
    Given adf15 data in input, select pec for specified spectral line

    Parameters
    ----------
    df15_data
        adf15 data
    transition
        transition for spectral line as specified in adf15
    wavelength
        wavelength of spectral line as specified in adf15

    Returns
    -------
    pec data of desired spectral line

    """

    pec = deepcopy(df15_data)

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
