""" Draft for future object containing all info on specific diagnostic.
For spectrometers, this includes:
- element, charge state, transition and wavelength measured
- LOS geometry
- instrument function
- ...

and links to functions to:
- read atomic data
- make fractional abundance
- calculate radiated power
- integration of the spectral shapes along the LOS given input profiles
- ...
"""

import scipy.constants as constants
from copy import deepcopy
import re
import numpy as np
import xarray as xr
from xarray import DataArray

from hda.hdaadas import ADASReader
from hda.atomdat import fractional_abundance
from hda.atomdat import get_atomdat
from hda.atomdat import radiated_power

from indica.numpy_typing import ArrayLike


class Spectrometer:
    """
    Data and methods to model spectrometer measurements

    Parameters
    ----------
    reader
        ADASreader class to read atomic data
    element
        Name of the element emitting the measured spectral components (e.g. "ar")
    charge
        Charge state (e.g. "16" for Ar16+)
    transition
        Measured transition as written in the ADAS files
        (e.g. "(1)1(1.0)-(1)0(0.0)" for w-line of He-like Ar)
    wavelength
        Measured wavelength in Angstroms as written in the ADAS files (e.g. 4.0)
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
        reader: ADASReader,
        element: str,
        charge: str,
        transition="",
        wavelength=None,
        name="",
        recom=False,
        geometry=None,
    ):
        self.reader = reader
        self.name = name
        self.element = element
        charge = charge.strip()
        m = re.search(r"(\d+)\S?", charge, re.I)
        self.charge = int(m.group(1))
        self.transition = transition
        self.wavelength = wavelength
        self.geometry = geometry

        # Read all available atomic data
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        self.atomdat_files = files

        # Use average tokamak electron density dependence
        for k in atomdat.keys():
            atomdat[k] = (
                atomdat[k]
                .interp(electron_density=5.0e19, method="nearest")
                .drop_vars(["electron_density"])
            )

        self.atomdat = atomdat
        self.ion_charges = self.atomdat["scd"].coords["ion_charges"]

        # Calculate ionization balance end emission characteristics
        # in local ionization equilibrium using standard Te
        el_temp = np.linspace(0, 1, 100) ** 2 * (20.0e3 - 20)
        el_temp += 20

        fz, emiss, lz_tot = self.radiation_characteristics(el_temp, recom=recom)
        self.atomdat["fz"] = xr.where(np.isnan(fz), 0, fz)
        self.atomdat["emiss"] = xr.where(np.isnan(emiss), 0, emiss)
        self.atomdat["lz_tot"] = xr.where(np.isnan(lz_tot), 0, lz_tot)

    def radiation_characteristics(self, el_temp, recom=False):
        """

        Parameters
        ----------
        el_temp
            Electron temperature for interpolation of atomic data
        recom
            Set to True if recombination emission is to be included

        Returns
        -------

        """
        atomdat = deepcopy(self.atomdat)
        for k in atomdat.keys():
            atomdat[k] = atomdat[k].interp(
                electron_temperature=el_temp, method="quadratic"
            )

        fz = fractional_abundance(atomdat["scd"], atomdat["acd"], element=self.element,)

        lz_tot_fz = radiated_power(
            atomdat["plt"], atomdat["prb"], fz, element=self.element
        )
        lz_tot = lz_tot_fz.sum(axis=0)

        if "index" in atomdat["pec"].coords:
            atomdat["pec"] = atomdat["pec"].swap_dims({"index": "type"})
            emiss = atomdat["pec"].sel(type="excit") * fz.sel(ion_charges=self.charge)
            if recom is True and "recom" in atomdat["pec"].type:
                emiss += atomdat["pec"].sel(type="recom") * fz.sel(
                    ion_charges=self.charge + 1
                )
        else:
            emiss = atomdat["pec"] * fz.sel(ion_charges=self.charge)

        emiss.name = (
            f"{self.element}{self.charge}+ " f"{self.wavelength} A emission region"
        )
        emiss[emiss < 0] = 0.0

        return fz, emiss, lz_tot

    def simulate_measurements(
        self, electron_density, electron_temperature, ion_temperature
    ):
        """
        Initialize data variables given time and radial coordinates, perform
        forward model of measurement given input plasma profiles

        Returns
        -------

        """

        # Initialize variables
        time = electron_temperature.coords["t"]

        # TODO: calculate rho along LOS here if not given in input
        rho_los = self.geometry["rho"]
        el_dens_los = electron_density.interp(rho_poloidal=rho_los)
        el_dens_los = xr.where(rho_los <= 1, el_dens_los, 0)
        el_temp_los = electron_temperature.interp(rho_poloidal=rho_los)
        el_temp_los = xr.where(rho_los <= 1, el_temp_los, 0)
        ion_temp_los = ion_temperature.interp(rho_poloidal=rho_los)
        ion_temp_los = xr.where(rho_los <= 1, ion_temp_los, 0)

        fz = self.atomdat["fz"].interp(
            electron_temperature=el_temp_los, method="quadratic"
        )
        emiss = (
            self.atomdat["emiss"].interp(
                electron_temperature=el_temp_los, method="quadratic"
            )
            * el_dens_los ** 2
        )
        emiss = xr.where((rho_los <= 1) * np.isfinite(emiss), emiss, 0)

        self.fz = fz
        self.emiss = emiss

        # Position of emission, and values of plasma profiles at those positions
        vals = np.full((len(time)), np.nan)
        coords = [("t", time)]
        attrs = {"err_in": deepcopy(vals), "err_out": deepcopy(vals)}
        self.pos = DataArray(deepcopy(vals), coords=coords, attrs=deepcopy(attrs))
        self.el_temp = DataArray(deepcopy(vals), coords=coords, attrs=deepcopy(attrs))
        self.ion_temp = DataArray(deepcopy(vals), coords=coords, attrs=deepcopy(attrs))

        for i, t in enumerate(time):
            rho_tmp = rho_los.sel(t=t).values
            rho_min = np.min(rho_tmp)
            x = np.array(range(len(emiss.sel(t=t))))
            y = emiss.sel(t=t)
            avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)

            self.pos[i] = rho_tmp[int(avrg)]
            self.pos.attrs["err_in"][i] = np.abs(
                rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)]
            )
            if self.pos[i] == rho_min:
                self.pos.attrs["err_in"][i] = 0.0
            if self.pos.attrs["err_in"][i] > self.pos[i]:
                self.pos.attrs["err_in"][i] = self.pos[i] - rho_min
            self.pos.attrs["err_out"][i] = np.abs(
                rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)]
            )

            x = emiss.sel(t=t)
            y = el_temp_los.sel(t=t)
            te_avrg, te_err_in, te_err_out, _, _ = calc_moments(
                x, y, ind_in=ind_in, ind_out=ind_out, simmetry=True
            )
            self.el_temp[i] = te_avrg
            self.el_temp.attrs["err_in"][i] = te_err_in
            self.el_temp.attrs["err_out"][i] = te_err_out

            y = ion_temp_los.sel(t=t)
            ti_avrg, ti_err_in, ti_err_out, _, _ = calc_moments(
                x, y, ind_in=ind_in, ind_out=ind_out, simmetry=True
            )
            self.ion_temp[i] = ti_avrg
            self.ion_temp.attrs["err_in"][i] = ti_err_in
            self.ion_temp.attrs["err_out"][i] = ti_err_out

    def bremsstrahlung(
        self,
        electron_temperature,
        electron_density,
        wavelength,
        zeff,
        gaunt_approx="callahan",
    ):
        """
        Calculate Bremsstrahlung along LOS

        Parameters
        ----------
        electron_temperature
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

        """

        def gaunt(self, electron_temperature, approx="callahan"):
            gaunt_funct = {
                "callahan": lambda electron_temperature: 1.35
                * electron_temperature ** 0.15
            }

            return gaunt_funct[approx]

        const = constants.e / (
            np.sqrt(2)
            * (3 * np.pi * constants.m_e) ** 1.5
            * constants.epsilon_0 ** 3
            * constants.c ** 2
        )
        gaunt = gaunt(electron_temperature, approx=gaunt_approx)
        ev_to_k = constants.physical_constants["electron volt-kelvin relationship"][0]
        wlenght = wavelength * 1.0e-9  # nm to m
        exponent = exp(
            -(constants.h * constants.c)
            / (wlenght * constants.k * ev_to_k * electron_temperature)
        )
        bremss = (
            const
            * (electron_density ** 2 * zeff / np.sqrt(constants.k * temperature))
            * (exponent / wlenght ** 2)
            * gaunt
        )

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
