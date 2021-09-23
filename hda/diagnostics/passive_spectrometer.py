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
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from hda.atomdat import get_atomdat

from indica.numpy_typing import ArrayLike


class Spectrometer:
    """
    Data and methods to model passive spectrometer measurements

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
        calc_defaults=False
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
        self.atomdat = atomdat
        self.ion_charges = np.arange(self.atomdat["scd"].coords["ion_charges"].size + 1)

        self.FractionalAbundance = FractionalAbundance(
            self.atomdat["scd"],
            self.atomdat["acd"],
            CCD=self.atomdat["ccd"],
        )

        self.PowerLoss = PowerLoss(
            self.atomdat["plt"],
            self.atomdat["prb"],
            PRC=self.atomdat["prc"],
        )

        # Calculate ionization balance end emission characteristics
        # in local ionization equilibrium using standard Te
        if calc_defaults:
            Te = 50 + np.linspace(1, 0, 50)**1.5 * (10.0e3 - 50)
            rho = np.linspace(0.0, 1.0, Te.size)
            Te = DataArray(
                Te,
                coords={"rho_poloidal": rho},
                dims=["rho_poloidal"],
            )
            Ne = DataArray(
                xr.full_like(Te, 5.0e19).values,
                coords={"rho_poloidal": rho},
                dims=["rho_poloidal"],
            )
            Nh = DataArray(
                rho ** 6 * (1.e15 - 1.e11) + 1.e11,
                coords={"rho_poloidal": rho},
                dims=["rho_poloidal"],
            )

            fz, emiss, lz_tot = self.radiation_characteristics(
                Te, Ne, Nh=Nh, recom=recom
            )
            self.atomdat["fz"] = xr.where(np.isnan(fz), 0, fz)
            self.atomdat["emiss"] = xr.where(np.isnan(emiss), 0, emiss)
            self.atomdat["lz_tot"] = xr.where(np.isnan(lz_tot), 0, lz_tot)

    def radiation_characteristics(
        self, Te, Ne, Nh=None, tau=None, recom=False, attrs=False,
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

        fz = self.FractionalAbundance(Ne, Te, Nh, tau=tau)
        fz = fz.assign_coords(electron_temperature=("rho_poloidal", Te))
        fz = fz.assign_coords(electron_density=("rho_poloidal", Ne))
        fz = fz.assign_coords(neutral_density=("rho_poloidal", Nh))

        lz_tot = self.PowerLoss(
            Ne,
            Te,
            fz,
            Nh=Nh,
        ).sum("ion_charges")
        lz_tot = lz_tot.assign_coords(electron_temperature=("rho_poloidal", Te))
        lz_tot = lz_tot.assign_coords(electron_density=("rho_poloidal", Ne))
        lz_tot = lz_tot.assign_coords(neutral_density=("rho_poloidal", Nh))

        if "index" in self.atomdat["pec"].coords:
            self.atomdat["pec"] = self.atomdat["pec"].swap_dims({"index": "type"})
            exc = (
                self.atomdat["pec"]
                .sel(type="excit")
                .indica.interp2d(
                    electron_temperature=Te,
                    electron_density=Ne,
                    method="cubic",
                    assume_sorted=True,
                )
            )
            emiss = exc * fz.sel(ion_charges=self.charge)
            if recom is True and "recom" in self.atomdat["pec"].type:
                rec = (
                    self.atomdat["pec"]
                    .sel(type="recom")
                    .indica.interp2d(
                        electron_temperature=Te,
                        electron_density=Ne,
                        method="cubic",
                        assume_sorted=True,
                    )
                )
                emiss += rec * fz.sel(ion_charges=self.charge + 1)
        else:
            pec = (
                self.atomdat["pec"]
                    .indica.interp2d(
                    electron_temperature=Te,
                    electron_density=Ne,
                    method="cubic",
                    assume_sorted=True,
                )
            )
            emiss = pec * fz.sel(ion_charges=self.charge)

        emiss.name = (
            f"{self.element}{self.charge}+ " f"{self.wavelength} A emission region"
        )
        emiss = xr.where(emiss >= 0, emiss, 0)

        return fz, emiss, lz_tot

    def simulate_measurements(
        self,
        rho_los,
        Ne,
        Te,
        Ti,
        Nh=None,
        recom=False,
        calc_emiss=True,
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
            fz, emiss, lz_tot = self.radiation_characteristics(
                Te,
                Ne=Ne,
                Nh=Nh,
                recom=recom,
            )
        else:
            fz = self.fz
            emiss = self.emiss
            lz_tot = self.lz_tot


        emiss = xr.where(np.isnan(emiss), 0, emiss) * Ne ** 2
        emiss = xr.where((rho_los <= 1) * np.isfinite(emiss), emiss, 0)

        rho_tmp = rho_los.values
        rho_min = np.min(rho_tmp)

        x = np.array(range(len(emiss)))
        y = emiss

        avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)

        self.pos.values = rho_tmp[int(avrg)]
        self.pos.attrs["err_in"] = np.abs(
            rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)]
        )
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

        if calc_emiss:
            self.fz = fz
            self.emiss = emiss
            self.lz_tot = lz_tot

    def bremsstrahlung(
        self,
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

        gaunt_funct = {
            "callahan": lambda Te: 1.35 * Te ** 0.15
        }

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
            -(constants.h * constants.c)
            / (wlenght * constants.k * ev_to_k * Te)
        )
        bremss = (
            const
            * (
                Ne ** 2
                * zeff
                / np.sqrt(constants.k * Te)
            )
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
