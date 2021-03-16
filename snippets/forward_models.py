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

from copy import deepcopy

from snippets.atomdat import fractional_abundance
from snippets.atomdat import get_atomdat
from snippets.atomdat import radiated_power

from indica.readers import ADASReader


class spectrometer:
    """
    Data and methods to model spectrometer measurements

    Parameters
    ----------
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
        element: str,
        charge: str,
        transition="",
        wavelength=None,
        el_dens=None,
        name="",
    ):
        self.name = name
        self.reader = ADASReader()
        self.element = element
        self.charge = int(charge)
        self.transition = transition
        self.wavelength = wavelength

        # Read all available atomic data
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        self.atomdat_files = files

        # Sum recombination and excitation components
        if "transition" in atomdat["pec"].coords.keys():
            atomdat["pec"] = atomdat["pec"].swap_dims({"index": "type"}).sel(
                type="excit"
            ) + atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom")
            drop = ["wavelength", "transition"]
            atomdat["pec"] = atomdat["pec"].drop_vars(drop)

        if el_dens:
            for k in atomdat.keys():
                atomdat[k] = atomdat[k].interp(
                    electron_density=el_dens, method="nearest"
                )
                atomdat[k] = atomdat[k].drop_vars(["electron_density"])

        self.atomdat = atomdat

        self.exp = None
        self.sim = None

    def radiation_characteristics(self, el_temp, el_dens, tau=1.0):
        atomdat = deepcopy(self.atomdat)
        for k in atomdat.keys():
            atomdat[k] = atomdat[k].interp(
                electron_temperature=el_temp, method="quadratic"
            )

        fz = fractional_abundance(
            atomdat["scd"],
            atomdat["acd"],
            ne_tau=tau,
            element=self.element,
        )

        tot_rad_pow_fz = radiated_power(
            atomdat["plt"], atomdat["prb"], fz, element=self.element
        )
        tot_rad_pow = tot_rad_pow_fz.sum(axis=0)

        emiss = atomdat["pec"] * fz.sel(ion_charges=self.charge) * el_dens ** 2
        emiss.name = (
            f"{self.element}{self.charge}+ " f"{self.wavelength} A emission region"
        )
        emiss[emiss < 0] = 0.0

        return fz, emiss, tot_rad_pow


class spectrometer_he_like:
    def __init__(
        self,
        element="ar",
        charge="16",
        transition="(1)1(1.0)-(1)0(0.0)",
        wavelength=4.0,
        el_dens=None,
    ):
        """Models to perform forward modelling of X-ray crystal spectrometer"""
        self.reader = ADASReader()
        self.element = element
        # self.charge = int(charge)
        self.transition = transition
        self.wavelength = wavelength
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )

        self.atomdat_files = files
        self.exp = None
        self.sim = None

        if el_dens:
            for k in atomdat.keys():
                atomdat[k] = atomdat[k].interp(
                    electron_density=el_dens, method="nearest"
                )
                atomdat[k] = atomdat[k].drop_vars(["electron_density"])

        self.atomdat = atomdat


class spectrometer_passive_c5:
    def __init__(
        self,
        element="c",
        charge="5",
        transition="n=8-n=7",
        wavelength=5292.7,
        el_dens=None,
    ):
        """Models to perform forward modelling of spectrometer measuring passive C5"""
        self.reader = ADASReader()
        self.element = element
        # self.charge = int(charge)
        self.transition = transition
        self.wavelength = wavelength
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        atomdat["pec"] = atomdat["pec"].swap_dims({"index": "type"}).sel(
            type="excit"
        ) + atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom")
        self.atomdat_files = files
        self.exp = None
        self.sim = None

        if el_dens:
            for k in atomdat.keys():
                atomdat[k] = atomdat[k].interp(
                    electron_density=el_dens, method="nearest"
                )
                atomdat[k] = atomdat[k].drop_vars(["electron_density"])

        self.atomdat = atomdat
