""" Draft for future object containing all info on specific diagnostic. For spectrometers, this includes:
- element, charge state, transition and wavelength measured
- LOS geometry
- instrument function
- ...

and links to functions to:
- read atomic data
- make fractional abundance
- calculate radiated power
- ...
"""

import numpy as np

from indica.readers import ADASReader
from snippets.atomdat import get_atomdat


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
        self.charge = int(charge)
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
                try:
                    atomdat[k] = atomdat[k].interp(
                        electron_density=el_dens, method="nearest"
                    )
                    atomdat[k] = atomdat[k].drop_vars(["electron_density"])
                except:
                    atomdat[k] = atomdat[k].interp(
                        log10_electron_density=np.log10(el_dens), method="nearest"
                    )
                    atomdat[k] = atomdat[k].drop_vars(["log10_electron_density"])

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
        self.charge = int(charge)
        self.transition = transition
        self.wavelength = wavelength
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        atomdat["pec"] = atomdat["pec"].swap_dims({"index": "type"}).sel(
            type="excit"
        ) + atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom")
        # atomdat["pec"] = ( atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom") )
        self.atomdat_files = files
        self.exp = None
        self.sim = None

        if el_dens:
            for k in atomdat.keys():
                try:
                    atomdat[k] = atomdat[k].interp(
                        electron_density=el_dens, method="nearest"
                    )
                    atomdat[k] = atomdat[k].drop_vars(["electron_density"])
                except:
                    atomdat[k] = atomdat[k].interp(
                        log10_electron_density=np.log10(el_dens), method="nearest"
                    )
                    atomdat[k] = atomdat[k].drop_vars(["log10_electron_density"])

        self.atomdat = atomdat
