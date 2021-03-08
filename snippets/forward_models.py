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

from indica.readers import ADASReader
from snippets.atomdat import get_atomdat


class spectrometer_he_like:
    def __init__(
        self,
        element="ar",
        charge=16,
        transition="(1)1(1.0)-(1)0(0.0)",
        wavelength=4.0,
        ti=500.0,
    ):
        """Models to perform forward modelling of X-ray crystal spectrometer"""
        self.reader = ADASReader()
        self.element = element
        self.charge = charge
        self.transition = transition
        self.wavelength = wavelength
        self.ti = ti
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        self.atomdat_files = files
        self.atomdat = atomdat
        self.sim = None


class spectrometer_passive_c5:
    def __init__(
        self, element="c", charge=5, transition="n=8-n=7", wavelength=5292.7, ti=330
    ):
        """Models to perform forward modelling of spectrometer measuring passive C5"""
        self.reader = ADASReader()
        self.element = element
        self.charge = charge
        self.transition = transition
        self.wavelength = wavelength
        self.ti = ti
        files, atomdat = get_atomdat(
            self.reader, element, charge, transition=transition, wavelength=wavelength
        )
        atomdat["pec"] = atomdat["pec"].swap_dims({"index": "type"}).sel(
            type="excit"
        ) + atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom")
        # atomdat["pec"] = ( atomdat["pec"].swap_dims({"index": "type"}).sel(type="recom") )
        self.atomdat_files = files
        self.atomdat = atomdat
