"""
Set of dictionaries providing standard names and units for plasma physics quantities

- UNITS default in Indica = MKS + eV for temperature + nm for wavelength
- DATATYPES = (long_name, units) to be assigned as attribute to DataArray
- ELEMENTS = [Z, A, name] of elements in the periodic table
"""

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

GeneralDataType = str
SpecificDataType = str

#: Structure for type information for :py:class:`xarray.DataArray` objects.
ArrayType = Tuple[Optional[GeneralDataType], Optional[SpecificDataType]]

#: Structure for type information for :py:class:`xarray.Dataset` objects.
DatasetType = Tuple[Optional[SpecificDataType], Dict[str, GeneralDataType]]

DataType = Union[ArrayType, DatasetType]

UNITS: dict = {
    "length": "m",
    "temperature": "eV",
    "density": "m$^{-3}$",
    "density_integrated": "m$^{-2}$",
    "brightness": "W m$^{-2}$",
    "emissivity": "W m$^{-3}$",
    "velocity": "m/s",
    "magnetic_flux": r"Wb/2$\pi$",
    "f": "Wb m",
    "current": "A",
    "energy": "J",
    "percent": "%",
    "angular_frequency": "rad/s",
    "frequency": "Hz",
    "time": "s",
    "ionisation_rate": "",
    "recombination_rate": "",
    "emission_rate": "W m$^3$",
}

DATATYPES: Dict[str, Tuple[str, str]] = {
    "time": ("Time", UNITS["time"]),
    "electron_density": ("Electron density", UNITS["density"]),
    "electron_density_integrated": ("Electron density", UNITS["density_integrated"]),
    "electron_temperature": ("Electron temperature", UNITS["temperature"]),
    "ion_temperature": ("Ion temperature", UNITS["temperature"]),
    "toroidal_rotation": ("Toroidal rotation", UNITS["velocity"]),
    "angular_freq": (
        "Angular frequency",
        UNITS["angular_frequency"],
    ),
    "emissiivity": ("Emissivity", UNITS["emissivity"]),
    "brightness": ("Brightness", UNITS["brightness"]),
    "line_intensity": ("Line intensity", UNITS["brightness"]),
    "spectra": ("Spectra", UNITS["brightness"]),
    "spectra_fit": ("Spectra (fit)", UNITS["brightness"]),
    "chi_squared": ("Chi$^2$", ""),
    "effective_charge": ("Effective charge", ""),
    "equilibrium_f": ("f", UNITS["f"]),
    "poloidal_flux": ("Poloidal flux", UNITS["magnetic_flux"]),
    "poloidal_flux_axis": ("Poloidal flux (axis)", UNITS["magnetic_flux"]),
    "poloidal_flux_boundary": ("Poloidal flux (separatrix)", UNITS["magnetic_flux"]),
    "poloidal_flux_normalised": ("Normalised poloidal flux", UNITS["magnetic_flux"]),
    "toroidal_flux": ("Toroidal flux", UNITS["magnetic_flux"]),
    "major_radius_hfs": ("R$_{HFS}$", UNITS["length"]),
    "major_radius_lfs": ("R$_{LFS}$", UNITS["length"]),
    "volume_jacobian": ("Volume Jacobian", "m$^3$"),
    "area_jacobian": ("Volume Jacobian", "m$^2$"),
    "major_radius": ("R", UNITS["length"]),
    "z": ("z", UNITS["length"]),
    "rho_poloidal": (r"$\rho_{pol}$", ""),
    "rho_toroidal": (r"$\rho_{tor}$", ""),
    "major_radius_magnetic_axis": ("R$_{mag}$", UNITS["length"]),
    "major_radius_geometric_axis": ("R$_{geo}$", UNITS["length"]),
    "major_radius_boundary": ("R$_{boundary}$", UNITS["length"]),
    "minor_radius": ("r$_{min}$", UNITS["length"]),
    "minor_radius_boundary": ("a", UNITS["length"]),
    "z_magnetic_axis": ("z$_{mag}$", UNITS["length"]),
    "z_boundary": ("z$_{boundary}$", UNITS["length"]),
    "plasma_current": ("Plasma current", UNITS["current"]),
    "stored_energy": ("Stored energy", UNITS["energy"]),
    "concentration": ("Concentration", UNITS["percent"]),
    "scd": ("SCD rate coefficient", UNITS["ionisation_rate"]),
    "acd": ("ACD rate coefficient", UNITS["recombination_rate"]),
    "ccd": ("CCD rate coefficient", UNITS["recombination_rate"]),
    "pec": ("PE coefficient", UNITS["emission_rate"]),
    "plt": ("PLT coefficient", UNITS["emission_rate"]),
    "prb": ("PRB coefficient", UNITS["emission_rate"]),
    "prc": ("PRC coefficient", UNITS["emission_rate"]),
}

ELEMENTS: dict = {
    "h": [1, 1, "hydrogen"],
    "d": [1, 2, "deuterium"],
    "t": [1, 3, "tritium"],
    "he": [2, 4, "helium"],
    "li": [3, 7, "lithium"],
    "be": [4, 9, "beryllium"],
    "b": [5, 11, "boron"],
    "c": [6, 12, "carbon"],
    "n": [7, 14, "nitrogen"],
    "o": [8, 16, "oxygen"],
    "f": [9, 19, "fluorine"],
    "ne": [10, 20, "neon"],
    "na": [11, 23, "sodium"],
    "mg": [12, 24, "magnesium"],
    "al": [13, 27, "aluminium"],
    "si": [14, 28, "silicon"],
    "p": [15, 31, "phosphorus"],
    "s": [16, 32, "sulphur"],
    "cl": [17, 35, "chlorine"],
    "ar": [18, 40, "argon"],
    "k": [19, 39, "potassium"],
    "ca": [20, 40, "calcium"],
    "sc": [21, 45, "scandium"],
    "ti": [22, 48, "titanium"],
    "v": [23, 51, "vanadium"],
    "cr": [24, 52, "chromium"],
    "mn": [25, 55, "manganese"],
    "fe": [26, 56, "iron"],
    "co": [27, 59, "cobalt"],
    "ni": [28, 59, "nickel"],
    "cu": [29, 64, "copper"],
    "zn": [30, 65, "zinc"],
    "ga": [31, 70, "gallium"],
    "ge": [32, 73, "germanium"],
    "as": [33, 75, "arsenic"],
    "se": [34, 79, "selenium"],
    "br": [35, 80, "bromine"],
    "kr": [36, 84, "krypton"],
    "rb": [37, 85, "rubidium"],
    "sr": [38, 88, "strontium"],
    "y": [39, 89, "yttrium"],
    "zr": [40, 91, "zirconium"],
    "nb": [41, 93, "niobium"],
    "mo": [42, 96, "molybdenum"],
    "tc": [43, 98, "technetium"],
    "ru": [44, 101, "ruthenium"],
    "rh": [45, 103, "rhodium"],
    "pd": [46, 106, "palladium"],
    "ag": [47, 108, "silver"],
    "cd": [48, 112, "cadmium"],
    "in": [49, 115, "indium"],
    "sn": [50, 119, "tin"],
    "sb": [51, 122, "antimony"],
    "te": [52, 128, "tellurium"],
    "i": [53, 127, "iodine"],
    "xe": [54, 131, "xenon"],
    "cs": [55, 133, "cesium"],
    "ba": [56, 137, "barium"],
    "la": [57, 139, "lanthanum"],
    "ce": [58, 140, "cerium"],
    "pr": [59, 141, "praseodymium"],
    "nd": [60, 144, "neodymium"],
    "pm": [61, 145, "promethium"],
    "sm": [62, 150, "samarium"],
    "eu": [63, 152, "europium"],
    "gd": [64, 157, "gadolinium"],
    "tb": [65, 159, "terbium"],
    "dy": [66, 163, "dysprosium"],
    "ho": [67, 165, "holmium"],
    "er": [68, 167, "erbium"],
    "tm": [69, 169, "thulium"],
    "yb": [70, 173, "ytterbium"],
    "lu": [71, 175, "lutetium"],
    "hf": [72, 178, "hafnium"],
    "ta": [73, 181, "tantalum"],
    "w": [74, 184, "tungsten"],
}


class DatatypeWarning(Warning):
    """A Warning produced when some class uses a datatype which has not been
    defined in this module.

    """

    pass
