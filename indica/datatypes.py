"""Prototype of how information on different data types can be stored.

"""

from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


GeneralDataType = str
SpecificDataType = str

#: Information on the general datatypes, stored in a dictionary. Keys
#  are the names of the datatypes, values are tuples where the first
#  element is a description and the second is the units for the data
#  (empty if unitless).
GENERAL_DATATYPES: Dict[GeneralDataType, Tuple[str, str]] = {
    "angular_freq": (
        "Angular speed at which a species of ion completes a loop of the Tokamak",
        "\rad s^{-1}",
    ),
    "asymmetry": (
        "Parameter describing asymettry between quantities on HFS and LFS",
        "",
    ),
    "concentration": ("Portion of nuclei which are the given type of ion", "%",),
    "effective_charge": (
        "Ratio of positive ion charge to electron charge in plasma",
        "",
    ),
    "emissivity": ("Radiation power produced per unit volume of space", "W m^{-3}"),
    "f_value": (
        "Product of toroidal magnetic field strength and major radius",
        "Wb m",
    ),
    "ion_coeffs": ("Effective ionisation coefficients", "m^3 s^{-1}"),
    "line_emissions": ("Line emissions from excitation", "W m^3"),
    "luminous_flux": (
        "Radiation power received per unit area at some point",
        "W m^{-2}",
    ),
    "magnetic_flux": ("Unnormalised poloidal component of magnetic flux", "Wb/2\\pi"),
    "major_rad": (
        "Horizontal position within the tokamak along the major radius",
        "m",
    ),
    "minor_rad": ("Distance of a point from the magnetic axis", "m"),
    "norm_flux_pol": (
        "Square root of normalised poloidal component of magnetic flux",
        "",
    ),
    "norm_flux_tor": (
        "Square root of normalised toroidal component of magnetic flux",
        "",
    ),
    "number_density": ("Number of particles per cubic metre", "m^{-3}"),
    "temperature": ("Thermal temperature of some particals", "eV"),
    "toroidal_flux": ("Unnormalised toroidal component of magnetic flux", "Wb"),
    "recomb_coeffs": ("Effective recombination coefficients", "m^3 s^{-1}"),
    "recomb_emissions": ("Emissions from recombination and bremsstrahlung", "W m^3"),
    "sxr_line_emissions": ("SXR-filtered line emissions from excitation", "W m^3"),
    "sxr_recomb_emissions": (
        "SXR-filtered emissions from recombination and bremsstrahlung",
        "W m^3",
    ),
    "vol_jacobian": (
        "Derivative of enclosed volume with respect to normalised poloidal flux",
        "m^3",
    ),
    "z": ("Vertical position from mid-plane of Tokamak", "m"),
}

#: A dictionary containing information on what the general datatype is
#  applied to. This could be a type of ion, subatomic particle,
#  etc. The key is a designator for the specific datatype and the
#  value is a description.
SPECIFIC_DATATYPES: Dict[SpecificDataType, str] = {
    "bolometric": "All wavelengths of radiation",
    "beryllium": "Beryllium ions in plasma",
    "electrons": "Electron gas in plasma",
    "hfs": "High flux surface",
    "lfs": "Low flux surface",
    "mag_axis": "Magnetic axis for equilibrium in tokamak",
    "nickle": "Nickle ions in plasma",
    "plasma": "The plasma as a whole",
    "separatrix": "Sepeparatrix surface for equilibrium in tokamak",
    "sxr": "Soft X-rays",
    "tungston": "Tungston ions in plasma",
}


#: A mapping between ADAS datatypes for ADF11 data and the general
# datatype used by indica.
ADF11_GENERAL_DATATYPES: Dict[str, GeneralDataType] = {
    "scd": "ion_coeff",
    "rcd": "recomb_coeffs",
    "plt": "line_emissions",
    "plsx": "sxr_line_emissions",
    "prb": "recomb_emissions",
    "prsx": "sxr_recomb_emissions",
}


#: A list of the chemical elements, up to Tungston, ordered by atomic
# number. Can be useful to be able to look up the name.
ORDERED_ELEMENTS: List[SpecificDataType] = [
    "",
    "hydrogen",
    "helium",
    "lithium",
    "beryllium",
    "boron",
    "carbon",
    "nitrogen",
    "oxygen",
    "fluorine",
    "neon",
    "sodium",
    "magnesium",
    "aluminium",
    "silicon",
    "phosphorus",
    "sulphur",
    "chlorine",
    "argon",
    "potassium",
    "calcium",
    "scandium",
    "titanium",
    "vanadium",
    "chromium",
    "magnanese",
    "iron",
    "cobalt",
    "nickel",
    "copper",
    "zinc",
    "gallium",
    "germanium",
    "arsenic",
    "selenium",
    "bromine",
    "krypton",
    "rubidium",
    "strontium",
    "yttrium",
    "zirconium",
    "niobium",
    "molybdenum",
    "technetium",
    "ruthenium",
    "rhodium",
    "palladium",
    "silver",
    "cadmium",
    "indium",
    "tin",
    "antimony",
    "tellurium",
    "iodine",
    "xenon",
    "cesium",
    "barium",
    "lanthanum",
    "cerium",
    "praseodymium",
    "neodymium",
    "promethium",
    "samarium",
    "europium",
    "gadolinium",
    "terbium",
    "dysprosium",
    "holmium",
    "erbium",
    "thulium",
    "ytterbium",
    "lutetium",
    "hafnium",
    "tantalum",
    "tungston",
]


#: Dictionary describing which general datatypes are valid for each specific
#  datatype.
COMPATIBLE_DATATYPES: Dict[SpecificDataType, List[GeneralDataType]] = defaultdict(
    lambda: [
        "angular_freq",
        "concentration",
        "effective_charge",
        "number_density",
        "temperature",
    ],
    {
        "bolometric": ["luminous_flux"],
        "electrons": ["angular_freq", "number_density", "temperature"],
        "hfs": ["major_rad", "z"],
        "lfs": ["major_rad", "z"],
        "mag_axis": ["magnetic_flux", "major_rad", "minor_rad", "z"],
        "plasma": [
            "angular_freq",
            "effective_charge",
            "magnetic_flux",
            "norm_flux_pol",
            "norm_flux_tor",
            "number_density",
            "temperature",
            "toroidal_flux",
            "vol_jacobian",
        ],
        "separatrix": ["magnetic_flux", "major_rad", "minor_rad", "z"],
        "sxr": ["luminous_flux"],
    },
)


#: The specific datatypes corresponding to an element/ion in the plasma
ELEMENTS = set(SPECIFIC_DATATYPES) - set(COMPATIBLE_DATATYPES)

#: Structure for type information for :py:class:`xarray.DataArray` objects.
ArrayType = Tuple[GeneralDataType, Optional[SpecificDataType]]

#: Structure for type information for :py:class:`xarray.Dataset` objects.
DatasetType = Tuple[SpecificDataType, Dict[str, GeneralDataType]]

DataType = Union[ArrayType, DatasetType]


class DatatypeWarning(Warning):
    """A Warning produced when some class uses a datatype which has not been
    defined in this module.

    """

    pass
