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
        "rad s^-1",
    ),
    "asymmetry": (
        "Parameter describing asymmetry between quantities on HFS and LFS",
        "",
    ),
    "concentration": (
        "Portion of nuclei which are the given type of ion",
        "%",
    ),
    "effective_charge": (
        "Zeff ratio of ion charge to electron charge in plasma",
        "",
    ),
    "emissivity": ("Radiation power produced per unit volume of space", "W m^-3"),
    "line_emission": ("Line emission from excitation", "W m^3"),
    "luminous_flux": (
        "Radiation power received per unit area at some point",
        "W m^-2",
    ),
    "major_rad": (
        "Horizontal position within the tokamak along the major radius",
        "m",
    ),
    "minor_rad": ("Distance of a point from the magnetic axis", "m"),
    "z": ("Vertical position from mid-plane of Tokamak", "m"),
    "vol_jacobian": (
        "Derivative of enclosed volume with respect to normalised poloidal flux",
        "m^3",
    ),
    "f_value": (
        "Product of toroidal magnetic field strength and major radius",
        "Wb m",
    ),
    "magnetic_flux": ("Poloidal component of magnetic flux", "Wb/2\\pi"),
    "norm_flux_pol": (
        "Square root of normalised poloidal component of magnetic flux",
        "",
    ),
    "norm_flux_tor": (
        "Square root of normalised toroidal component of magnetic flux",
        "",
    ),
    "toroidal_flux": ("Toroidal component of magnetic flux", "Wb"),
    "number_density": ("Number of particles per cubic metre", "m^-3"),
    "temperature": ("Thermal temperature of some particals", "eV"),
    "time": ("Time into the pulse", "s"),
    "ion_coeff": ("Effective ionisation rate coefficient", "m^3 s^-1"),
    "recomb_coeff": ("Effective recombination coefficient", "m^3 s^-1"),
    "recomb_emission": ("Emission from recombination and bremsstrahlung", "W m^3"),
    "sxr_line_emission": ("SXR-filtered line emission from excitation", "W m^3"),
    "sxr_recomb_emission": (
        "SXR-filtered emission from recombination and bremsstrahlung",
        "W m^3",
    ),
    "photon_emissivity_coefficient_ca": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_cl": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_ic": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_ls": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_llu": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_pju": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_bnd": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "photon_emissivity_coefficient_pjr": (
        "ADAS photon emissivity coefficients",
        "W m^3",
    ),
    "ionisation_rate": (
        "Effective ionisation rate coefficient of all relevant ionisation stages of \
            given impurity element",
        "m^3 s^-1",
    ),
    "recombination_rate": (
        "Effective recombination rate of all relevant ionisation stages of \
            given impurity element",
        "m^3 s^-1",
    ),
    "charge-exchange_rate": (
        "Charge exchange cross coupling coefficient of all relevant ionisation stages\
             of given impurity element",
        "m^3 s^-1",
    ),
    "line_power_coefficient": (
        "Radiated power of line emission from excitation of all relevant ionisation \
            stages of given impurity element",
        "w m^3",
    ),
    "charge-exchange_power_coefficient": (
        "Radiated power of charge exchange emission of all relevant ionisation stages \
            of given impurity element",
        "w m^3",
    ),
    "recombination_power_coefficient": (
        "Radiated power from recombination and bremsstrahlung of given impurity \
            element",
        "w m^3",
    ),
    "initial_fractional_abundance": (
        "Initial fractional abundance for given impurity element, not normalized\
            , as in the sum of all stages should equal the total number density\
                for the given impurity element",
        "m^-3",
    ),
    "fractional_abundance": (
        "Fractional abundance of all ionisation stages of \
            given impurity element (normalized)",
        "",
    ),
    "total_radiated_power_loss": (
        "Total radiated power of all ionisation stages of given impurity element",
        "W m^3",
    ),
    "impurity_concentration": (
        "Impurity concentration of given element",
        "",
    ),
    "toroidal_rotation": ("Toroidal rotation speed of the plasma", "m s^-1"),
    "mean_charge": (
        "Mean charge of given element, in units of electron charge",
        "",
    ),
    "weighting": ("Dimensionless weighting to use when fitting data.", ""),
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
    "tungsten": "Tungsten ions in plasma",
    "argon": "Argon ions in plasma",
    "impurity_element": "Chosen impurity element in plasma",
    "thermal_hydrogen": "Thermal hydrogen in plasma",
}


#: A mapping between ADAS datatypes for ADF11 data and the general
# datatype used by indica.
ADF11_GENERAL_DATATYPES: Dict[str, GeneralDataType] = {
    "scd": "ion_coeff",
    "acd": "recomb_coeff",
    "ccd": "charge_exchange_recomb_coeff",
    "plt": "line_emission",
    "prc": "charge_exchange_emission",
    "pls": "sxr_line_emission",
    "prb": "recomb_emission",
    "prs": "sxr_recomb_emission",
}

ADF12_GENERAL_DATATYPES: Dict[str, GeneralDataType] = {
    "cx": "charge_exchange_cross-section_coefficient",
}

ADF15_GENERAL_DATATYPES: Dict[str, GeneralDataType] = {
    "ca": "photon_emissivity_coefficient_ca",
    "cl": "photon_emissivity_coefficient_cl",
    "ic": "photon_emissivity_coefficient_ic",
    "ls": "photon_emissivity_coefficient_ls",
    "llu": "photon_emissivity_coefficient_llu",
    "pju": "photon_emissivity_coefficient_pju",
    "bnd": "photon_emissivity_coefficient_bnd",
    "pjr": "photon_emissivity_coefficient_pjr",
}

# Format is {str(element_symbol):
# [int(charge), int(mass of most common isotope), str(element_name)]}
ELEMENTS: Dict[SpecificDataType, List[Union[int, int, SpecificDataType]]] = {
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

#: Dictionary describing which general datatypes are valid for each specific
#  datatype.
COMPATIBLE_DATATYPES: Dict[SpecificDataType, List[GeneralDataType]] = defaultdict(
    lambda: [
        "angular_freq",
        "concentration",
        "effective_charge",
        "number_density",
        "temperature",
        "weighting",
        "ionisation_rate",
        "recombination_rate",
        "charge-exchange_rate",
        "line_power_coefficient",
        "recombination_power_coefficient",
        "charge-exchange_power_coefficient",
        "initial_fractional_abundance",
        "fractional_abundance",
        "total radiated power loss",
    ],
    {
        "bolometric": ["luminous_flux", "weighting"],
        "electrons": ["angular_freq", "number_density", "temperature", "weighting"],
        "hfs": ["major_rad", "z", "weighting"],
        "lfs": ["major_rad", "z", "weighting"],
        "mag_axis": ["magnetic_flux", "major_rad", "minor_rad", "z", "weighting"],
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
            "weighting",
            "toroidal_rotation",
        ],
        "separatrix": ["magnetic_flux", "major_rad", "minor_rad", "z", "weighting"],
        "sxr": ["luminous_flux", "weighting"],
        "thermal_hydrogen": ["number_density"],
        "impurity_element": [
            "ionisation_rate",
            "recombination_rate",
            "charge-exchange_rate",
            "line_power_coefficient",
            "recombination_power_coefficient",
            "charge-exchange_power_coefficient",
            "initial_fractional_abundance",
            "fractional_abundance",
            "total radiated power loss",
            "impurity_concentration",
            "mean_charge",
            "time",
        ],
    },
)


#: The specific datatypes corresponding to an element/ion in the plasma
SPECIFIC_ELEMENTS = set(SPECIFIC_DATATYPES) - set(COMPATIBLE_DATATYPES)

#: Structure for type information for :py:class:`xarray.DataArray` objects.
ArrayType = Tuple[Optional[GeneralDataType], Optional[SpecificDataType]]

#: Structure for type information for :py:class:`xarray.Dataset` objects.
DatasetType = Tuple[Optional[SpecificDataType], Dict[str, GeneralDataType]]

DataType = Union[ArrayType, DatasetType]


class DatatypeWarning(Warning):
    """A Warning produced when some class uses a datatype which has not been
    defined in this module.

    """

    pass
