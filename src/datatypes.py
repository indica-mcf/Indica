"""Prototype of how information on different data types can be stored.

"""

#: Information on the general datatypes, stored in a dictionary. Keys
#  are the names of the datatypes, values are tuples where the first
#  element is a description and the second is the units for the data
#  (empty if unitless).
GENERAL_DATATYPES = {"angular_freq": ("Angular speed at which a species of"
                                      " ion completes a loop of the Tokamak",
                                      "\rad s^{-1}"),
                     "concentration": ("Portion of nuclei which are the given"
                                       " type of ion", "%"),
                     "effective_charge": ("Ratio of positive ion charge to "
                                          "electron charge in plasma", ""),
                     "luminous_flux": ("Radiation power received per unit "
                                       "area at some point", "W m^{-2}"),
                     "major_rad": ("Horizontal position within the tokamak "
                                   "along the major radius", "m"),
                     "number_density": ("Number of particles per cubic metre",
                                        "m^{-3}"),
                     "temperature": ("Thermal temperature of some particals",
                                     "eV"),
                     "z": ("Vertical position from mid-plane of Tokamak", "m"),
                     }

#: A dictionary containing information on what the general datatype is
#  applied to. This could be a type of ion, subatomic particle,
#  etc. The key is a designator for the specific datatype and the
#  value is a description.
SPECIFIC_DATATYPES = {"beryllium": "Beryllium ions in plasma",
                      "electrons": "Electron gas in plasma",
                      "mag_axis": "Magnetic axis for equilibrium in tokamak",
                      "nickle": "Nickle ions in plasma",
                      "plasma": "The plasma as a whole",
                      "separatrix_axis": "Sepeparatrix axis for equilibrium "
                                         "in tokamak",
                      "sxr": "Soft X-rays",
                      "tungston": "Tungston ions in plasma",
                      }


class DatatypeWarning(Warning):
    """A Warning produced when some class uses a datatype which has not been
    defined in this module.

    """
    pass
