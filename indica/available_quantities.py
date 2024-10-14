"""
Quantities read by each reader method of the abstractreader class.

Dictionary structure:
    {"Name of the reader method":
        "Quantity name" :{
            ("Datatype identified",
                ["Coord1 quantity name", "Coord2  quantity name", ...])
        }

For the Datatype identifier string see indica/datatypes.py.

Dimensions (having Coordinates == []) names
must be equal to their datatype identifier

"""

from typing import Dict
from typing import Tuple

READER_QUANTITIES: Dict[str, Dict[str, Tuple[str, list]]] = {
    "get_thomson_scattering": {
        "t": ("t", []),
        "channel": ("channel", []),
        "x": ("x", []),
        "y": ("y", []),
        "z": ("z", []),
        "R": ("R", []),
        "ne": ("electron_density", ["t", "channel"]),
        "te": ("electron_temperature", ["t", "channel"]),
        "chi2": ("chi_squared", ["t", "channel"]),
    },
    "get_spectrometer": {
        "t": ("t", []),
        "channel": ("channel", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "wavelength": ("wavelength", []),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
    },
    "get_charge_exchange": {
        "t": ("t", []),
        "channel": ("channel", []),
        "wavelength": ("wavelength", []),
        "x": ("x", []),
        "y": ("y", []),
        "z": ("z", []),
        "R": ("R", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "vtor": ("toroidal_rotation", ["t", "channel"]),
        "ti": ("ion_temperature", ["t", "channel"]),
        "conc": ("concentration", ["t", "channel"]),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
        "fit": ("spectra_fit", ["t", "channel", "wavelength"]),
    },
    "get_helike_spectroscopy": {
        "t": ("t", []),
        "channel": ("channel", []),
        "wavelength": ("wavelength", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "ti_w": ("ion_temperature", ["t"]),
        "ti_z": ("ion_temperature", ["t"]),
        "te_n3w": ("electron_temperature", ["t"]),
        "te_kw": ("electron_temperature", ["t"]),
        "int_w": ("line_intensity", ["t"]),
        "int_k": ("line_intensity", ["t"]),
        "int_tot": ("line_intensity", ["t"]),
        "int_n3": ("line_intensity", ["t"]),
        "spectra_raw": ("spectra_raw", ["t", "wavelength"]),
        "spectra": ("spectra", ["t", "wavelength"]),
        "background": ("intensity", ["t"]),
    },
    "get_profile_fits": {
        "t": ("t", []),
        "channel": ("channel", []),
        "rhop_fit": ("rhop_fit", []),
        "R_fit": ("R_fit", []),
        "rhop": ("rhop", []),
        "R": ("R", []),
        "z": ("z", []),
        "Rshift": ("major_radius_shift", ["t"]),
        "ne_rho": ("electron_density", ["t", "rhop_fit"]),
        "te_rho": ("electron_temperature", ["t", "rhop_fit"]),
        "pe_rho": ("electron_pressure", ["t", "rhop_fit"]),
        "ne_R": ("electron_density", ["t", "R_fit"]),
        "te_R": ("electron_temperature", ["t", "R_fit"]),
        "pe_R": ("electron_pressure", ["t", "R_fit"]),
        "ne_data": ("electron_density", ["t", "channel"]),
        "te_data": ("electron_temperature", ["t", "channel"]),
        "pe_data": ("electron_pressure", ["t", "channel"]),
        "R_shift": ("major_radius_shift", ["t"]),
    },
    "get_diode_filters": {
        "t": ("t", []),
        "channel": ("channel", []),
        "label": ("label", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_interferometry": {
        "t": ("t", []),
        "channel": ("channel", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "ne": ("electron_density_integrated", ["t"]),
    },
    "get_equilibrium": {
        "t": ("t", []),
        "psin": ("psin", []),
        "index": ("index", []),
        "R": ("R", []),
        "z": ("z", []),
        "f": ("equilibrium_f", ["t", "psin"]),
        "ftor": ("toroidal_flux", ["t", "psin"]),
        "psi": ("poloidal_flux", ["t", "z", "R"]),
        "rmji": ("major_radius_hfs", ["t", "psin"]),
        "rmjo": ("major_radius_lfs", ["t", "psin"]),
        "vjac": ("volume_jacobian", ["t", "psin"]),
        "ajac": ("area_jacobian", ["t", "psin"]),
        "rbnd": ("major_radius_boundary", ["t", "index"]),
        "zbnd": ("z_boundary", ["t", "index"]),
        "faxs": ("poloidal_flux_axis", ["t"]),
        "fbnd": ("poloidal_flux_boundary", ["t"]),
        "rgeo": ("major_radius_geometric", ["t"]),
        "rmag": ("major_radius_magnetic_axis", ["t"]),
        "zmag": ("z_magnetic_axis", ["t"]),
        "ipla": ("plasma_current", ["t"]),
        "wp": ("equilibrium_stored_energy", ["t"]),
    },
    "get_radiation": {
        "t": ("t", []),
        "channel": ("channel", []),
        "label": ("label", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_zeff": {
        "t": ("t", []),
        "rhop": ("rhop", []),
        "Rshift": ("major_radius_shift", ["t"]),
        "zeff_avrg": ("effective_charge", ["t"]),
        "zeff_hi": ("effective_charge", ["t"]),
        "zeff_low": ("effective_charge", ["t"]),
        "zeff": ("effective_charge", ["t", "rhop"]),
        "R_shift": ("major_radius_shift", ["t"]),
    },
    # "get_astra": {  # TODO: check it all!!!!!!!!!!!!!!!!
    #     "t":("t", []),
    #     "psin":("poloidal_flux_normalised", []),
    #     "index":("index", []),
    #     "psi_r":("R", []),
    #     "psi_z":("z", []),
    #     "f": ("equilibrium_f", ["t", "psin"]),
    #     "ftor": ("toroidal_flux", ["t", "psin"]),
    #     "psi": ("poloidal_flux", ["t", "z", "R"]),
    #     "rmji": ("major_radius_hfs", ["t", "psin"]),
    #     "rmjo": ("major_radius_lfs", ["t", "psin"]),
    #     "volume": ("volume", ["t", "psin"]),
    #     "area": ("area", ["t", "psin"]),
    #     "rbnd": ("major_radius_boundary", ["t", "index"]),
    #     "zbnd": ("z_boundary", ["t", "index"]),
    #     "faxs": ("poloidal_flux_axis", ["t"]),
    #     "fbnd": ("poloidal_flux_boundary", ["t"]),
    #     "rgeo": ("major_radius_geometric", ["t"]),
    #     "zgeo": ("z_geometric", ["t"]),
    #     "rmag": ("major_radius_magnetic_axis", ["t"]),
    #     "zmag": ("z_magnetic_axis", ["t"]),
    #     "ipla": ("plasma_current", ["t"]),
    #     "wp": ("equilibrium_stored_energy", ["t"]),
    #     "upl": "loop_voltage",  # V
    #     "wth": "equilibrium_stored_energy",
    #     "wtherm": "thermal_stored_energy",
    #     "wfast": "fast_ion_stored_energy",
    #     "j_bs": "bootstrap_current_density",  # MA/m2
    #     "j_nbi": "nbi_current_density",  # MA/m2
    #     "j_oh": "ohmic_current_density",  # MA/m2
    #     "j_tot": "total_current_density",  # MA/m2
    #     "ne": "electron_density",  # 10^19 m^-3
    #     "ni": "ion_density",  # 10^19 m^-3
    #     "nf": "fast_ion_density",  # 10^19 m^-3
    #     "omega_tor": "toroidal_angular_frequency",  # 1/s
    #     "q_oh": "ohmic_power_density",  # MW/m3
    #     # "qnbi": "nbi_power_density_ions",  # MW/m3
    #     # "qnbe": "nbi_power_density_electrons",  # MW/m3
    #     "sbm": "nbi_particle_source",  # 10^19/m^3/s
    #     "swall": "wall_particle_source",  # 10^19/m^3/s
    #     "stot": "total_particle_source",  # 10^19/m^3/s
    #     "te": "electron_temperature",  # keV
    #     "ti": "ion_temperature",  # keV
    #     "zeff": "effective_charge",
    #     "p": "total_pressure",  #
    #     "pblon": "parallel_fast_ion_pressure",  #
    #     "pbper": "perpendicular_fast_ion_pressure",  #
    #     "pnb": "injected_nbi_power",  # W
    #     "pabs": "absorbed_nbi_power",  # W
    #     "p_oh": "absorbed_ohmic_power",  # W
    #     "q": "safety_factor",  #
    #     "sigmapar": "parallel_conductivity",  # 1/(Ohm*m)
    #     "nn": "thermal_neutral_density",  # 10^19/m^3
    #     "niz1": "impurity_density",  # what element?
    #     "niz2": "impurity_density",  # what element?
    #     "niz3": "impurity_density",  # what element?
    # },
}

PLASMA_QUANTITIES = {
    "public_attrs": {
        "t": ("t", []),
        "rhop": ("rhop", []),
        "index": ("index", []),
        "R": ("R", []),
        "z": ("z", []),
        "element": ("element", []),
        "impurity": ("impurity", []),
        "electron_temperature": ("electron_temperature", ["t", "rhop"]),
        "electron_density": ("electron_density", ["t", "rhop"]),
        "neutral_density": ("neutral_density", ["t", "rhop"]),
        "tau": ("tau", ["t", "rhop"]),
        "ion_temperature": ("ion_temperature", ["t", "rhop"]),
        "toroidal_rotation": ("toroidal_rotation", ["t", "rhop"]),
        "fast_ion_density": ("fast_ion_density", ["t", "rhop"]),
        "pressure_fast_parallel": ("pressure_fast_parallel", ["t", "rhop"]),
        "impurity_density": ("impurity_density", ["t", "rhop", "impurity"]),
    },
    "private_attrs": {
        "rmag": ("major_radius_magnetic_axis", ["t"]),
        "zmag": ("z_magnetic_axis", ["t"]),
        "pth": ("thermal_pressure_integral", ["t"]),
        "ptot": ("total_pressure_integral", ["t"]),
        "wth": ("thermal_stored_energy", ["t"]),
        "wp": ("total_stored_energy", ["t"]),
        "rmji": ("major_radius_hfs", ["t", "rhop"]),
        "rmjo": ("major_radius_lfs", ["t", "rhop"]),
        "rmin": ("minor_radius", ["t", "rhop"]),
        "volume": ("volume", ["t", "rhop"]),
        "area": ("area", ["t", "rhop"]),
        "pressure_fast": ("total_fast_ion_pressure", ["t", "rhop"]),
        "pressure_el": ("electron_pressure", ["t", "rhop"]),
        "zeff": ("effective_charge", ["t", "rhop", "element"]),
        "ion_density": ("ion_density", ["t", "rhop", "element"]),
        "meanz": ("mean_charge", ["t", "rhop", "element"]),
        "total_radiation": ("total_radiated_power_emission", ["t", "rhop", "element"]),
        "sxr_radiation": ("sxr_radiated_power_emission", ["t", "rhop", "element"]),
        "prad_tot": ("total_radiated_power", ["t", "element"]),
        "prad_sxr": ("sxr_radiated_power", ["t", "element"]),
        "fz": ("fractional_abundance", ["t", "rhop", "ion_charge"]),
        "lz_tot": ("total_radiation_loss_parameter", ["t", "rhop", "ion_charge"]),
        "lz_sxr": ("sxr_radiation_loss_parameter", ["t", "rhop", "ion_charge"]),
    },
}
