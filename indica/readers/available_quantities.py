"""
Quantities read by each reader method of the abstractreader class.

Dictionary structure:
    {"Name of the reader method":
        "Quantity name" :{
            ("Datatype identified",
                ["Coord1 quantity name", "Coord2  quantity name", ...])
        }

For the Datatype identifier string see indica/datatypes.py.

"""

from typing import Dict, Tuple

AVAILABLE_QUANTITIES: Dict[str, Dict[str, Tuple[str, list]]] = {
    "get_thomson_scattering": {
        "t":("time", []),
        "channel":("channel", []),
        "x":("x", []),
        "y":("y", []),
        "z":("z", []),
        "R":("R", []),
        "ne": ("electron_density", ["t", "channel"]),
        "te": ("electron_temperature", ["t", "channel"]),
        "chi2": ("chi_squared", ["t", "channel"]),
    },
    "get_spectrometer": {
        "t":("time", []),
        "channel":("channel", []),
        "location":("location", []),
        "direction":("direction", []),
        "wavelength":("wavelength", []),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
    },
    "get_charge_exchange": {
        "t":("time", []),
        "channel":("channel", []),
        "wavelength":("wavelength", []),
        "x":("x", []),
        "y":("y", []),
        "z":("z", []),
        "R":("R", []),
        "location":("location", []),,
        "direction":("direction", []),
        "vtor": ("toroidal_rotation", ["t", "channel"]),
        "ti": ("ion_temperature", ["t", "channel"]),
        "conc": ("concentration", ["t", "channel"]),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
        "fit": ("spectra_fit", ["t", "channel", "wavelength"]),
    },
    "get_helike_spectroscopy": {
        "t":("time", []),
        "channel":("channel", []),
        "wavelength":("wavelength", []),
        "location":("location", []),,
        "direction":("direction", []),
        "ti_w": "ion_temperature",
        "ti_z": "ion_temperature",
        "te_n3w": "electron_temperature",
        "te_kw": "electron_temperature",
        "int_w": "line_intensity",
        "int_k": "line_intensity",
        "int_tot": "line_intensity",
        "int_n3": "line_intensity",
        "raw_spectra": ("raw_spectra", ["t", "channel", "wavelength"]),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
        "background": ("intensity", ["t", "channel", "wavelength"]),
    },
    "get_profile_fits": {
        "t":("time", []),
        "channel":("channel", []),
        "R":("R", []),
        "z":("z", []),
        "rhop":("rho_poloidal", []),
        "Rshift":("major_radius_shift", ["t"]),
        "ne_rho": ("electron_density", ["t", "rhop"]),
        "te_rho": ("electron_temperature", ["t", "rhop"]),
        "pe_rho": ("electron_pressure", ["t", "rhop"]),
        "ne_R": ("electron_density", ["t", "R"]),
        "te_R": ("electron_temperature", ["t", "R"]),
        "pe_R": ("electron_pressure", ["t", "R"]),
        "ne_data": ("electron_density", ["t", "channel"]),
        "te_data": ("electron_temperature", ["t", "channel"]),
        "pe_data": ("electron_pressure", ["t", "channel"]),
        "R_shift": ("major_radius_shift", ["t"]),
    },
    "get_diode_filters": {
        "t":("time", []),
        "channel":("channel", []),
        "label":("label", []),
        "location":("location", []),,
        "direction":("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_interferometry": {
        "t":("time", []),
        "channel":("channel", []),
        "location":("location", []),,
        "direction":("direction", []),
        "ne": ("electron_density_integrated", ["t"]),
    },
    "get_equilibrium": {
        "t":("time", []),
        "psin":("poloidal_flux_normalised", []),
        "index":("index", []),
        "R":("R", []),
        "z":("z", []),
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
        "t":("time", []),
        "channel":("channel", []),
        "label":("label", []),
        "location":("location", []),
        "direction":("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_zeff": {
        "t":("time", []),
        "rhop":("rho_poloidal", []),
        "Rshift":("major_radius_shift", ["t"]),
        "zeff_avrg": ("effective_charge", ["t"]),
        "zeff_hi": ("effective_charge", ["t"]),
        "zeff_low": ("effective_charge", ["t"]),
        "zeff": ("effective_charge", ["t", "rhop"]),
        "R_shift": ("major_radius_shift", ["t"]),
    },
    "get_astra": {  # TODO: check it all!!!!!!!!!!!!!!!!
        "t":("time", []),
        "psin":("poloidal_flux_normalised", []),
        "index":("index", []),
        "psi_r":("R", []),
        "psi_z":("z", []),
        "f": ("equilibrium_f", ["t", "psin"]),
        "ftor": ("toroidal_flux", ["t", "psin"]),
        "psi": ("poloidal_flux", ["t", "z", "R"]),
        "rmji": ("major_radius_hfs", ["t", "psin"]),
        "rmjo": ("major_radius_lfs", ["t", "psin"]),
        "volume": ("volume", ["t", "psin"]),
        "area": ("area", ["t", "psin"]),
        "rbnd": ("major_radius_boundary", ["t", "index"]),
        "zbnd": ("z_boundary", ["t", "index"]),
        "faxs": ("poloidal_flux_axis", ["t"]),
        "fbnd": ("poloidal_flux_boundary", ["t"]),
        "rgeo": ("major_radius_geometric", ["t"]),
        "zgeo": ("z_geometric", ["t"]),
        "rmag": ("major_radius_magnetic_axis", ["t"]),
        "zmag": ("z_magnetic_axis", ["t"]),
        "ipla": ("plasma_current", ["t"]),
        "wp": ("equilibrium_stored_energy", ["t"]),
        # "upl": "loop_voltage",  # V
        # "wth": "equilibrium_stored_energy",
        # "wtherm": "thermal_stored_energy",
        # "wfast": "fast_ion_stored_energy",
        # "j_bs": "bootstrap_current_density",  # MA/m2
        # "j_nbi": "nbi_current_density",  # MA/m2
        # "j_oh": "ohmic_current_density",  # MA/m2
        # "j_tot": "total_current_density",  # MA/m2
        # "ne": "electron_density",  # 10^19 m^-3
        # "ni": "ion_density",  # 10^19 m^-3
        # "nf": "fast_ion_density",  # 10^19 m^-3
        # "omega_tor": "toroidal_angular_frequency",  # 1/s
        # "q_oh": "ohmic_power_density",  # MW/m3
        # # "qnbi": "nbi_power_density_ions",  # MW/m3
        # # "qnbe": "nbi_power_density_electrons",  # MW/m3
        # "sbm": "nbi_particle_source",  # 10^19/m^3/s
        # "swall": "wall_particle_source",  # 10^19/m^3/s
        # "stot": "total_particle_source",  # 10^19/m^3/s
        # "te": "electron_temperature",  # keV
        # "ti": "ion_temperature",  # keV
        # "zeff": "effective_charge",
        # "p": "total_pressure",  #
        # "pblon": "parallel_fast_ion_pressure",  #
        # "pbper": "perpendicular_fast_ion_pressure",  #
        # "pnb": "injected_nbi_power",  # W
        # "pabs": "absorbed_nbi_power",  # W
        # "p_oh": "absorbed_ohmic_power",  # W
        # "q": "safety_factor",  #
        # "sigmapar": "parallel_conductivity",  # 1/(Ohm*m)
        # "nn": "thermal_neutral_density",  # 10^19/m^3
        # "niz1": "impurity_density",  # what element?
        # "niz2": "impurity_density",  # what element?
        # "niz3": "impurity_density",  # what element?
    },
}
