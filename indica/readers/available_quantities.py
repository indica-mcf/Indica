"""
Quantities that can be read with the current abstract reader implementation

To each quantity is associated a datatype string corresponding to indica/datatypes.py
"""


from typing import Dict

AVAILABLE_QUANTITIES: Dict[str, Dict[str, str]] = {
    "get_thomson_scattering": {
        "ne": "electron_density",
        "te": "electron_temperature",
        "chi2": "chi_squared",
    },
    "get_spectrometer": {
        "spectra": "spectra",
    },
    "get_charge_exchange": {
        # "angf": ("angular_freq", "ion"),
        "vtor": "toroidal_rotation",
        "ti": "ion_temperature",
        "spectra": "spectra",
        "fit": "spectra_fit",
    },
    "get_bremsstrahlung_spectroscopy": {
        "zeff": "effective_charge",
    },
    "get_helike_spectroscopy": {
        "int_w": "line_intensity",
        "int_k": "line_intensity",
        "int_tot": "line_intensity",
        "int_n3": "line_intensity",
        "te_kw": "electron_temperature",
        "te_n3w": "electron_temperature",
        "ti_w": "ion_temperature",
        "ti_z": "ion_temperature",
        "spectra": "spectra",
    },
    "get_ppts": {
        "ne_rho": "electron_density",
        "te_rho": "electron_temperature",
        "pe_rho": "electron_pressure",
        "ne_R": "electron_density",
        "te_R": "electron_temperature",
        "pe_R": "electron_pressure",
        "ne_data": "electron_density",
        "te_data": "electron_temperature",
        "pe_data": "electron_pressure",
        "rshift": "major_radius_shift",
    },
    "get_diode_filters": {
        "brightness": "brightness",
    },
    "get_interferometry": {
        "ne": "electron_density_integrated",
    },
    "get_equilibrium": {
        "f": "equilibrium_f",
        "faxs": "poloidal_flux_axis",
        "fbnd": "poloidal_flux_boundary",
        "ftor": "toroidal_flux",
        "psi": "poloidal_flux",
        "rmji": "major_radius_hfs",
        "rmjo": "major_radius_lfs",
        "vjac": "volume_jacobian",
        "ajac": "area_jacobian",
        "rmag": "major_radius_magnetic_axis",
        "rgeo": "major_radius_geometric_axis",
        "rbnd": "major_radius_boundary",
        "zmag": "z_magnetic_axis",
        "zbnd": "z_boundary",
        "ipla": "plasma_current",
        "wp": "equilibrium_stored_energy",
    },
    "get_cyclotron_emissions": {
        "te": "electron_temperature",
    },
    "get_radiation": {
        "brightness": "brightness",
    },
    "get_astra": {  # TODO: check all units!!!!
        "faxs": "poloidal_flux_axis",
        "fbnd": "poloidal_flux_boundary",
        "ftor": "toroidal_flux",
        "psi": "poloidal_flux",
        "rmji": "major_radius_hfs",
        "rmjo": "major_radius_lfs",
        "volume": "volume",
        "area": "area",
        "rmag": "major_radius_magnetic_axis",
        "rgeo": "major_radius_geometric_axis",
        "rbnd": "major_radius_boundary",
        "zmag": "z_magnetic_axis",
        "zgeo": "z_geometric",
        "zbnd": "z_boundary",
        "ipla": "plasma_current",
        "wp": "total_stored_energy",
        "upl": "loop_voltage",  # V
        "wth": "equilibrium_stored_energy",
        "wtherm": "thermal_stored_energy",
        "wfast": "fast_ion_stored_energy",
        "j_bs": "bootstrap_current_density",  # MA/m2
        "j_nbi": "nbi_current_density",  # MA/m2
        "j_oh": "ohmic_current_density",  # MA/m2
        "j_tot": "total_current_density",  # MA/m2
        "ne": "electron_density",  # 10^19 m^-3
        "ni": "ion_density",  # 10^19 m^-3
        "nf": "fast_ion_density",  # 10^19 m^-3
        "omega_tor": "toroidal_angular_frequency",  # 1/s
        "q_oh": "ohmic_power_density",  # MW/m3
        "q_nbi": "nbi_power_density",  # MW/m3
        "sbm": "nbi_particle_source",  # 10^19/m^3/s
        "swall": "wall_particle_source",  # 10^19/m^3/s
        "stot": "total_particle_source",  # 10^19/m^3/s
        "te": "electron_temperature",  # keV
        "ti": "ion_temperature",  # keV
        "zeff": "effective_charge",
        "p": "total_pressure",  #
        "pblon": "parallel_fast_ion_pressure",  #
        "pbper": "perpendicular_fast_ion_pressure",  #
        "pnb": "injected_nbi_power",  # W
        "pabs": "absorbed_nbi_power",  # W
        "p_oh": "absorbed_ohmic_power",  # W
        "q": "safety_factor",  #
        "sigmapar": "parallel_conductivity",  # 1/(Ohm*m)
        "nn": "thermal_neutral_density",  # 10^19/m^3
        "niz1": "impurity_density",  # what element?
        "niz2": "impurity_density",  # what element?
        "niz3": "impurity_density",  # what element?
    },
}
