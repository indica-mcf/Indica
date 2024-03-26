"""
Quantities that can be read with the current abstract reader implementation
TODO: change the tuple to DataArray (long_name, units) - see examples in abstractreader
"""

from typing import Dict
from typing import Tuple

from indica.datatypes import DATATYPES

AVAILABLE_QUANTITIES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "get_thomson_scattering": {
        "ne": DATATYPES["electron_density"],
        "te": DATATYPES["electron_temperature"],
        "chi2": DATATYPES["chi_squared"],
    },
    "get_spectrometer": {
        "spectra": DATATYPES["spectra"],
    },
    "get_charge_exchange": {
        # "angf": ("angular_freq", "ion"),
        "vtor": DATATYPES["toroidal_rotation"],
        "ti": DATATYPES["ion_temperature"],
        "spectra": DATATYPES["spectra"],
        "fit": DATATYPES["spectra_fit"],
    },
    "get_bremsstrahlung_spectroscopy": {
        "zeff": DATATYPES["effective_charge"],
    },
    "get_helike_spectroscopy": {
        "int_w": DATATYPES["line_intensity"],
        "int_k": DATATYPES["line_intensity"],
        "int_tot": DATATYPES["line_intensity"],
        "int_n3": DATATYPES["line_intensity"],
        "te_kw": DATATYPES["electron_temperature"],
        "te_n3w": DATATYPES["electron_temperature"],
        "ti_w": DATATYPES["ion_temperature"],
        "ti_z": DATATYPES["ion_temperature"],
        "spectra": DATATYPES["spectra"],
    },
    "get_diode_filters": {
        "brightness": DATATYPES["brightness"],
    },
    "get_interferometry": {
        "ne": DATATYPES["electron_density_integrated"],
    },
    "get_equilibrium": {
        "f": DATATYPES["equilibrium_f"],
        "faxs": DATATYPES["poloidal_flux_axis"],
        "fbnd": DATATYPES["poloidal_flux_boundary"],
        "ftor": DATATYPES["toroidal_flux"],
        "psi": DATATYPES["poloidal_flux"],
        "rmji": DATATYPES["major_radius_hfs"],
        "rmjo": DATATYPES["major_radius_lfs"],
        "vjac": DATATYPES["volume_jacobian"],
        "ajac": DATATYPES["area_jacobian"],
        "rmag": DATATYPES["major_radius_magnetic_axis"],
        "rgeo": DATATYPES["major_radius_geometric_axis"],
        "rbnd": DATATYPES["major_radius_boundary"],
        "zmag": DATATYPES["z_magnetic_axis"],
        "zbnd": DATATYPES["z_boundary"],
        "ipla": DATATYPES["plasma_current"],
        "wp": DATATYPES["equilibrium_stored_energy"],
        "psin": DATATYPES["poloidal_flux_normalised"],
    },
    "get_cyclotron_emissions": {
        "te": DATATYPES["electron_temperature"],
    },
    "get_radiation": {
        "brightness": DATATYPES["brightness"],
    },
    "get_astra": {  # TODO: check all units!!!!
        "faxs": DATATYPES["poloidal_flux_axis"],
        "fbnd": DATATYPES["poloidal_flux_boundary"],
        "ftor": DATATYPES["toroidal_flux"],
        "psi": DATATYPES["poloidal_flux"],
        "rmji": DATATYPES["major_radius_hfs"],
        "rmjo": DATATYPES["major_radius_lfs"],
        "volume": DATATYPES["volume"],
        "area": DATATYPES["area"],
        "rmag": DATATYPES["major_radius_magnetic_axis"],
        "rgeo": DATATYPES["major_radius_geometric_axis"],
        "rbnd": DATATYPES["major_radius_boundary"],
        "zmag": DATATYPES["z_magnetic_axis"],
        "zgeo": DATATYPES["z_geometric"],
        "zbnd": DATATYPES["z_boundary"],
        "ipla": DATATYPES["plasma_current"],
        "wp": DATATYPES["total_stored_energy"],
        "psin": DATATYPES["poloidal_flux_normalised"],
        "upl": DATATYPES["loop_voltage"],  # V
        "wth": DATATYPES["equilibrium_stored_energy"],
        "wtherm": DATATYPES["thermal_stored_energy"],
        "wfast": DATATYPES["fast_ion_stored_energy"],
        "j_bs": DATATYPES["bootstrap_current_density"],  # MA/m2
        "j_nbi": DATATYPES["nbi_current_density"],  # MA/m2
        "j_oh": DATATYPES["ohmic_current_density"],  # MA/m2
        "j_tot": DATATYPES["total_current_density"],  # MA/m2
        "ne": DATATYPES["electron_density"],  # 10^19 m^-3
        "ni": DATATYPES["ion_density"],  # 10^19 m^-3
        "nf": DATATYPES["fast_ion_density"],  # 10^19 m^-3
        "omega_tor": DATATYPES["toroidal_angular_frequency"],  # 1/s
        "q_oh": DATATYPES["ohmic_power_density"],  # MW/m3
        "q_nbi": DATATYPES["nbi_power_density"],  # MW/m3
        "sbm": DATATYPES["nbi_particle_source"],  # 10^19/m^3/s
        "swall": DATATYPES["wall_particle_source"],  # 10^19/m^3/s
        "stot": DATATYPES["total_particle_source"],  # 10^19/m^3/s
        "te": DATATYPES["electron_temperature"],  # keV
        "ti": DATATYPES["ion_temperature"],  # keV
        "zeff": DATATYPES["effective_charge"],
        "p": DATATYPES["total_pressure"],  #
        "pblon": DATATYPES["parallel_fast_particle_pressure"],  #
        "pbper": DATATYPES["perpendicular_fast_particle_pressure"],  #
        "pnb": DATATYPES["injected_nbi_power"],  # W
        "pabs": DATATYPES["absorbed_nbi_power"],  # W
        "p_oh": DATATYPES["absorbed_ohmic_power"],  # W
        "q": DATATYPES["safety_factor"],  #
        "sigmapar": DATATYPES["parallel_conductivity"],  # 1/(Ohm*m)
        "nn": DATATYPES["thermal_neutral_density"],  # 10^19/m^3
        "niz1": DATATYPES["impurity_density"],  # what element?
        "niz2": DATATYPES["impurity_density"],  # what element?
        "niz3": DATATYPES["impurity_density"],  # what element?
    },
}
