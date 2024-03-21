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
        "wp": DATATYPES["stored_energy"],
        "psin": DATATYPES["poloidal_flux_normalised"],
    },
    "get_cyclotron_emissions": {
        "te": DATATYPES["electron_temperature"],
    },
    "get_radiation": {
        "brightness": DATATYPES["brightness"],
    },
    "get_astra": {
        "f": ("f_value", "plasma"),
        "faxs": ("magnetic_flux_axis", "poloidal"),
        "fbnd": ("magnetic_flux_separatrix", "poloidal"),
        "ftor": ("magnetic_flux", "toroidal"),
        # "rmji": ("major_rad", "hfs"),
        # "rmjo": ("major_rad", "lfs"),
        "psi": ("magnetic_flux", "poloidal"),
        "psi_1d": ("magnetic_flux", "poloidal"),
        # "vjac": ("volume_jacobian", "plasma"),
        # "ajac": ("area_jacobian", "plasma"),
        "volume": ("volume", "plasma"),
        "area": ("area", "plasma"),
        "rmag": ("major_rad", "mag_axis"),
        "rgeo": ("major_rad", "geometric"),
        "zmag": ("z", "mag_axis"),
        "zgeo": ("z", "geometric"),
        "rbnd": ("major_rad", "separatrix"),
        "zbnd": ("z", "separatrix"),
        "ipla": ("current", "plasma"),
        "upl": (
            "voltage",
            "loop",
        ),  # Loop voltage V
        "wth": (
            "stored_energy",
            "equilibrium",
        ),
        "wtherm": (
            "stored_energy",
            "thermal",
        ),
        "wfast": (
            "stored_energy",
            "fast",
        ),  # Thermal stored energy
        "j_bs": ("current_density", "bootstrap"),  # Bootstrap current density,MA/m2
        "j_nbi": (
            "current_density",
            "neutral_beam",
        ),  # NB driven current density,MA/m2
        "j_oh": ("current_density", "ohmic"),  # Ohmic current density,MA/m2
        "j_rf": ("current_density", "rf"),  # EC driven current density,MA/m2
        "j_tot": ("current_density", "total"),  # Total current density,MA/m2
        "ne": ("density", "electron"),  # Electron density, 10^19 m^-3
        "ni": ("density", "main_ion"),  # Main ion density, 10^19 m^-3
        "nf": ("density", "fast"),
        "omega_tor": (
            "rotation_frequency",
            "toroidal",
        ),  # Toroidal rotation frequency, 1/s
        "qe": ("heat_flux", "electron"),  # electron power flux, MW
        "qi": ("heat_flux", "ion"),  # ion power flux, MW
        "qn": ("heat_flux", "total"),  # total electron flux, 10^19/s
        "qnbe": (
            "power_density_nbi",
            "electron",
        ),  # Beam power density to electrons, MW/m3
        "qnbi": ("power_density_nbi", "ion"),  # Beam power density to ions, MW/m3
        "q_oh": (
            "power_density_ohm",
            "total",
        ),  # Ohmic heating power profile, MW/m3
        "q_rf": (
            "power_density_rf",
            "electron",
        ),  # RF power density to electron,MW/m3
        "sbm": ("particle_source", "nbi"),  # Particle source from beam, 10^19/m^3/s
        "swall": (
            "particle_source",
            "wall_neutrals",
        ),  # Particle source from wall neutrals, 10^19/m^3/s
        "stot": ("particle_source", "total"),  # Total electron source,10^19/s/m3
        "te": ("temperature", "electron"),  # Electron temperature, keV
        "ti": ("temperature", "ion"),  # Ion temperature, keV
        "zeff": ("effective_charge", "plasma"),  # Effective ion charge
        "p": ("pressure", "total"),  # PRESSURE(PSI_NORM)
        "pblon": ("fast_pressure", "parallel"),
        "pbper": ("fast_pressure", "perpendicular"),
        "pnb": ("nbi", "injected_power"),  # Injected NBI power, W
        "pabs": ("nbi", "absorbed_power"),  # Absorber NBI power, W
        "p_oh": ("ohmic", "power"),  # Absorber NBI power, W
        "q": ("safety_factor", "plasma"),  # Q_PROFILE(PSI_NORM)
        "sigmapar": ("conductivity", "parallel"),  # Parallel conductivity,1/(Ohm*m)
        "nn": (
            "density",
            "thermal_neutral",
        ),  # ...missing information on what elements are used
        "niz1": (
            "density",
            "impurity",
        ),  # ...missing information on what elements are used
        "niz2": ("density", "impurity"),
        "niz3": ("density", "impurity"),
    },
}
