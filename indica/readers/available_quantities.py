"""
Quantities that can be read with the current abstract reader implementation
"""

from typing import Dict

from ..datatypes import ArrayType


AVAILABLE_QUANTITIES: Dict[str, Dict[str, ArrayType]] = {
    "get_thomson_scattering": {
        "ne": ("number_density", "electrons"),
        "te": ("temperature", "electrons"),
        "chi2": ("chi-squared", "fit")
    },
    "get_charge_exchange": {
        "vtor": ("linear_rotation", "ion"),
        # "angf": ("angular_freq", "ion"),
        "ti": ("temperature", "ion"),
    },
    "get_bremsstrahlung_spectroscopy": {
        "zeff": ("effective_charge", "plasma"),
    },
    "get_helike_spectroscopy": {
        "int_w": ("intensity", "spectral_line"),
        "int_k": ("intensity", "spectral_line"),
        "int_tot": ("intensity", "spectral_line"),
        "int_n3": ("intensity", "spectral_line"),
        "te_kw": ("temperature", "electrons"),
        "te_n3w": ("temperature", "electrons"),
        "ti_w": ("temperature", "ions"),
        "ti_z": ("temperature", "ions"),
        "spectra": ("spectra", "passive"),
    },
    "get_diode_filters": {
        "brightness": ("luminous_flux", None),
    },
    "get_interferometry": {
        "ne": ("density", "electrons"),
    },
    "get_equilibrium": {
        "f": ("f_value", "plasma"),
        "faxs": ("magnetic_flux_axis", "poloidal"),
        "fbnd": ("magnetic_flux_separatrix", "poloidal"),
        "ftor": ("magnetic_flux", "toroidal"),
        "rmji": ("major_rad", "hfs"),
        "rmjo": ("major_rad", "lfs"),
        "psin": ("magnetic_flux_normalised", "poloidal"),
        "psi": ("magnetic_flux", "poloidal"),
        "vjac": ("volume_jacobian", "plasma"),
        "ajac": ("area_jacobian", "plasma"),
        "rmag": ("major_rad", "mag_axis"),
        "rgeo": ("major_rad", "geometric"),
        "rbnd": ("major_rad", "separatrix"),
        "zmag": ("z", "mag_axis"),
        "zbnd": ("z", "separatrix"),
        "ipla": ("current", "plasma"),
        "wp": ("energy", "plasma"),
    },
    "get_cyclotron_emissions": {
        "te": ("temperature", "electrons"),
    },
    "get_radiation": {
        "brightness": ("luminous_flux", None),
    },
    "get_astra": {
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
        "ftor": ("flux", "toroidal"),
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
        "volume": ("volume", "plasma"),  # Parallel conductivity,1/(Ohm*m)
    },
}
