"""
Set of dictionaries providing standard names and units for plasma physics quantities

- UNITS default in Indica = MKS + eV for temperature + nm for wavelength
- DATATYPES = (long_name, units) to be assigned as attribute to DataArray
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
    "none": "",
    "length": "m",
    "temperature": "eV",
    "density": "$1/m^3$",
    "particle_source": "$1/m^3 s$",
    "density_integrated": "1/$m^2$",
    "brightness": "$W/m^2$",
    "emissivity": "$W/m^3$",
    "velocity": "m/s",
    "magnetic_flux": r"$Wb/2\pi$",
    "f": "Wb m",
    "current": "A",
    "current_density": "$A/m^2$",
    "energy": "J",
    "power": "W",
    "percent": "%",
    "angular_frequency": "rad/s",
    "frequency": "Hz",
    "time": "s",
    "ionisation_rate": "",
    "recombination_rate": "",
    "emission_rate": "W $m^3$",
    "volume_jacobian": "$m^3$",
    "area_jacobian": "$m^2$",
    "volume": "1/$m^3$",
    "area": "$1/m^2$",
    "voltage": "V",
    "pressure": "Pa",
    "conductivity": r"1/($\Ohm$ m)",
}

DATATYPES: Dict[str, Tuple[str, str]] = {
    "time": ("Time", UNITS["time"]),
    "electron_density": ("$N_e$", UNITS["density"]),
    "ion_density": ("$N_i$", UNITS["density"]),
    "fast_ion_density": ("$N_{fast}$", UNITS["density"]),
    "impurity_density": ("$N_{imp}$", UNITS["density"]),
    "thermal_neutral_density": ("$N_{neutrals, th}$", UNITS["density"]),
    "electron_density_integrated": (r"$\int N_e$", UNITS["density_integrated"]),
    "electron_temperature": ("$T_e$", UNITS["temperature"]),
    "ion_temperature": ("$T_i$", UNITS["temperature"]),
    "toroidal_rotation": ("$V_{tor}$", UNITS["velocity"]),
    "toroidal_angular_frequency": (
        r"$\omega_{tor}$",
        UNITS["angular_frequency"],
    ),
    "emissiivity": ("Emissivity", UNITS["emissivity"]),
    "brightness": ("Brightness", UNITS["brightness"]),
    "line_intensity": ("Line intensity", UNITS["brightness"]),
    "spectra": ("Spectra", UNITS["brightness"]),
    "spectra_fit": ("Spectra (fit)", UNITS["brightness"]),
    "chi_squared": (r"$\chi^2$", ""),
    "effective_charge": ("$Z_{eff}$", ""),
    "equilibrium_f": ("f", UNITS["f"]),
    "poloidal_flux": ("Poloidal flux", UNITS["magnetic_flux"]),
    "poloidal_flux_axis": ("Poloidal flux (axis)", UNITS["magnetic_flux"]),
    "poloidal_flux_boundary": ("Poloidal flux (separatrix)", UNITS["magnetic_flux"]),
    "poloidal_flux_normalised": ("Normalised poloidal flux", UNITS["magnetic_flux"]),
    "toroidal_flux": ("Toroidal flux", UNITS["magnetic_flux"]),
    "major_radius_hfs": ("$R_{HFS}$", UNITS["length"]),
    "major_radius_lfs": ("$R_{LFS}$", UNITS["length"]),
    "volume_jacobian": ("$V_{jac}$", UNITS["volume_jacobian"]),
    "area_jacobian": ("$A_{jac}$", UNITS["area_jacobian"]),
    "volume": ("Volume", UNITS["volume"]),
    "area": ("Cross-sectional area", UNITS["area"]),
    "major_radius": ("R", UNITS["length"]),
    "z": ("z", UNITS["length"]),
    "z_geometric": ("$z_{geo}$", UNITS["length"]),
    "z_magnetic_axis": ("$z_{mag}$", UNITS["length"]),
    "z_boundary": ("$z_{boundary}$", UNITS["length"]),
    "rho_poloidal": (r"$\rho_{pol}$", ""),
    "rho_toroidal": (r"$\rho_{tor}$", ""),
    "major_radius_magnetic_axis": ("$R_{mag}$", UNITS["length"]),
    "major_radius_geometric_axis": ("$R_{geo}$", UNITS["length"]),
    "major_radius_boundary": ("$R_{boundary}$", UNITS["length"]),
    "minor_radius": ("$r_{min}$", UNITS["length"]),
    "minor_radius_boundary": ("a", UNITS["length"]),
    "plasma_current": ("$I_{P}$", UNITS["current"]),
    "concentration": ("Concentration", UNITS["percent"]),
    "scd": ("SCD rate coefficient", UNITS["ionisation_rate"]),
    "acd": ("ACD rate coefficient", UNITS["recombination_rate"]),
    "ccd": ("CCD rate coefficient", UNITS["recombination_rate"]),
    "pec": ("PEC coefficient", UNITS["emission_rate"]),
    "plt": ("PLT coefficient", UNITS["emission_rate"]),
    "prb": ("PRB coefficient", UNITS["emission_rate"]),
    "prc": ("PRC coefficient", UNITS["emission_rate"]),
    "loop_voltage": ("$V_{loop}$", UNITS["voltage"]),
    "equilibrium_stored_energy": ("$W_{eq}$", UNITS["energy"]),
    "thermal_stored_energy": ("$W_{th}$", UNITS["energy"]),
    "fast_ion_stored_energy": ("$W_{fast}$", UNITS["energy"]),
    "total_stored_energy": ("$W_{tot}$", UNITS["energy"]),
    "ohmic_current_density": ("$j_{ohm}$", UNITS["current_density"]),
    "bootstrap_current_density": ("$j_{BS}$", UNITS["current_density"]),
    "nbi_current_density": ("$j_{NBI}$", UNITS["current_density"]),
    "total_current_density": ("$j_{tot}$", UNITS["current_density"]),
    "ohmic_power_density": ("$P_{ohm}$", UNITS["power"]),
    "nbi_power_density": ("$P_{nbi}$", UNITS["power"]),
    "nbi_particle_source": ("$S_{nbi}$", UNITS["particle_source"]),
    "wall_particle_source": ("$S_{wall}$", UNITS["particle_source"]),
    "total_particle_source": ("$S_{tot}$", UNITS["particle_source"]),
    "thermal_pressure": ("$P_{th}", UNITS["pressure"]),
    "parallel_fast_particle_pressure": ("$P_{fast,//}$", UNITS["pressure"]),
    "perpendicular_fast_particle_pressure": (r"$P_{fast,\perp}$", UNITS["pressure"]),
    "total_pressure": ("$P_{tot}", UNITS["pressure"]),
    "injected_nbi_power": ("$P_{NBI, inj}$", UNITS["power"]),
    "absorbed_nbi_power": ("$P_{NBI, abs}$", UNITS["power"]),
    "absorbed_ohmic_power": ("$P_{ohm, abs}$", UNITS["power"]),
    "safety_factor": ("q", UNITS["none"]),
    "parallel_conductivity": (r"$\sigma$", UNITS["conductivity"]),
}
