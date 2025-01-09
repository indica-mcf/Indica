"""
Quantities read by each reader method of the datareader class.

Dictionary structure:
    {"Name of the reader method":
        "Quantity name" :{
            ("Datatype identified",
                ["Coord1 quantity name", "Coord2  quantity name", ...])
        }

For the Datatype identifier string see indica/datatypes.py.

NB Dimensions (having Coordinates == []) must have key == datatype identifier

"""

from typing import Dict
from typing import Tuple

READER_QUANTITIES: Dict[str, Dict[str, Tuple[str, list]]] = {
    "get_thomson_scattering": {
        "t": ("t", ["t"]),
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
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "wavelength": ("wavelength", []),
        "spectra_raw": ("spectra_raw", ["t", "wavelength"]),
        "spectra": ("spectra", ["t", "channel", "wavelength"]),
    },
    "get_charge_exchange": {
        "t": ("t", ["t"]),
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
        "t": ("t", ["t"]),
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
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "rhop": ("rhop", []),
        "R": ("R_fit", []),
        "z": ("z_fit", []),
        "R_data": ("R", ["channel"]),
        "rhop_data": ("rhop", ["t", "channel"]),
        "ne_rhop": ("electron_density", ["t", "rhop"]),
        "te_rhop": ("electron_temperature", ["t", "rhop"]),
        "pe_rhop": ("electron_pressure", ["t", "rhop"]),
        "ne_R": ("electron_density", ["t", "R"]),
        "te_R": ("electron_temperature", ["t", "R"]),
        "pe_R": ("electron_pressure", ["t", "R"]),
        "ne_data": ("electron_density", ["t", "channel"]),
        "te_data": ("electron_temperature", ["t", "channel"]),
        "pe_data": ("electron_pressure", ["t", "channel"]),
        "R_shift": ("R_shift", ["t"]),
    },
    "get_diode_filters": {
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "label": ("label", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_interferometry": {
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "ne": ("electron_density_integrated", ["t", "channel"]),
    },
    "get_polarimetry": {
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "dphi": ("faraday_rotation_integrated", ["t", "channel"]),
    },
    "get_equilibrium": {
        "t": ("t", ["t"]),
        "psin": ("psin", []),
        "index": ("index", []),
        "R": ("R", []),
        "z": ("z", []),
        "rgeo": ("R_geo", ["t"]),
        "rmag": ("R_mag", ["t"]),
        "zmag": ("z_mag", ["t"]),
        "psi_axis": ("poloidal_flux_axis", ["t"]),
        "psi_boundary": ("poloidal_flux_boundary", ["t"]),
        "ipla": ("plasma_current", ["t"]),
        "wp": ("equilibrium_stored_energy", ["t"]),
        "rbnd": ("R_boundary", ["t", "index"]),
        "zbnd": ("z_boundary", ["t", "index"]),
        "f": ("equilibrium_f", ["t", "psin"]),
        "ftor": ("toroidal_flux", ["t", "psin"]),
        "rmji": ("R_hfs", ["t", "psin"]),
        "rmjo": ("R_lfs", ["t", "psin"]),
        "vjac": ("volume_jacobian", ["t", "psin"]),
        "ajac": ("area_jacobian", ["t", "psin"]),
        "psi": ("poloidal_flux", ["t", "z", "R"]),
    },
    "get_radiation": {
        "t": ("t", ["t"]),
        "channel": ("channel", []),
        "label": ("label", []),
        "location": ("location", []),
        "direction": ("direction", []),
        "brightness": ("brightness", ["t", "channel"]),
    },
    "get_zeff": {
        "t": ("t", ["t"]),
        "rhop": ("rhop", []),
        "zeff_avrg": ("effective_charge", ["t"]),
        "zeff_hi": ("effective_charge", ["t"]),
        "zeff_low": ("effective_charge", ["t"]),
        "zeff": ("effective_charge", ["t", "rhop"]),
        "R_shift": ("R_shift", ["t"]),
    },
    # "get_astra": {  # TODO: implement
    # },
}
