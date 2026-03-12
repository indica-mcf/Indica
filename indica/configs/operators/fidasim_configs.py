from pathlib import Path

import numpy as np

# Paths & environment: file locations for FIDASIM install and output.
FIDASIM_ROOT = Path.home() / ".Indica" / "fidasim"
FIDASIM_BASE_DIR = str(FIDASIM_ROOT / "FIDASIM-2.0.0")
FIDASIM_OUTPUT_DIR = str(FIDASIM_ROOT / "output")
FIDASIM_BIN_PATH = str(Path(FIDASIM_BASE_DIR) / "fidasim")
FIDASIM_FI_DIST_FILE = str(
    FIDASIM_ROOT / "dists" / "9188_150_rfx" / "a5fidasim_distribution.h5"
)

# Simulation switches: enable/disable individual fidasim capabilities
SIMULATION_SWITCHES = {
    "calc_npa": 0,
    "calc_pnpa": 0,
    "calc_brems": 1,
    "calc_bes": 1,
    "calc_fida": 0,
    "calc_pfida": 0,
    "calc_birth": 1,
    "calc_dcx": 1,
    "calc_halo": 0,
    "calc_cold": 0,
    "calc_neutron": 0,
    "calc_fida_wght": 1,
    "calc_npa_wght": 1,
    "dump_dcx": 1,
}

# Plasma grid: interpolation bounds and resolution for building the FIDASIM plasma grid.
PLASMA_INTERP_GRID_SETTINGS = {
    "rmin": 11,
    "rmax": 99,
    "zmin": -50,
    "zmax": 50,
    "nr": 200,
    "nz": 100,
}

# Monte Carlo settings: high-resolution (fine) particle counts.
MC_SETTINGS_FINE = {
    "n_fida": 5000000,
    "n_pfida": 5000000,
    "n_npa": 5000000,
    "n_pnpa": 5000000,
    "n_nbi": 500000,
    "n_halo": 500000,
    "n_dcx": 500000,
    "n_birth": 10000,
}

# Monte Carlo settings: low-resolution (coarse) particle counts.
MC_SETTINGS_COARSE = {
    "n_fida": 5000000,
    "n_pfida": 5000000,
    "n_npa": 5000000,
    "n_pnpa": 5000000,
    "n_nbi": 50000,
    "n_halo": 5000,
    "n_dcx": 5000,
    "n_birth": 10000,
}

# Spectral grid: wavelength range and resolution for synthetic spectra.
WAVELENGTH_GRID_SETTINGS = {
    "lambdamin": 647.0,
    "lambdamax": 667.0,
    "nlambda": 2000,
}

# Weight-function grid: resolution for precomputed weights (if enabled).
WEIGHT_FUNCTION_SETTINGS = {
    "ne_wght": 50,
    "np_wght": 50,
    "nphi_wght": 100,
    "emax_wght": 100.0,
    "nlambda_wght": 1000,
    "lambdamin_wght": 647.0,
    "lambdamax_wght": 667.0,
}


# Build general settings: run metadata and result locations.
def build_general_settings(shot, time, runid, beam_save_dir, fida_dir):
    return {
        "device": "ST-40",
        "shot": shot,
        "time": time,
        "runid": runid,
        "comment": "test",
        "result_dir": beam_save_dir,
        "tables_file": fida_dir + "/tables/atomic_tables.h5",
    }


# Build NBI settings: beam energy/power/species mix used by FIDASIM.
def build_nbi_settings(st40_beams):
    return {
        "einj": st40_beams["einj"],
        "pinj": st40_beams["pinj"],
        "current_fractions": np.array((st40_beams["current_fractions"])),
        "ab": st40_beams["ab"],
    }


# Build plasma settings: ion mass and impurity charge state.
def build_plasma_settings(plasma_ion_amu, imp_charge):
    return {
        "ai": plasma_ion_amu,
        "impurity_charge": imp_charge,
    }
