import os
from pathlib import Path

# Paths & environment: file locations for FIDASIM install and output.
# FOR FIDASIM INSTALLATION:
# https://d3denergetic.github.io/FIDASIM/page/01_getting_started/01_install.html
# Keep all local FIDASIM inputs/outputs under lowercase ~/.indica.
# FIDASIM installation path must be provided via FIDASIM_DIR.
FIDASIM_ROOT = Path.home() / ".indica" / "fidasim"
FIDASIM_BASE_DIR = os.environ.get("FIDASIM_DIR", "")
FIDASIM_OUTPUT_DIR = str(FIDASIM_ROOT / "output")
FIDASIM_BIN_PATH = str(Path(FIDASIM_BASE_DIR) / "fidasim") if FIDASIM_BASE_DIR else ""
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

# Plasma grid: resolution for building the FIDASIM plasma grid.
# Interpolation bounds defined by LineOfSightTransform._machine_dims
PLASMA_INTERP_GRID_SETTINGS = {
    "nr": 200,
    "nz": 100,
}

# Monte Carlo settings: high-resolution (fine) particle counts.
MC_SETTINGS_FINE = {
    "n_fida": 1000000,
    "n_pfida": 1000000,
    "n_npa": 1000000,
    "n_pnpa": 1000000,
    "n_nbi": 100000,
    "n_halo": 100000,
    "n_dcx": 100000,
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
