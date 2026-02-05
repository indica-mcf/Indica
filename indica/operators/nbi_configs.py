import numpy as np

GEOMETRY_PKL_PATH = "geometry_pkl_files/TriWaSp_geometry_7los_50-77_sector1.pkl"
TE_FIDASIM_CODE_PATH = "/home/jussi.hakosalo/te-fidasim"
TE_FIDASIM_INPUT_REWRITE_FROM = "/home/bart.lomanowski/TE-fidasim/"
FIDASIM_BASE_DIR = "/home/jussi.hakosalo/fidasim/FIDASIM-2.0.0"
FIDASIM_INPUT_REWRITE_TO = FIDASIM_BASE_DIR + "/"
FIDASIM_OUTPUT_DIR = "/home/jussi.hakosalo/fidasim_output"
FIDASIM_BIN_PATH = FIDASIM_BASE_DIR + "/fidasim"
TE_FIDASIM_FI_DIST_FILE = "/home/jussi.hakosalo/te-fidasim/9188_150_rfx/a5fidasim_distribution.h5"
NBI_USER = "jussi.hakosalo"

# FIDASIM simulation settings
SIMULATION_SWITCHES = {
    'calc_npa': 0,
    'calc_pnpa': 0,
    'calc_brems': 1,
    'calc_bes': 1,
    'calc_fida': 0,
    'calc_pfida': 0,
    'calc_birth': 1,
    'calc_dcx': 1,
    'calc_halo': 1,
    'calc_cold': 0,
    'calc_neutron': 0,
    'calc_fida_wght': 1,
    'calc_npa_wght': 1,
    'dump_dcx': 1,
}

MC_SETTINGS_FINE = {
    'n_fida': 5000000,
    'n_pfida': 5000000,
    'n_npa': 5000000,
    'n_pnpa': 5000000,
    'n_nbi': 500000,
    'n_halo': 500000,
    'n_dcx': 500000,
    'n_birth': 10000,
}

MC_SETTINGS_COARSE = {
    'n_fida': 5000000,
    'n_pfida': 5000000,
    'n_npa': 5000000,
    'n_pnpa': 5000000,
    'n_nbi': 50000,
    'n_halo': 5000,
    'n_dcx': 5000,
    'n_birth': 10000,
}

WAVELENGTH_GRID_SETTINGS = {
    'lambdamin': 647.0,
    'lambdamax': 667.0,
    'nlambda': 2000,
}

WEIGHT_FUNCTION_SETTINGS = {
    'ne_wght': 50,
    'np_wght': 50,
    'nphi_wght': 100,
    'emax_wght': 100.0,
    'nlambda_wght': 1000,
    'lambdamin_wght': 647.,
    'lambdamax_wght': 667.,
}


def build_general_settings(shot, time, runid, beam_save_dir, fida_dir):
    return {
        'device': 'ST-40',
        'shot': shot,
        'time': time,
        'runid': runid,
        'comment': 'test',
        'result_dir': beam_save_dir,
        'tables_file': fida_dir + '/tables/atomic_tables.h5',
    }


def build_nbi_settings(st40_beams):
    return {
        'einj': st40_beams['einj'],
        'pinj': st40_beams['pinj'],
        'current_fractions': np.array((st40_beams['current_fractions'])),
        'ab': st40_beams['ab'],
    }


def build_plasma_settings(plasma_ion_amu, imp_charge):
    return {
        'ai': plasma_ion_amu,
        'impurity_charge': imp_charge,
    }
