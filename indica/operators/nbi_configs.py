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

PLASMA_INTERP_GRID_SETTINGS = {
    'rmin': 11,
    'rmax': 99,
    'zmin': -50,
    'zmax': 50,
    'nr': 200,
    'nz': 100,
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


def get_rfx_geo():
    """

    """
    rfx = {}

    rfx["name"] = "rfx"
    rfx["shape"] = 2
    rfx["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    rfx["src"] = 100 * np.array([-2.0199, -2.6323, 0.0])
    tangency = 100 * np.array([0.2985, -0.2955, 0.0])
    rfx["axis"] = (tangency - rfx["src"]) / np.linalg.norm(tangency - rfx["src"])
    rfx["widy"] = 17.2
    rfx["widz"] = 17.2
    # rfx["widy"] = 1. # narrow beams
    # rfx["widz"] = 1. # narrow beams
    rfx["divy"] = np.array([0.014, 0.014, 0.014])
    rfx["divz"] = np.array([0.014, 0.014, 0.014])
    # rfx["divy"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    # rfx["divz"] = np.array([0.0014,0.0014,0.0014]) # narrow beams
    rfx["focy"] = 160.0
    rfx["focz"] = 160.0
    # rfx["focy"] = 300.0 # narrow beams
    # rfx["focz"] = 300.0 # narrow beams
    rfx["naperture"] = 0

    ## Pencil-like
    #rfx["widy"] = 1.0
    #rfx["widz"] = 1.0
    #rfx["divy"] = np.array([0.001,0.001,0.001])
    #rfx["divz"] = np.array([0.001,0.001,0.001])

    return rfx


def get_hnbi_geo():
    """

    """
    hnbi = {}

    hnbi["name"] = "hnbi"
    hnbi["shape"] = 2
    hnbi["data_source"] = "RFX DNBI & HNBI - 18062019 - VER. 2.pdf"
    hnbi["src"] = 100 * np.array([3.322, 3.945, 0.0])
    tangency = 100 * np.array([-0.2985, 0.2955, 0.0])
    hnbi["axis"] = (tangency - hnbi["src"]) / np.linalg.norm(tangency - hnbi["src"])
    #hnbi["widy"] = 25.0
    #hnbi["widz"] = 25.0
    hnbi["widy"] = 12.5  # numbers from Jari on 29/06/23 via teams
    hnbi["widz"] = 12.5
    hnbi["divy"] = np.array([0.014, 0.014, 0.014])
    hnbi["divz"] = np.array([0.014, 0.014, 0.014])
    #hnbi["focy"] = 420.0
    #hnbi["focz"] = 420.0
    hnbi["focy"] = 355.0
    hnbi["focz"] = 355.0
    hnbi["naperture"] = 0
    # inputs["pinj"] = 0.6
    # inputs["einj"] = 55.0
    # inputs["current_fractions"] = np.array([0.64,0.25,0.11])

    ## Pencil-like
    #hnbi["widy"] = 1.0
    #hnbi["widz"] = 1.0
    #hnbi["divy"] = np.array([0.001,0.001,0.001])
    #hnbi["divz"] = np.array([0.001,0.001,0.001])

    return hnbi
