"""
Configuration information for reading and writing
 - read_params : pulse times to read and info to bin/interpolate on common time-axis
 - tree_names: names of all trees to be read
 - static_.. : static nodes to write to SQL
 - global_.. : global nodes to write to SQL
"""

from indica.converters.time import get_tlabels_dt

READ_PARAMS = {
    "tlim": (0.0, 0.5),  # (tstart, tend) to read in from MDS+
    "dt": 0.005,  # width of time-bin for common time base
    "overlap": 0.5,
}  # overlap of time bins (0.5 = 50%)
READ_PARAMS["t_sql"] = get_tlabels_dt(
    READ_PARAMS["tlim"][0], READ_PARAMS["tlim"][1], READ_PARAMS["dt"]
)

TREE_NAMES = [
    "efit",
    "smmh",
    "xrcs",
    # "lines",
    "cxff_pi",
    "cxff_tws_c",
    "cxff_tws_b",
    "cxqf_tws_c",
    "t1d_blom_xy1",
    "t1d_sxrc_xy2",
    "zeff_brems",
]
#   For Steve's tests:
#       Equilibrium globals - the whole lot, including Vloop, etc.
#   Profiles:
#       Equilibrium pressure profiles
#   Engineering data:
#       PCS waveforms realtime references
#       Realtime equilibrium
#       PCS state
#       NBI power
#   MHD
#       Even vs Odd powers like in JET
#   "gsfit",  "pfit"
#   "ppts",
#   "ts",
#   "dialoop",
#   "lm_avantes",
#   "diam_dd", "npa",
#   "spd",
#   "gas", "ipd",
#   "lp"

# Currently using st40_database to read these
# TODO: change to static_all = ["git_id", "version", "datetime"] once Indica reader for this complete
STATIC_ALL = ["code_version.git_id", "code_version.version", "code_version.datetime"]
STATIC_SYSTEM_SPECIFIC = {}

# These are MDS+ node names, not Indica native variables, to guarantee MDS-Trends consistency
GLOBAL_ALL = []
GLOBAL_SYSTEM_SPECIFIC = {
    "equilibrium": ["faxs", "fbnd", "rmag", "rgeo", "zmag", "zgeo", "ipla", "wp"],
    "helike_spectroscopy": [
        "ti_w",
        "ti_z",
        "te_n3w",
        "te_kw",
        "int_w",
        "int_k",
        "int_tot",
        "int_n3",
    ],
    "charge_exchange": ["ti", "vtor"],
    "interferometry": ["ne_int"],
    "radiation_inversion": ["prad"],
    "zeff": ["bremss_avrg", "zeff_avrg"],
    "thomson_scattering": ["te", "ne"],
    "profile_fits": ["te_rhop", "ne_rhop"],
}
