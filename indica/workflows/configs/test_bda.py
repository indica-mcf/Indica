pulse = 11089
pulse_to_write = 43000000

diagnostics = ["xrcs", "cxff_tws_c", "ts", "efit"]
opt_quantity = ["xrcs.spectra", "cxff_tws_c.ti", "ts.ne", "ts.te"]

param_names = [
    # "Ne_prof.y1",
    # "Ne_prof.y0",
    # "Ne_prof.peaking",
    # "Ne_prof.wcenter",
    # "Ne_prof.wped",
    # "Niz1_prof.y1",
    "Niz1_prof.y0",
    # "Niz1_prof.wcenter",
    # "Niz1_prof.wped",
    # "Niz1_prof.peaking",
    # "Te_prof.y0",
    # "Te_prof.wped",
    # "Te_prof.wcenter",
    # "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    # "Ti_prof.wcenter",
    # "Ti_prof.peaking",
]

plasma_settings = dict(
        main_ion="h",
        impurities=("ar", "c"),
        impurity_concentration=(0.001, 0.005),
        n_rad=10,
    )

phantom = False
mock = True
set_ts = False
ts_split = ""
ts_R_shift = 0.02
profile_params_to_update = {}
model_init = {}
revisions = {}
filters = {}
tstart = 0.05
tend = 0.06
dt = 0.01
starting_samples = 5
iterations = 5
nwalkers = 4
stopping_criteria_factor = 0.005
sample_method="random"
stopping_criteria="mode"
burn_frac=0.00

mds_write = False
best = False
plot = False
run = "TEST"
run_info = "Test run"
dirname = None
