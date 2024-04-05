pulse = 11211

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
    "Niz1_prof.peaking",
    # "Te_prof.y0",
    # "Te_prof.wped",
    # "Te_prof.wcenter",
    # "Te_prof.peaking",
    "Ti_prof.y0",
    # "Ti_prof.wped",
    # "Ti_prof.wcenter",
    "Ti_prof.peaking",
]

phantom = False
set_ts = True,
ts_split = ""
ts_R_shift = 0
profile_params_to_update = {}
model_init = {}
revisions = {}
filters = {}
tstart = 0.05
tend = 0.06
dt = 0.01
starting_samples = 100
iterations = 1000
nwalkers = 10
stopping_criteria_factor = 0.01
mds_write = False
best = False
plot = True
run = "RUN01"
dirname = None
