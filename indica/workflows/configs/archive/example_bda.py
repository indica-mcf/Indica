from typing import Dict

pulse = 11089
pulse_to_write = 43000000

diagnostics = [
            "xrcs",
            "cxff_tws_c",
            "ts", ]
quant_to_optimise = [
                    "xrcs.intens",
                    "cxff_tws_c.ti",
                    "ts.ne",
                    "ts.te"]

param_names = [
    "impurity_density:ar.y0",
    # "impurity_density:ar.wcenter",
    # "impurity_density:ar.wped",
    # "impurity_density:ar.peaking",
    "ion_temperature.y0",
    # "ion_temperature.wped",
    "ion_temperature.wcenter",
    "ion_temperature.peaking",
]

pca_profiles = ["ion_temperature", "impurity_density:ar"]
pca_components = 3

plasma_settings = dict(
    main_ion="h",
    impurities=("ar",),
    impurity_concentration=(0.001,),
    n_rad=20,
)

phantom = False
set_ts = True
profile_params_to_update: Dict = {}

filter_coords: Dict = {"cxff_pi":
                       {"ti": ("channel", (3, 5)), "vtor": ("channel", (3, 5))},
                        "cxff_tws_c":
                        {"ti": ("channel", (0, 1)), "vtor": ("channel", (0, 1))}
                       }
model_init: Dict = {"xrcs": {"window_masks": [slice(0.394, 0.396)]}}

apply_rshift = True
tstart = 0.05
tend = 0.06
dt = 0.01
starting_samples = 100
iterations = 3000
nwalkers = 16
stopping_criteria_factor = 0.001
sample_method = "high_density"
stopping_criteria = "mode"
burn_frac = 0.20

mds_write = True
best = False
plot = True
run = "RUN01"
run_info = "Example run"
dirname = None
