# ipython environment
"""
source .venv/bin/activate.fish
ipython
%load_ext autoreload
%autoreload 2
"""

# imports

import getpass
import pickle
from typing import Any
from typing import Dict

from bayes_utils import create_LOSData
from bayes_utils import LOSType
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.readers import PPFReader
from indica.readers.selectors import use_cached_ignore_channels
from indica.utilities import coord_array

# config

pulse = 90279

N_rho = 7
# N_R = 25
N_z = 25
N_los_points = 65

elements = ["W"]

server = "https://sal.jet.uk"

# coord arrays

# R = coord_array(np.linspace(1.83, 3.9, N_R), "R")
rho = coord_array(np.linspace(0, 1, N_rho), "rho_poloidal")
z = coord_array(np.linspace(-1.75, 2.0, N_z), "z")
# equally spaced times to mitigate equally spaced assumption of
# half_interval in bin_to_time_labels
# TODO: raise issue
# t = coord_array(np.array([45.17, 45.85, 46.17]), "t")
t = coord_array(np.linspace(45.17, 46.17, 3), "t")

# read PPF data
reader = PPFReader(
    pulse=pulse,
    tstart=float(t.isel(t=0)),
    tend=float(t.isel(t=-1)),
    server=server,
    selector=use_cached_ignore_channels,
)
reader.authenticate("kcollie", getpass.getpass())

diagnostics = {
    "efit": reader.get(uid="jetppf", instrument="eftp", revision=0),
    "hrts": reader.get(uid="jetppf", instrument="hrts", revision=0),
    "sxr": reader.get(uid="jetppf", instrument="sxr", revision=0),
    "zeff": reader.get(uid="jetppf", instrument="ks3", revision=0),
    "bolo": reader.get(uid="jetppf", instrument="bolo", revision=0),
}
efit_equilibrium = Equilibrium(equilibrium_data=diagnostics["efit"])
for key, diag in diagnostics.items():
    for data in diag.values():
        if hasattr(data.attrs["transform"], "equilibrium"):
            del data.attrs["transform"].equilibrium
        if "efit" not in key.lower():
            data.indica.equilibrium = efit_equilibrium

# set up coordinates
flux_coords = FluxSurfaceCoordinates(kind="poloidal")
flux_coords.set_equilibrium(efit_equilibrium)

# set up data
sxr_los_data = create_LOSData(
    los_diagnostic=diagnostics["sxr"]["v"],
    los_coord_name="sxr_v_coords",
    hrts_diagnostic=diagnostics["hrts"],
    flux_coords=flux_coords,
    rho=rho,
    t=t,
    N_los_points=N_los_points,
    elements=elements,
    los_type=LOSType.SXR,
)

bolo_los_data = create_LOSData(
    los_diagnostic=diagnostics["bolo"]["kb5v"],
    los_coord_name="bolo_kb5v_coords",
    hrts_diagnostic=diagnostics["hrts"],
    flux_coords=flux_coords,
    rho=rho,
    t=t,
    N_los_points=N_los_points,
    elements=elements,
    los_type=LOSType.BOLO,
)

# write out data required for run
pre_computed: Dict[str, Any] = {}
pre_computed["sxr_los_data"] = sxr_los_data
pre_computed["bolo_los_data"] = bolo_los_data
pre_computed["N_rho"] = N_rho
pre_computed["N_los_points"] = N_los_points
pre_computed["rho"] = rho
pre_computed["t"] = t
pre_computed["elements"] = elements
with open("stan_model_data.pkl", "wb") as pkl_file:
    pickle.dump(pre_computed, pkl_file)
