# ipython environment
"""
source .venv/bin/activate.fish
ipython
%load_ext autoreload
%autoreload 2
"""

# imports

import getpass
import os

from bayes_utils import create_LOSData
from bayes_utils import LOSType
import cmdstanpy
import matplotlib.pyplot as plt
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

main_ion = "d"
high_z = "w"
zeff_el = "ne"

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
    elements=["W"],
    los_type=LOSType.SXR,
)

# compile stan model
model_file = os.path.join("emissivity.stan")
model = cmdstanpy.CmdStanModel(stan_file=model_file)

# run stan model

t_index = 1

data = {
    # Impurity densities data:
    "N_rho": N_rho,
    # Lines of sight data:
    "N_los_points": N_los_points,
    # SXR data:
    "sxr_N_los": sxr_los_data.N_los,
    # stan 1-based
    "sxr_rho_lower_indices": sxr_los_data.rho_lower_indices.isel(t=t_index) + 1,
    "sxr_rho_interp_lower_frac": sxr_los_data.rho_interp_lower_frac.isel(t=t_index),
    "sxr_R_square_diff": sxr_los_data.R_square_diff.isel(t=t_index),
    "sxr_los_values": sxr_los_data.los_values.isel(t=t_index),
    #    "los_errors": binned_camera.error.isel(t=t_index),
    "sxr_los_errors": sxr_los_data.los_errors.isel(t=t_index),
}

samples = model.sample(data=data, chains=16, parallel_chains=8)
# opt_fit = model.optimize(data=data)

draws = samples.draws_xr()

# plot fit

plt.figure()
plt.xlabel("rho")
plt.ylabel("lfs_midplane emissivity")
plt.errorbar(
    rho.data,
    draws.lfs_values.mean(dim=("chain", "draw")),
    yerr=draws.lfs_values.std(dim=("chain", "draw")),
    marker="x",
)
plt.grid()
plt.show()
