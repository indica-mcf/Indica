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

import cmdstanpy
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.converters import FluxMajorRadCoordinates
from indica.converters import FluxSurfaceCoordinates
from indica.converters import ImpactParameterCoordinates
from indica.converters.time import bin_to_time_labels
from indica.equilibrium import Equilibrium
from indica.readers import PPFReader
from indica.readers.selectors import use_cached_ignore_channels
from indica.utilities import coord_array

# config

pulse = 90279

N_rho = 7
# N_R = 25
N_z = 25
N_intervals = 65

main_ion = "d"
high_z = "w"
zeff_el = "ne"

server = "https://sal.jet.uk"

# coord arrays

# R = coord_array(np.linspace(1.83, 3.9, N_R), "R")
rho = coord_array(np.linspace(0, 1, N_rho), "rho_poloidal")
z = coord_array(np.linspace(-1.75, 2.0, N_z), "z")
t = coord_array(np.array([45.17, 45.85, 46.17]), "t")

# read PPF data
reader = PPFReader(
    pulse=pulse,
    tstart=float(t.isel(t=0)),
    tend=float(t.isel(t=-1)),
    server=server,
    selector=use_cached_ignore_channels,
)
reader.authenticate(getpass.getuser(), getpass.getpass())

diagnostics = {
    "efit": reader.get(uid="jetppf", instrument="eftp", revision=0),
    #    "hrts": reader.get(uid="jetppf", instrument="hrts", revision=0),
    "sxr": reader.get(uid="jetppf", instrument="sxr", revision=0),
    #    "zeff": reader.get(uid="jetppf", instrument="ks3", revision=0),
    #    "bolo": reader.get(uid="jetppf", instrument="bolo", revision=0),
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

rho_maj_radius = FluxMajorRadCoordinates(flux_coords)

# set up data

# bin camera data, drop excluded channels
binned_camera = bin_to_time_labels(
    t.data, diagnostics["sxr"]["v"].dropna(dim="sxr_v_coords")
)

x2 = np.linspace(0, 1, N_intervals)
camera = xr.Dataset(
    {"camera": binned_camera},
    {binned_camera.attrs["transform"].x2_name: x2},
    {"transform": binned_camera.attrs["transform"]},
)

rho_los_points, R_los_points = camera.indica.convert_coords(rho_maj_radius)

# TODO: work out if this is the best/right way to deal with values outside plasma:
rho_los_points = rho_los_points.clip(0, 1).fillna(1)

R_los_points_lfs_midplane, _ = flux_coords.convert_to_Rz(
    rho_los_points, xr.zeros_like(rho_los_points), rho_los_points.t
)

R_square_diff = R_los_points**2 - R_los_points_lfs_midplane**2

# TODO: check this
# since we clipped before, some values of R_los_points_lfs_midplane
# don't match R_los_points - need to clip again
R_square_diff = R_square_diff.clip(max=0)

# find indices for interpolation, subtract 1 to get lower indices
rho_indices = rho.searchsorted(rho_los_points) - 1
# clip indices because we subtracted 1 to select lower
# clip indices because there should be a value above them
rho_indices = rho_indices.clip(min=0, max=N_rho - 2)

rho_indices = xr.DataArray(
    data=rho_indices, dims=rho_los_points.dims, coords=rho_los_points.coords
)

rho_dropped = rho.drop("rho_poloidal")
lower = rho_dropped[rho_indices]
upper = rho_dropped[rho_indices + 1]

rho_interp_lower_frac = (upper - rho_los_points) / (upper - lower)

# weights:
ip_coords = ImpactParameterCoordinates(camera.attrs["transform"], flux_coords, times=t)
rho_max = ip_coords.rhomax()
impact_param, _ = camera.indica.convert_coords(ip_coords)
weights = camera.camera * (0.02 + 0.18 * np.abs(impact_param))

# compile stan model

model_file = os.path.join("emissivity.stan")
model = cmdstanpy.CmdStanModel(stan_file=model_file)

N_los = len(binned_camera.sxr_v_coords)
t_index = 1

data = {
    # Impurity densities data:
    "N_rho": N_rho,
    # Lines of sight data:
    "N_los_points": N_intervals,
    # SXR data:
    "sxr_N_los": N_los,
    "sxr_rho_lower_indices": rho_indices.isel(t=t_index) + 1,  # stan 1-based so add 1
    "sxr_rho_interp_lower_frac": rho_interp_lower_frac.isel(t=t_index),
    "sxr_R_square_diff": R_square_diff.isel(t=t_index),
    "sxr_los_values": binned_camera.isel(t=t_index),
    #    "los_errors": binned_camera.error.isel(t=t_index),
    "sxr_los_errors": weights.isel(t=t_index),
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

# testing - plot los errors
plt.figure()
plt.xlabel("sxr_v_coords")
plt.ylabel("los_values")
plt.errorbar(
    binned_camera.sxr_v_coords.data,
    binned_camera.isel(t=t_index).data,
    weights.isel(t=t_index).data,
    marker="x",
)
plt.grid()
plt.show()
