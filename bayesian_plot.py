import pickle

import matplotlib.pyplot as plt
import xarray as xr

from indica.converters import FluxSurfaceCoordinates
from indica.converters import ImpactParameterCoordinates

t_index = 1

# read run data
with open("stan_model_data.pkl", "rb") as pkl_file:
    pre_computed = pickle.load(pkl_file)

rho = pre_computed["rho"]
N_rho = pre_computed["N_rho"]
N_los_points = pre_computed["N_los_points"]
sxr_los_data = pre_computed["sxr_los_data"]
bolo_los_data = pre_computed["bolo_los_data"]

draws = xr.open_dataset("bayesian_samples.nc")

# plot the positive midplane density and asymmetry parameter
profiles_fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
plt.title("Full sampling Results")

ax0.errorbar(
    rho.data,
    draws.lfs_values.mean(dim=("chain", "draw")).squeeze(),
    yerr=draws.lfs_values.std(dim=("chain", "draw")).squeeze(),
    marker="x",
)
ax1.errorbar(
    rho.data,
    draws.asym_params.mean(dim=("chain", "draw")).squeeze(),
    yerr=draws.asym_params.std(dim=("chain", "draw")).squeeze(),
    marker="x",
)

ax0.set_ylabel("density profile")
ax1.set_ylabel("asymmetry parameter")
plt.xlabel("rho")

ax0.grid(True)
ax1.grid(True)

plt.savefig("profiles.pdf")

# Pinch the equilibrium for impact parameters
# TODO: find a sensible way to do this
los_coords = pre_computed["sxr_los_data"].los_values.attrs["transform"]
equilibrium = los_coords.equilibrium
flux_coords = FluxSurfaceCoordinates(kind="poloidal")
flux_coords.set_equilibrium(equilibrium)
ip_coords = ImpactParameterCoordinates(
    los_coords, flux_coords, times=pre_computed["sxr_los_data"].los_values.t
)

sxr_fig = plt.figure()
plt.errorbar(
    ip_coords.rho_min.isel(t=t_index),
    pre_computed["sxr_los_data"].los_values.isel(t=t_index),
    yerr=pre_computed["sxr_los_data"].los_errors.isel(t=t_index),
    fmt=".",
    label="Data",
)
plt.errorbar(
    ip_coords.rho_min.isel(t=t_index),
    draws.predicted_sxr_los_vals.mean(dim=("chain", "draw")),
    yerr=draws.predicted_sxr_los_vals.std(dim=("chain", "draw")),
    label="Back-fit",
)
plt.xlim(-1, 1)
plt.grid()
plt.legend()
plt.xlabel("Rho min")
plt.title("SXR Profile Reconstruction")
plt.savefig("sxr_reconstruction.pdf")

los_coords = pre_computed["bolo_los_data"].los_values.attrs["transform"]
equilibrium = los_coords.equilibrium
flux_coords = FluxSurfaceCoordinates(kind="poloidal")
flux_coords.set_equilibrium(equilibrium)
ip_coords = ImpactParameterCoordinates(
    los_coords, flux_coords, times=pre_computed["bolo_los_data"].los_values.t
)

# Take only non-dropped rho_min values
# TODO: this is particularly disgusting
dropped_coords = set(
    pre_computed["bolo_los_data"]
    .los_values.dropped.isel(t=t_index)
    .bolo_kb5v_coords.data
)
all_coords = set(ip_coords.rho_min.isel(t=t_index).bolo_kb5v_coords.data)
selected_coords = list(all_coords - dropped_coords)
rho_min = ip_coords.rho_min.isel(t=t_index).sel(bolo_kb5v_coords=selected_coords)

bolo_fig = plt.figure()
plt.errorbar(
    rho_min,
    pre_computed["bolo_los_data"].los_values.isel(t=t_index),
    yerr=pre_computed["bolo_los_data"].los_errors.isel(t=t_index),
    fmt=".",
    label="Data",
)
plt.errorbar(
    rho_min,
    draws.predicted_bolo_los_vals.mean(dim=("chain", "draw")),
    yerr=draws.predicted_bolo_los_vals.std(dim=("chain", "draw")),
    label="Back-fit",
)
plt.xlim(-1, 1)
plt.grid()
plt.legend()
plt.xlabel("Rho min")
plt.title("BOLO Profile Reconstruction")
plt.savefig("bolo_reconstruction.pdf")
