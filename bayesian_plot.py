import pickle

import matplotlib.pyplot as plt
import xarray as xr

# read run data
with open("stan_model_data.pkl", "rb") as pkl_file:
    pre_computed = pickle.load(pkl_file)

rho = pre_computed["rho"]
N_rho = pre_computed["N_rho"]
N_los_points = pre_computed["N_los_points"]
sxr_los_data = pre_computed["sxr_los_data"]
bolo_los_data = pre_computed["bolo_los_data"]

draws = xr.open_dataset("bayesian_samples_original.nc")

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
plt.title("Full sampling Results")

ax0.errorbar(
    rho.data,
    draws.lfs_values.mean(dim=("chain", "draw"))[0],
    yerr=draws.lfs_values.std(dim=("chain", "draw"))[0],
    marker="x",
)
ax1.errorbar(
    rho.data,
    draws.asym_params.mean(dim=("chain", "draw"))[0],
    yerr=draws.asym_params.std(dim=("chain", "draw"))[0],
    marker="x",
)

ax0.set_ylabel("density profile")
ax1.set_ylabel("asymmetry parameter")
plt.xlabel("rho")

ax0.grid(True)
ax1.grid(True)

plt.show()
