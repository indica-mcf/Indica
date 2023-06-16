import os
import pickle

import cmdstanpy
import matplotlib.pyplot as plt

plt.ion()

# read run data
with open("stan_model_data.pkl", "rb") as pkl_file:
    pre_computed = pickle.load(pkl_file)

rho = pre_computed["rho"]
N_rho = pre_computed["N_rho"]
N_los_points = pre_computed["N_los_points"]
sxr_los_data = pre_computed["sxr_los_data"]
bolo_los_data = pre_computed["bolo_los_data"]

# compile stan model
model_file = os.path.join("emissivity.stan")
model = cmdstanpy.CmdStanModel(stan_file=model_file)

# run stan model

t_index = 1

data = {
    # Impurity densities data:
    "N_elements": 1,
    "N_rho": N_rho,
    # Lines of sight data:
    "N_los_points": N_los_points,
    # SXR data:
    "sxr_N_los": sxr_los_data.N_los,
    # stan 1-based
    "sxr_rho_lower_indices": sxr_los_data.rho_lower_indices.isel(t=t_index) + 1,
    "sxr_rho_interp_lower_frac": sxr_los_data.rho_interp_lower_frac.isel(t=t_index),
    "sxr_R_square_diff": sxr_los_data.R_square_diff.isel(t=t_index),
    "sxr_ne_x_power_loss": sxr_los_data.premult_values.isel(t=t_index),
    "sxr_los_values": sxr_los_data.los_values.isel(t=t_index),
    #    "los_errors": binned_camera.error.isel(t=t_index),
    "sxr_los_errors": sxr_los_data.los_errors.isel(t=t_index),
    # BOLO data:
    "bolo_N_los": bolo_los_data.N_los,
    # stan 1-based
    "bolo_rho_lower_indices": bolo_los_data.rho_lower_indices.isel(t=t_index) + 1,
    "bolo_rho_interp_lower_frac": bolo_los_data.rho_interp_lower_frac.isel(t=t_index),
    "bolo_R_square_diff": bolo_los_data.R_square_diff.isel(t=t_index),
    "bolo_ne_x_power_loss": bolo_los_data.premult_values.isel(t=t_index),
    "bolo_los_values": bolo_los_data.los_values.isel(t=t_index),
    #    "los_errors": binned_camera.error.isel(t=t_index),
    "bolo_los_errors": bolo_los_data.los_errors.isel(t=t_index),
}

# # likelihood-maximization (faster, worse results, no errors)
# opt_fit = model.optimize(data=data)
#
# # plot results
# plt.figure()
# plt.title("Optimization Results")
# plt.xlabel("rho")
# plt.ylabel("lfs_midplane emissivity")
# plt.plot(
#     rho.data,
#     opt_fit.lfs_values[0],
#     marker="x",
# )
# plt.grid()
# plt.show()

# Full Bayesian sampling (slower, better results, errors)
# Use samples.summary() and samples.diagnose() to check convergence
samples = model.sample(data=data, chains=16, parallel_chains=8)
draws = samples.draws_xr()
# Save data
draws.to_netcdf("bayesian_samples.nc")

# plot fit

plt.figure()
plt.title("Full sampling Results")
plt.xlabel("rho")
plt.ylabel("lfs_midplane emissivity")
plt.errorbar(
    rho.data,
    draws.lfs_values.mean(dim=("chain", "draw"))[0],
    yerr=draws.lfs_values.std(dim=("chain", "draw"))[0],
    marker="x",
)
plt.grid()
plt.show()
