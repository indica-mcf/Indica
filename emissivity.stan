functions {
	/// Performs linear interpolation on an asymmetric quantity (model parameter)
	/// lower and upper vals are the grid values that the point lies between
	/// interp_lower_frac is where the point is between lower and upper as a fraction
	/// rho_point = (lower-upper)*t
	/// R_square_diff is the difference between the R value at the point and the R value at the same rho on the hfs midplane
	real interp_asymmetric(real lfs_lower_val, real lfs_upper_val, real asym_lower_val, real asym_upper_val, real interp_lower_frac, real R_square_diff) {
		real interp_lfs = lfs_lower_val*interp_lower_frac + lfs_upper_val * (1-interp_lower_frac);
		real interp_asym = asym_lower_val*interp_lower_frac + asym_upper_val*(1-interp_lower_frac);
		return interp_lfs * exp(interp_asym * R_square_diff);
	}

	/// Predict a set of LOS values given model parameters
	vector predict_los_vals(
		int N_los,
		int N_los_points,
		array[] real lfs_values,
		array[] real asym_params,
		array[,] int rho_lower_indices,
		array[,] real rho_interp_lower_frac,
		array[,] real R_square_diff
	){
		vector[N_los] predicted_los_vals;

		for (i_los in 1:N_los) {
			real los_val = 0;
			for (i_point in 1:N_los_points) {
				int i_low = rho_lower_indices[i_los, i_point];
				los_val += interp_asymmetric(
					lfs_values[i_low],
					lfs_values[i_low+1],
					asym_params[i_low],
					asym_params[i_low + 1],
					rho_interp_lower_frac[i_los, i_point],
					R_square_diff[i_los, i_point]
				);
			}
			predicted_los_vals[i_los] = los_val;
		}

		return predicted_los_vals;
	}
}

data {
	// Data affecting impurity densities
	// Number of rho points in impurity density grids
	int<lower=1> N_rho;

	// Data affecting lines of sight
	// Number of points in each line of sight
	int<lower=1> N_los_points;

	// Data required for SXR LOS calculations
	int<lower=1> sxr_N_los;
	array[sxr_N_los, N_los_points] int<lower=1> sxr_rho_lower_indices;
	array[sxr_N_los, N_los_points] real<lower=0, upper=1> sxr_rho_interp_lower_frac;
	// TODO: verify upper=0 here:
	array[sxr_N_los, N_los_points] real<upper=0> sxr_R_square_diff;
	vector<lower=0>[sxr_N_los] sxr_los_values;
	vector<lower=0>[sxr_N_los] sxr_los_errors;
}

parameters {
	array[N_rho] real<lower=0> lfs_values;
	array[N_rho] real asym_params;
}

transformed parameters {
	// Predict SXR LOS values
	vector<lower=0>[sxr_N_los] predicted_sxr_los_vals = predict_los_vals(sxr_N_los, N_los_points, lfs_values, asym_params, sxr_rho_lower_indices, sxr_rho_interp_lower_frac, sxr_R_square_diff);
}

model {
	// LOS values should be distributed like this:
	predicted_sxr_los_vals ~ normal(sxr_los_values, sxr_los_errors);
}
