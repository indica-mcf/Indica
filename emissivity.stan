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
		int N_elements,
		int N_los,
		int N_los_points,
		array[,] real lfs_values,
		array[,] real asym_params,
		array[,] int rho_lower_indices,
		array[,] real rho_interp_lower_frac,
		array[,] real R_square_diff,
		array[,,] real ne_x_power_loss
	){
		vector[N_los] predicted_los_vals;

		// predict each line of sight
		for (i_los in 1:N_los) {
			real los_val = 0;

			// sum over points in LOS
			for (i_point in 1:N_los_points) {
				int i_low = rho_lower_indices[i_los, i_point];

				// sum over elements
				for (i_element in 1:N_elements) {
					los_val += ne_x_power_loss[i_los, i_point, i_element] * interp_asymmetric(
						lfs_values[i_element, i_low],
						lfs_values[i_element, i_low+1],
						asym_params[i_element, i_low],
						asym_params[i_element, i_low + 1],
						rho_interp_lower_frac[i_los, i_point],
						R_square_diff[i_los, i_point]
					);
				}
			}
			predicted_los_vals[i_los] = los_val;
		}

		return predicted_los_vals;
	}
}

data {
	// Data affecting impurity densities
	int<lower=1> N_elements;
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
	// ne*power_loss for each los, los_point, element
	array[sxr_N_los, N_los_points, N_elements] real<lower=0> sxr_ne_x_power_loss;
	vector<lower=0>[sxr_N_los] sxr_los_values;
	vector<lower=0>[sxr_N_los] sxr_los_errors;

	// Data required for BOLO LOS calculations
	int<lower=1> bolo_N_los;
	array[bolo_N_los, N_los_points] int<lower=1> bolo_rho_lower_indices;
	array[bolo_N_los, N_los_points] real<lower=0, upper=1> bolo_rho_interp_lower_frac;
	// TODO: verify upper=0 here:
	array[bolo_N_los, N_los_points] real<upper=0> bolo_R_square_diff;
	// ne*power_loss for each los, los_point, element
	array[bolo_N_los, N_los_points, N_elements] real<lower=0> bolo_ne_x_power_loss;
	vector<lower=0>[bolo_N_los] bolo_los_values;
	vector<lower=0>[bolo_N_los] bolo_los_errors;
}

parameters {
	// TODO: revise limits
	array[N_elements, N_rho] real<lower=0, upper=1e17> lfs_values;
	array[N_elements, N_rho] real<lower=-10, upper=10> asym_params;
	real<lower=0.1, upper=5> sxr_calibration_factor;
}

transformed parameters {
	// Predict SXR LOS values
	vector<lower=0>[sxr_N_los] predicted_sxr_los_vals = (1/sxr_calibration_factor) * predict_los_vals(N_elements, sxr_N_los, N_los_points, lfs_values, asym_params, sxr_rho_lower_indices, sxr_rho_interp_lower_frac, sxr_R_square_diff, sxr_ne_x_power_loss);
	vector<lower=0>[bolo_N_los] predicted_bolo_los_vals = predict_los_vals(N_elements, bolo_N_los, N_los_points, lfs_values, asym_params, bolo_rho_lower_indices, bolo_rho_interp_lower_frac, bolo_R_square_diff, bolo_ne_x_power_loss);
}

model {
	sxr_calibration_factor ~ normal(3.0, 3.0);

	// LOS values should be distributed like this:
	predicted_sxr_los_vals ~ normal(sxr_los_values, sxr_los_errors);
	predicted_bolo_los_vals ~ normal(bolo_los_values, bolo_los_errors);
}
