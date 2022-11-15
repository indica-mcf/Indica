functions {
	real interp_asymmetric(real lfs_lower_val, real lfs_upper_val, real asym_lower_val, real asym_upper_val, real interp_lower_frac, real R_square_diff) {
		real interp_lfs = lfs_lower_val*interp_lower_frac + lfs_upper_val * (1-interp_lower_frac);
		real interp_asym = asym_lower_val*interp_lower_frac + asym_upper_val*(1-interp_lower_frac);
		return interp_lfs * exp(interp_asym * R_square_diff);
	}

	vector predict_los_vals(
		int N_los,
		int N_points,
		array[] real lfs_values,
		array[] real asym_params,
		array[,] int rho_lower_indices,
		array[,] real rho_interp_lower_frac,
		array[,] real R_square_diff
	){
		vector[N_los] predicted_los_vals;

		for (i_los in 1:N_los) {
			real los_val = 0;
			for (i_point in 1:N_points) {
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
	int<lower=1> N_rho;

	int<lower=1> N_sxr_los;
	int<lower=1> N_points;
	array[N_sxr_los, N_points] int<lower=1> rho_lower_indices;
	array[N_sxr_los, N_points] real<lower=0, upper=1> rho_interp_lower_frac;
	// TODO: verify upper=0 here:
	array[N_sxr_los, N_points] real<upper=0> R_square_diff;
	array[N_sxr_los] real<lower=0> sxr_los_values;
	array[N_sxr_los] real<lower=0> sxr_los_errors;
}

parameters {
	array[N_rho] real<lower=0> lfs_values;
	array[N_rho] real asym_params;
}

transformed parameters {
	vector[N_sxr_los] predicted_sxr_los_vals = predict_los_vals(N_sxr_los, N_points, lfs_values, asym_params, rho_lower_indices, rho_interp_lower_frac, R_square_diff);
}

model {
	for (i_los in 1:N_sxr_los) {
		predicted_sxr_los_vals[i_los] ~ normal(sxr_los_values[i_los], sxr_los_errors[i_los]);
	}
}
