"""Experiment entrypoints for jussiphd workflows."""

from .multipulse_synthetic import bolometry_inversion_multipulse_synthetic
from .generate_synth_multipulse_splined import bolometry_inversion_multipulse_synthetic_splined
from .eps_profile_clustering import cluster_synthetic_eps_profiles
from .multipulse_synthetic_noisy_test_b import (
    bolometry_inversion_multipulse_synthetic_noisy_test_b,
)
from .multipulse_synthetic_poisson_eps import (
    bolometry_inversion_multipulse_synthetic_poisson_eps,
)
from .inference_timing_synthetic import benchmark_synthetic_inference_time
from .comparison import ComparisonConfig
from .comparison import compare_real_vs_synthetic
from .noise_matching_brightness import (
    calibrate_synthetic_brightness_noise_against_real,
)
from .noise_matching import calibrate_synthetic_noise_against_real
from .te_ne_profile_comparison import compare_te_ne_profile_sampling
from .multipulse_synthetic_expanded_equilibria import (
    build_multipulse_synthetic_expanded_equilibria_dataset,
)
from .multipulse_synthetic_expanded_equilibria_train import (
    bolometry_inversion_multipulse_synthetic_expanded_equilibria,
)
from .multipulse_synthetic_expanded_equilibria_constant_imp import (
    build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset,
)
from .multipulse_synthetic_expanded_equilibria_constant_imp_compare import (
    multipulse_synthetic_expanded_equilibria_constant_imp_contextual_comparison,
)
from .multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_compare import (
    multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_contextual_comparison,
)
from .multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_compare import (
    multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_contextual_comparison,
)

__all__ = [
    "bolometry_inversion_multipulse_synthetic",
    "bolometry_inversion_multipulse_synthetic_splined",
    "cluster_synthetic_eps_profiles",
    "bolometry_inversion_multipulse_synthetic_noisy_test_b",
    "bolometry_inversion_multipulse_synthetic_poisson_eps",
    "benchmark_synthetic_inference_time",
    "ComparisonConfig",
    "compare_real_vs_synthetic",
    "calibrate_synthetic_brightness_noise_against_real",
    "calibrate_synthetic_noise_against_real",
    "compare_te_ne_profile_sampling",
    "build_multipulse_synthetic_expanded_equilibria_dataset",
    "bolometry_inversion_multipulse_synthetic_expanded_equilibria",
    "build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset",
    "multipulse_synthetic_expanded_equilibria_constant_imp_contextual_comparison",
    "multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_contextual_comparison",
    "multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_contextual_comparison",
]
