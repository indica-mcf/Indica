"""Experiment entrypoints for jussiphd workflows."""

from .multipulse_synthetic import bolometry_inversion_multipulse_synthetic
from .multipulse_synthetic_noisy_test_b import (
    bolometry_inversion_multipulse_synthetic_noisy_test_b,
)
from .multipulse_synthetic_poisson_eps import (
    bolometry_inversion_multipulse_synthetic_poisson_eps,
)
from .inference_timing_synthetic import benchmark_synthetic_inference_time
from .comparison import ComparisonConfig
from .comparison import compare_real_vs_synthetic
from .noise_matching import calibrate_synthetic_noise_against_real

__all__ = [
    "bolometry_inversion_multipulse_synthetic",
    "bolometry_inversion_multipulse_synthetic_noisy_test_b",
    "bolometry_inversion_multipulse_synthetic_poisson_eps",
    "benchmark_synthetic_inference_time",
    "ComparisonConfig",
    "compare_real_vs_synthetic",
    "calibrate_synthetic_noise_against_real",
]
