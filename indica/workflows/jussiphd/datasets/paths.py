"""Centralized filesystem paths for reusable workflow datasets."""

from __future__ import annotations

from pathlib import Path

# workflows/jussiphd/datasets
DATASETS_ROOT = Path(__file__).resolve().parent

# Shared synthetic dataset used by multiple experiments.
MULTIPULSE_SYNTHETIC_DATA_DIR = DATASETS_ROOT / "multipulse_synthetic"
MULTIPULSE_SYNTHETIC_DATA_DIR_STR = str(MULTIPULSE_SYNTHETIC_DATA_DIR)


# Shared synthetic dataset used by multiple experiments, just in splines.
MULTIPULSE_SYNTHETIC_SPLINED_DATA_DIR = DATASETS_ROOT / "multipulse_synthetic_splined"
MULTIPULSE_SYNTHETIC_SPLINED_DATA_DIR_STR = str(MULTIPULSE_SYNTHETIC_SPLINED_DATA_DIR)

# Shared synthetic dataset generated from TE/NE anchor-cluster Gaussian families.
MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR = DATASETS_ROOT / "multipulse_synthetic_clustered"
MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR_STR = str(MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR)

# Expanded-brightness synthetic dataset reusing fixed eps over multiple equilibria.
MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR = (
    DATASETS_ROOT / "multipulse_synthetic_expanded_equilibria"
)
MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR = str(
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR
)

# Expanded-equilibria dataset with fixed impurity concentrations in sampled plasmas.
MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR = (
    DATASETS_ROOT / "multipulse_synthetic_expanded_equilibria_constant_imp"
)
MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR = str(
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR
)
