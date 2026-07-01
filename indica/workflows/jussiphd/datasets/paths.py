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