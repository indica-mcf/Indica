"""Robust outlier filtering for profile matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def _robust_modified_zscores(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute column-wise modified z-scores using median and MAD."""
    x = np.asarray(matrix, dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    scale = 1.4826 * np.maximum(mad, float(eps))
    return (x - med) / scale


def detect_profile_outliers(
    matrix: np.ndarray,
    point_z_threshold: float = 8.0,
    extreme_point_z_threshold: float = 15.0,
    min_bad_points: int = 2,
    bad_fraction_threshold: float = 0.08,
) -> dict[str, Any]:
    """Detect row outliers in a profile matrix using robust z-score heuristics."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {x.shape}")

    z = np.abs(_robust_modified_zscores(x))
    finite = np.isfinite(z)
    bad = finite & (z >= float(point_z_threshold))

    bad_count = bad.sum(axis=1).astype(int)
    denom = np.maximum(finite.sum(axis=1), 1)
    bad_fraction = bad_count / denom
    max_abs_z = np.where(finite, z, np.nan)
    max_abs_z = np.nanmax(max_abs_z, axis=1)
    max_abs_z = np.nan_to_num(max_abs_z, nan=0.0, posinf=np.inf)

    outlier_mask = (max_abs_z >= float(extreme_point_z_threshold)) | (
        (bad_count >= int(min_bad_points))
        & (bad_fraction >= float(bad_fraction_threshold))
    )
    keep_mask = ~outlier_mask

    return {
        "keep_mask": keep_mask,
        "outlier_mask": outlier_mask,
        "bad_count": bad_count,
        "bad_fraction": bad_fraction,
        "max_abs_robust_z": max_abs_z,
        "n_input": int(x.shape[0]),
        "n_kept": int(keep_mask.sum()),
        "n_outliers": int(outlier_mask.sum()),
        "parameters": {
            "point_z_threshold": float(point_z_threshold),
            "extreme_point_z_threshold": float(extreme_point_z_threshold),
            "min_bad_points": int(min_bad_points),
            "bad_fraction_threshold": float(bad_fraction_threshold),
        },
    }


def filter_paired_profile_outliers(
    primary: np.ndarray,
    secondary: np.ndarray,
    point_z_threshold: float = 8.0,
    extreme_point_z_threshold: float = 15.0,
    min_bad_points: int = 2,
    bad_fraction_threshold: float = 0.08,
) -> dict[str, Any]:
    """Filter paired profile matrices by removing rows outlier in either matrix."""
    x1 = np.asarray(primary, dtype=np.float64)
    x2 = np.asarray(secondary, dtype=np.float64)
    if x1.ndim == 1:
        x1 = x1[None, :]
    if x2.ndim == 1:
        x2 = x2[None, :]
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got {x1.shape} and {x2.shape}")
    if x1.shape[0] != x2.shape[0]:
        raise ValueError(
            f"Paired matrices must have same number of rows, got {x1.shape[0]} and {x2.shape[0]}"
        )

    p1 = detect_profile_outliers(
        x1,
        point_z_threshold=point_z_threshold,
        extreme_point_z_threshold=extreme_point_z_threshold,
        min_bad_points=min_bad_points,
        bad_fraction_threshold=bad_fraction_threshold,
    )
    p2 = detect_profile_outliers(
        x2,
        point_z_threshold=point_z_threshold,
        extreme_point_z_threshold=extreme_point_z_threshold,
        min_bad_points=min_bad_points,
        bad_fraction_threshold=bad_fraction_threshold,
    )

    outlier_union = np.asarray(p1["outlier_mask"]) | np.asarray(p2["outlier_mask"])
    keep_mask = ~outlier_union

    return {
        "primary_filtered": x1[keep_mask].astype(np.float32),
        "secondary_filtered": x2[keep_mask].astype(np.float32),
        "keep_mask": keep_mask,
        "outlier_mask": outlier_union,
        "primary_detection": p1,
        "secondary_detection": p2,
        "n_input": int(x1.shape[0]),
        "n_kept": int(keep_mask.sum()),
        "n_outliers": int(outlier_union.sum()),
    }

