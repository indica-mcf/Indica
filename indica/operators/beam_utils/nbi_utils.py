"""Common utilities for NBI preprocessing and field sanitization."""

from typing import Tuple

import numpy as np


def fill_nan_2d(field: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2D field by interpolation and a finite-value fallback."""
    arr = np.array(field, dtype=float, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if not np.isnan(arr).any():
        return arr

    # Pass 1: interpolate each row (R-like axis).
    x_row = np.arange(arr.shape[1], dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            arr[i, :] = np.interp(x_row, x_row[valid], row[valid])
        elif valid.sum() == 1:
            arr[i, :] = row[valid][0]

    # Pass 2: interpolate each column (z-like axis).
    x_col = np.arange(arr.shape[0], dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            arr[:, j] = np.interp(x_col, x_col[valid], col[valid])
        elif valid.sum() == 1:
            arr[:, j] = col[valid][0]

    # Final fallback for disconnected NaN regions.
    missing = np.isnan(arr)
    if missing.any():
        finite_nonzero = arr[np.isfinite(arr) & (arr != 0.0)]
        if finite_nonzero.size > 0:
            arr[missing] = float(np.mean(finite_nonzero))
        else:
            finite = arr[np.isfinite(arr)]
            arr[missing] = float(np.mean(finite)) if finite.size > 0 else 0.0

    return arr


def clean_magnetic_field_components(
    br: np.ndarray, bz: np.ndarray, bt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply consistent NaN handling to all magnetic field components."""
    return fill_nan_2d(br), fill_nan_2d(bz), fill_nan_2d(bt)
