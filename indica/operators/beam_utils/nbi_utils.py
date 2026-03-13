"""Common utilities for NBI preprocessing and field sanitization."""

from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np


def fill_nan_2d(field: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 2D field using 1D interpolation passes and a final fallback.

    Note that this could also be done cols first->then rows!
    1) Interpolate along rows to recover horizontal gaps.
    2) Interpolate along columns to recover vertical gaps left after row pass.
    3) If isolated NaNs still remain, replace them with a representative finite value.
    """
    arr = np.array(field, dtype=float, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if not np.isnan(arr).any():
        # Fast path: return immediately when no cleaning is required.
        return arr

    # Pass 1 (row-wise): fill NaN runs using nearest valid samples in each row.
    # If a row has only one valid point, broadcast it across that row.
    x_row = np.arange(arr.shape[1], dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            arr[i, :] = np.interp(x_row, x_row[valid], row[valid])
        elif valid.sum() == 1:
            arr[i, :] = row[valid][0]

    # Pass 2 (column-wise): same idea, now along columns to catch values the
    # row-wise pass could not infer (e.g., large vertical missing regions).
    x_col = np.arange(arr.shape[0], dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            arr[:, j] = np.interp(x_col, x_col[valid], col[valid])
        elif valid.sum() == 1:
            arr[:, j] = col[valid][0]

    # Final fallback: if disconnected NaN islands remain after interpolation,
    # fill with the mean of finite non-zero values; otherwise use mean(finite)
    # or 0.0 if everything is invalid.
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


def save_magnetic_field_maps(
    file_name: str,
    R_2d: np.ndarray,
    z_2d: np.ndarray,
    br: np.ndarray,
    bz: np.ndarray,
    bt: np.ndarray,
    time: Union[float, int],
    stage: str,
) -> None:
    """Save 2D (R, z) maps of magnetic field components to disk."""
    import matplotlib.pyplot as plt

    output_dir = Path(f"{file_name}_bfield_maps")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, (name, field) in zip(axes, [("br", br), ("bz", bz), ("bt", bt)]):
        field_2d = np.asarray(field)
        if field_2d.shape != R_2d.shape and field_2d.T.shape == R_2d.shape:
            field_2d = field_2d.T

        mesh = ax.pcolormesh(R_2d, z_2d, field_2d, shading="auto")
        fig.colorbar(mesh, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("R")
    axes[0].set_ylabel("z")

    time_value = float(np.asarray(time).item())
    fig.suptitle(f"Magnetic field components ({stage}) at t={time_value:.6f}")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"bfield_components_{stage}_t_{time_value:.6f}.png",
        dpi=200,
    )
    plt.close(fig)
