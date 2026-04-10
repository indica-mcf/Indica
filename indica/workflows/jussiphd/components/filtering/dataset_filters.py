"""Dataset filtering utilities for bolometry surrogate datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _load_2d_csv(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _resolve_output_path(
    input_path: Path,
    overwrite: bool,
    output_dir: str | None,
    default_suffix: str,
) -> Path:
    if overwrite and output_dir is None:
        return input_path
    target_dir = Path(output_dir) if output_dir is not None else input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        return target_dir / input_path.name
    return target_dir / f"{input_path.stem}{default_suffix}{input_path.suffix}"


def apply_zero_and_tomo_filters(
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    zero_tol: float = 0.0,
    zero_slack: int = 30,
    min_valid_channels_required: int = 1,
    overwrite: bool = True,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Apply zero-heavy and tomography-channel filters, then write filtered files.

    Filter 1 (zero-heavy):
      keep sample if total zero-count <= (minimum zero-count across samples + zero_slack)

    Filter 2 (tomo channel validation):
      keep sample if count(isfinite(b) & (b > 0)) >= min_valid_channels_required
    """
    b_in = Path(b_path)
    eps_in = Path(eps_path)
    b_all = _load_2d_csv(str(b_in))
    eps_all = _load_2d_csv(str(eps_in))

    if len(b_all) != len(eps_all):
        raise ValueError("b_path and eps_path must contain the same number of samples")

    zero_counts = (
        np.count_nonzero(np.abs(b_all) <= float(zero_tol), axis=1)
        + np.count_nonzero(np.abs(eps_all) <= float(zero_tol), axis=1)
    )
    zero_threshold = int(zero_counts.min()) + int(zero_slack)
    keep_zero = zero_counts <= zero_threshold

    b_after_zero = b_all[keep_zero]
    eps_after_zero = eps_all[keep_zero]

    valid_channels = np.isfinite(b_after_zero) & (b_after_zero > 0.0)
    n_valid_channels = valid_channels.sum(axis=1)
    keep_tomo = n_valid_channels >= int(min_valid_channels_required)

    b_final = b_after_zero[keep_tomo]
    eps_final = eps_after_zero[keep_tomo]

    meta_out_path: Path | None = None
    meta_status: str | None = None
    num_meta_rows = None

    if meta_path is not None:
        meta_in = Path(meta_path)
        if meta_in.exists():
            with meta_in.open(newline="") as f:
                rows = list(csv.reader(f))
            has_header = bool(rows) and rows[0] == ["pulse", "time_s"]
            data_rows = rows[1:] if has_header else rows

            if len(data_rows) == len(keep_zero):
                data_after_zero = [row for row, keep in zip(data_rows, keep_zero) if keep]
                data_final = [row for row, keep in zip(data_after_zero, keep_tomo) if keep]
                meta_out_path = _resolve_output_path(
                    meta_in,
                    overwrite=overwrite,
                    output_dir=output_dir,
                    default_suffix="_filtered",
                )
                with meta_out_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    if has_header:
                        writer.writerow(["pulse", "time_s"])
                    writer.writerows(data_final)
                num_meta_rows = len(data_final)
                meta_status = "written"
            else:
                meta_status = (
                    "meta row count mismatch; expected same rows as input samples. "
                    "meta not written"
                )
        else:
            meta_status = "meta file not found; meta not written"

    b_out = _resolve_output_path(
        b_in,
        overwrite=overwrite,
        output_dir=output_dir,
        default_suffix="_filtered",
    )
    eps_out = _resolve_output_path(
        eps_in,
        overwrite=overwrite,
        output_dir=output_dir,
        default_suffix="_filtered",
    )

    np.savetxt(b_out, b_final, delimiter=",")
    np.savetxt(eps_out, eps_final, delimiter=",")

    return {
        "b_path": str(b_out),
        "eps_path": str(eps_out),
        "meta_path": str(meta_out_path) if meta_out_path is not None else meta_path,
        "meta_status": meta_status,
        "input_pairs": int(len(b_all)),
        "after_zero_pairs": int(len(b_after_zero)),
        "output_pairs": int(len(b_final)),
        "removed_zero_filter": int(len(b_all) - len(b_after_zero)),
        "removed_tomo_filter": int(len(b_after_zero) - len(b_final)),
        "zero_threshold": int(zero_threshold),
        "min_valid_channels_required": int(min_valid_channels_required),
        "b_shape": tuple(b_final.shape),
        "eps_shape": tuple(eps_final.shape),
        "num_meta_rows": num_meta_rows,
    }
