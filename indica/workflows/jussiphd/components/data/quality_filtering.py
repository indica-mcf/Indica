"""Reusable dataset quality-filtering helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _slice_passes_basic_quality(
    b_slice: np.ndarray,
    eps_slice: np.ndarray,
    min_finite_fraction: float,
    min_nonzero_fraction: float,
    nonzero_threshold: float,
) -> bool:
    b = np.asarray(b_slice, dtype=np.float32).reshape(-1)
    eps = np.asarray(eps_slice, dtype=np.float32).reshape(-1)

    b_finite = float(np.isfinite(b).mean())
    eps_finite = float(np.isfinite(eps).mean())
    if b_finite < float(min_finite_fraction):
        return False
    if eps_finite < float(min_finite_fraction):
        return False

    b_nonzero = float(np.mean(np.abs(np.nan_to_num(b, nan=0.0)) > float(nonzero_threshold)))
    eps_nonzero = float(np.mean(np.abs(np.nan_to_num(eps, nan=0.0)) > float(nonzero_threshold)))
    if b_nonzero < float(min_nonzero_fraction):
        return False
    if eps_nonzero < float(min_nonzero_fraction):
        return False
    return True


def filter_dataset_csv_slices(
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    min_finite_fraction: float = 0.95,
    min_nonzero_fraction: float = 0.01,
    nonzero_threshold: float = 0.0,
    output_suffix: str = "quality_filtered",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Filter paired CSV slices by simple finite/nonzero quality checks."""
    b_in = Path(b_path)
    eps_in = Path(eps_path)
    if not b_in.exists() or not eps_in.exists():
        raise FileNotFoundError(f"Missing input CSV(s): {b_in}, {eps_in}")

    b_arr = np.loadtxt(b_in, delimiter=",", dtype=np.float32)
    eps_arr = np.loadtxt(eps_in, delimiter=",", dtype=np.float32)
    if b_arr.ndim == 1:
        b_arr = b_arr[None, :]
    if eps_arr.ndim == 1:
        eps_arr = eps_arr[None, :]
    if b_arr.shape[0] != eps_arr.shape[0]:
        raise ValueError(f"Row mismatch: b={b_arr.shape}, eps={eps_arr.shape}")

    meta_header: list[str] | None = None
    meta_rows: list[list[str]] | None = None
    if meta_path is not None and Path(meta_path).exists():
        with Path(meta_path).open(newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] == ["pulse", "time_s"]:
            meta_header = rows[0]
            meta_rows = rows[1:]
        else:
            meta_rows = rows
        if meta_rows is not None and len(meta_rows) != b_arr.shape[0]:
            raise ValueError(
                f"Meta row mismatch: meta={len(meta_rows)}, dataset={b_arr.shape[0]}"
            )

    keep_mask = np.zeros(b_arr.shape[0], dtype=bool)
    for i in range(b_arr.shape[0]):
        keep_mask[i] = _slice_passes_basic_quality(
            b_slice=b_arr[i],
            eps_slice=eps_arr[i],
            min_finite_fraction=min_finite_fraction,
            min_nonzero_fraction=min_nonzero_fraction,
            nonzero_threshold=nonzero_threshold,
        )

    n_total = int(len(keep_mask))
    n_kept = int(np.sum(keep_mask))
    n_filtered = int(n_total - n_kept)
    if n_kept == 0:
        raise RuntimeError(
            "All slices were filtered out by quality thresholds. "
            f"total={n_total}, min_finite_fraction={min_finite_fraction}, "
            f"min_nonzero_fraction={min_nonzero_fraction}, nonzero_threshold={nonzero_threshold}"
        )

    b_out_arr = b_arr[keep_mask]
    eps_out_arr = eps_arr[keep_mask]
    if overwrite:
        b_out = b_in
        eps_out = eps_in
    else:
        b_out = b_in.with_name(f"{b_in.stem}_{output_suffix}{b_in.suffix}")
        eps_out = eps_in.with_name(f"{eps_in.stem}_{output_suffix}{eps_in.suffix}")

    np.savetxt(b_out, b_out_arr, delimiter=",")
    np.savetxt(eps_out, eps_out_arr, delimiter=",")

    meta_out: str | None = None
    if meta_rows is not None:
        kept_meta = [row for row, keep in zip(meta_rows, keep_mask) if bool(keep)]
        in_meta = Path(meta_path) if meta_path is not None else None
        if overwrite and in_meta is not None:
            out_meta = in_meta
        elif in_meta is not None:
            out_meta = in_meta.with_name(f"{in_meta.stem}_{output_suffix}{in_meta.suffix}")
        else:
            out_meta = None
        if out_meta is not None:
            with out_meta.open("w", newline="") as f:
                writer = csv.writer(f)
                if meta_header is not None:
                    writer.writerow(meta_header)
                writer.writerows(kept_meta)
            meta_out = str(out_meta)

    print(f"Filter results: total={n_total}, kept={n_kept}, filtered={n_filtered}")

    return {
        "b_path": str(b_out),
        "eps_path": str(eps_out),
        "meta_path": meta_out,
        "num_rows_input": n_total,
        "num_rows_kept": n_kept,
        "num_rows_filtered": n_filtered,
        "min_finite_fraction": float(min_finite_fraction),
        "min_nonzero_fraction": float(min_nonzero_fraction),
        "nonzero_threshold": float(nonzero_threshold),
        "overwrite": bool(overwrite),
    }
