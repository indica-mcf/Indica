"""Reusable alignment and plotting helpers for paired real TE/NE profile datasets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from indica.workflows.jussiphd.components.filtering import filter_paired_profile_outliers


def load_csv_matrix(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float32)


def load_meta_rows(meta_path: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with Path(meta_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["pulse"]), float(row["t"])))
    return rows


def align_by_pulse(
    ne_rows: np.ndarray,
    ne_meta: list[tuple[int, float]],
    te_rows: np.ndarray,
    te_meta: list[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, str]]]:
    """Align middle-time NE/TE rows by pulse using FIFO matching on TE rows."""
    if ne_rows.shape[0] != len(ne_meta):
        raise ValueError(f"NE row/meta mismatch: rows={ne_rows.shape[0]} meta={len(ne_meta)}")
    if te_rows.shape[0] != len(te_meta):
        raise ValueError(f"TE row/meta mismatch: rows={te_rows.shape[0]} meta={len(te_meta)}")

    te_by_pulse: dict[int, list[int]] = defaultdict(list)
    for idx, (pulse, _t) in enumerate(te_meta):
        te_by_pulse[int(pulse)].append(int(idx))

    matched_ne: list[np.ndarray] = []
    matched_te: list[np.ndarray] = []
    matched_pulses: list[int] = []
    dropped: list[tuple[int, str]] = []

    for ne_idx, (pulse, _t_ne) in enumerate(ne_meta):
        candidates = te_by_pulse.get(int(pulse), [])
        if not candidates:
            dropped.append((int(pulse), "missing_te"))
            continue
        te_idx = candidates.pop(0)
        matched_ne.append(ne_rows[int(ne_idx)])
        matched_te.append(te_rows[int(te_idx)])
        matched_pulses.append(int(pulse))

    if not matched_ne:
        raise RuntimeError("No matching pulses between NE and TE reads after plasma gating.")

    return (
        np.asarray(matched_ne, dtype=np.float32),
        np.asarray(matched_te, dtype=np.float32),
        matched_pulses,
        dropped,
    )


def align_filter_plot_and_save_te_ne_profiles(
    ne_dataset: dict[str, Any],
    te_dataset: dict[str, Any],
    output_dir: str,
    plot_filename: str,
    ne_aligned_filename: str,
    te_aligned_filename: str,
    matched_meta_filename: str,
    outlier_report_filename: str,
    apply_outlier_filter: bool,
    outlier_point_z_threshold: float,
    outlier_extreme_point_z_threshold: float,
    outlier_min_bad_points: int,
    outlier_bad_fraction_threshold: float,
) -> dict[str, Any]:
    ne_rows = load_csv_matrix(ne_dataset["b_path"])
    te_rows = load_csv_matrix(te_dataset["b_path"])
    ne_meta = load_meta_rows(ne_dataset["meta_path"])
    te_meta = load_meta_rows(te_dataset["meta_path"])

    ne_aligned, te_aligned, matched_pulses, dropped = align_by_pulse(
        ne_rows=ne_rows,
        ne_meta=ne_meta,
        te_rows=te_rows,
        te_meta=te_meta,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ne_aligned_path = out / ne_aligned_filename
    te_aligned_path = out / te_aligned_filename
    matched_meta_path = out / matched_meta_filename
    outlier_report_path = out / outlier_report_filename
    plot_path = out / plot_filename

    outlier_summary: dict[str, Any] | None = None
    matched_pulses_before_filter = list(matched_pulses)
    if apply_outlier_filter:
        filtered = filter_paired_profile_outliers(
            primary=ne_aligned,
            secondary=te_aligned,
            point_z_threshold=outlier_point_z_threshold,
            extreme_point_z_threshold=outlier_extreme_point_z_threshold,
            min_bad_points=outlier_min_bad_points,
            bad_fraction_threshold=outlier_bad_fraction_threshold,
        )
        keep_mask = np.asarray(filtered["keep_mask"], dtype=bool)
        ne_aligned = np.asarray(filtered["primary_filtered"], dtype=np.float32)
        te_aligned = np.asarray(filtered["secondary_filtered"], dtype=np.float32)
        matched_pulses = [p for p, keep in zip(matched_pulses, keep_mask) if keep]

        with outlier_report_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_idx",
                    "pulse",
                    "removed_as_outlier",
                    "ne_max_abs_robust_z",
                    "te_max_abs_robust_z",
                    "ne_bad_count",
                    "te_bad_count",
                    "ne_bad_fraction",
                    "te_bad_fraction",
                ]
            )
            ne_det = filtered["primary_detection"]
            te_det = filtered["secondary_detection"]
            out_mask = np.asarray(filtered["outlier_mask"], dtype=bool)
            for i, pulse in enumerate(matched_pulses_before_filter):
                writer.writerow(
                    [
                        int(i),
                        int(pulse),
                        bool(out_mask[i]),
                        float(ne_det["max_abs_robust_z"][i]),
                        float(te_det["max_abs_robust_z"][i]),
                        int(ne_det["bad_count"][i]),
                        int(te_det["bad_count"][i]),
                        float(ne_det["bad_fraction"][i]),
                        float(te_det["bad_fraction"][i]),
                    ]
                )

        outlier_summary = {
            "applied": True,
            "n_input": int(filtered["n_input"]),
            "n_kept": int(filtered["n_kept"]),
            "n_outliers": int(filtered["n_outliers"]),
            "parameters": filtered["primary_detection"]["parameters"],
            "outlier_report_path": str(outlier_report_path),
        }
    else:
        outlier_summary = {
            "applied": False,
            "n_input": int(len(matched_pulses)),
            "n_kept": int(len(matched_pulses)),
            "n_outliers": 0,
            "outlier_report_path": None,
        }

    if ne_aligned.shape[0] == 0:
        raise RuntimeError("All matched Te/Ne profiles were removed by outlier filtering.")

    np.savetxt(ne_aligned_path, ne_aligned, delimiter=",")
    np.savetxt(te_aligned_path, te_aligned, delimiter=",")

    with matched_meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "pulse"])
        for i, pulse in enumerate(matched_pulses):
            writer.writerow([int(i), int(pulse)])

    fig, (ax_ne, ax_te) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, _pulse in enumerate(matched_pulses):
        ne = np.asarray(ne_aligned[i], dtype=np.float64)
        te = np.asarray(te_aligned[i], dtype=np.float64)
        ax_ne.plot(np.linspace(0.0, 1.0, ne.size, dtype=np.float64), ne, alpha=0.28, linewidth=1.2)
        ax_te.plot(np.linspace(0.0, 1.0, te.size, dtype=np.float64), te, alpha=0.28, linewidth=1.2)

    ax_ne.set_title("NE profiles (middle time index)")
    ax_ne.set_xlabel("normalized profile coordinate")
    ax_ne.set_ylabel("NE")
    ax_ne.grid(alpha=0.25)
    ax_te.set_title("TE profiles (middle time index)")
    ax_te.set_xlabel("normalized profile coordinate")
    ax_te.set_ylabel("TE")
    ax_te.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "ne_aligned_path": str(ne_aligned_path),
        "te_aligned_path": str(te_aligned_path),
        "matched_meta_path": str(matched_meta_path),
        "plot_path": str(plot_path),
        "num_matched": int(len(matched_pulses)),
        "num_dropped": int(len(dropped)),
        "dropped": dropped,
        "outlier_filter": outlier_summary,
    }

