"""Reusable TE/NE spline-anchor fitting and artifact export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from indica.workflows.jussiphd.components.preprocessing.spline_anchor_fitting import (
    fit_profiles_to_anchor_space,
    load_monospline_anchor_spec,
)
from indica.workflows.jussiphd.components.preprocessing.te_ne_alignment import load_csv_matrix


def fit_save_and_plot_te_ne_anchor_space(
    ne_aligned_path: str,
    te_aligned_path: str,
    matched_meta_path: str,
    output_dir: str,
    config_name: str,
    ne_profile_name: str,
    te_profile_name: str,
    ne_anchor_filename: str,
    te_anchor_filename: str,
    fit_summary_filename: str,
    fit_spec_filename: str,
    ne_anchor_plot_filename: str,
    te_anchor_plot_filename: str,
    ne_middle_knot_plot_filename: str,
    te_middle_knot_plot_filename: str,
    ne_fixed_start: float | None,
    ne_fixed_end: float | None,
    te_fixed_start: float | None,
    te_fixed_end: float | None,
    ne_start_min: float | None,
    ne_start_max: float | None,
    te_start_min: float | None,
    te_start_max: float | None,
) -> dict[str, Any]:
    ne = load_csv_matrix(ne_aligned_path)
    te = load_csv_matrix(te_aligned_path)
    if ne.shape[0] != te.shape[0]:
        raise ValueError(f"Aligned NE/TE row mismatch: {ne.shape[0]} vs {te.shape[0]}.")

    pulses: list[int] = []
    with Path(matched_meta_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pulses.append(int(row["pulse"]))
    if len(pulses) != ne.shape[0]:
        raise ValueError(
            f"Matched meta rows ({len(pulses)}) do not match aligned profiles ({ne.shape[0]})."
        )

    ne_spec = load_monospline_anchor_spec(profile_name=ne_profile_name, config_name=config_name)
    te_spec = load_monospline_anchor_spec(profile_name=te_profile_name, config_name=config_name)

    ne_fit = fit_profiles_to_anchor_space(
        ne,
        xknots=np.asarray(ne_spec["xknots"], dtype=np.float64),
        fixed_start=ne_fixed_start,
        fixed_end=ne_fixed_end,
    )
    te_fit = fit_profiles_to_anchor_space(
        te,
        xknots=np.asarray(te_spec["xknots"], dtype=np.float64),
        fixed_start=te_fixed_start,
        fixed_end=te_fixed_end,
    )

    keep_mask = np.asarray(ne_fit["ok_mask"], dtype=bool) & np.asarray(te_fit["ok_mask"], dtype=bool)
    ne_anchor_all = np.asarray(ne_fit["anchors"], dtype=np.float64)
    te_anchor_all = np.asarray(te_fit["anchors"], dtype=np.float64)
    ne_start = ne_anchor_all[:, 0]
    te_start = te_anchor_all[:, 0]
    if ne_start_min is not None:
        keep_mask &= ne_start >= float(ne_start_min)
    if ne_start_max is not None:
        keep_mask &= ne_start <= float(ne_start_max)
    if te_start_min is not None:
        keep_mask &= te_start >= float(te_start_min)
    if te_start_max is not None:
        keep_mask &= te_start <= float(te_start_max)
    if int(keep_mask.sum()) == 0:
        raise RuntimeError("Spline fitting failed for all aligned samples.")

    ne_anchors = np.asarray(ne_fit["anchors"], dtype=np.float32)[keep_mask]
    te_anchors = np.asarray(te_fit["anchors"], dtype=np.float32)[keep_mask]
    pulses_kept = [p for p, keep in zip(pulses, keep_mask) if keep]
    ne_rmse = np.asarray(ne_fit["rmse"], dtype=np.float32)[keep_mask]
    te_rmse = np.asarray(te_fit["rmse"], dtype=np.float32)[keep_mask]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ne_anchor_path = out / ne_anchor_filename
    te_anchor_path = out / te_anchor_filename
    fit_summary_path = out / fit_summary_filename
    fit_spec_path = out / fit_spec_filename
    ne_anchor_plot_path = out / ne_anchor_plot_filename
    te_anchor_plot_path = out / te_anchor_plot_filename
    ne_middle_knot_plot_path = out / ne_middle_knot_plot_filename
    te_middle_knot_plot_path = out / te_middle_knot_plot_filename

    np.savetxt(ne_anchor_path, ne_anchors, delimiter=",")
    np.savetxt(te_anchor_path, te_anchors, delimiter=",")

    with fit_summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "pulse", "ne_rmse", "te_rmse", "ne_start", "te_start"])
        for i, pulse in enumerate(pulses_kept):
            writer.writerow(
                [
                    int(i),
                    int(pulse),
                    float(ne_rmse[i]),
                    float(te_rmse[i]),
                    float(ne_anchors[i, 0]),
                    float(te_anchors[i, 0]),
                ]
            )

    fit_spec = {
        "config_name": config_name,
        "ne_spec": {
            "profile_name": ne_spec["profile_name"],
            "xknots": [float(v) for v in np.asarray(ne_spec["xknots"]).reshape(-1)],
            "fixed_start": None if ne_fixed_start is None else float(ne_fixed_start),
            "fixed_end": None if ne_fixed_end is None else float(ne_fixed_end),
            "fixed_start_param": None if ne_fixed_start is None else str(ne_spec["fixed_start_param"]),
            "fixed_end_param": None if ne_fixed_end is None else str(ne_spec["fixed_end_param"]),
            "n_anchors": int(ne_spec["n_anchors"]),
        },
        "te_spec": {
            "profile_name": te_spec["profile_name"],
            "xknots": [float(v) for v in np.asarray(te_spec["xknots"]).reshape(-1)],
            "fixed_start": None if te_fixed_start is None else float(te_fixed_start),
            "fixed_end": None if te_fixed_end is None else float(te_fixed_end),
            "fixed_start_param": None if te_fixed_start is None else str(te_spec["fixed_start_param"]),
            "fixed_end_param": None if te_fixed_end is None else str(te_spec["fixed_end_param"]),
            "n_anchors": int(te_spec["n_anchors"]),
        },
        "start_range_filter": {
            "ne_start_min": None if ne_start_min is None else float(ne_start_min),
            "ne_start_max": None if ne_start_max is None else float(ne_start_max),
            "te_start_min": None if te_start_min is None else float(te_start_min),
            "te_start_max": None if te_start_max is None else float(te_start_max),
        },
        "n_input": int(ne.shape[0]),
        "n_fit_kept": int(len(pulses_kept)),
        "n_fit_dropped": int(ne.shape[0] - len(pulses_kept)),
    }
    fit_spec_path.write_text(json.dumps(fit_spec, indent=2))

    ne_x = np.asarray(ne_spec["xknots"], dtype=np.float64).reshape(-1)
    te_x = np.asarray(te_spec["xknots"], dtype=np.float64).reshape(-1)

    fig_ne, ax_ne = plt.subplots(figsize=(8.5, 5.0))
    for row in ne_anchors:
        ax_ne.plot(ne_x, np.asarray(row, dtype=np.float64), color="#1f77b4", alpha=0.20, linewidth=1.1)
    ax_ne.plot(ne_x, np.nanmean(ne_anchors, axis=0), color="black", linewidth=2.4, label="mean anchors")
    ax_ne.set_title("NE spline anchors (fitted, middle-time real profiles)")
    ax_ne.set_xlabel("knot coordinate (xknots)")
    ax_ne.set_ylabel("anchor value")
    ax_ne.grid(alpha=0.25)
    ax_ne.legend()
    fig_ne.tight_layout()
    fig_ne.savefig(ne_anchor_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_ne)

    fig_te, ax_te = plt.subplots(figsize=(8.5, 5.0))
    for row in te_anchors:
        ax_te.plot(te_x, np.asarray(row, dtype=np.float64), color="#d62728", alpha=0.20, linewidth=1.1)
    ax_te.plot(te_x, np.nanmean(te_anchors, axis=0), color="black", linewidth=2.4, label="mean anchors")
    ax_te.set_title("TE spline anchors (fitted, middle-time real profiles)")
    ax_te.set_xlabel("knot coordinate (xknots)")
    ax_te.set_ylabel("anchor value")
    ax_te.grid(alpha=0.25)
    ax_te.legend()
    fig_te.tight_layout()
    fig_te.savefig(te_anchor_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_te)

    x_full_ne = np.linspace(0.0, 1.0, ne.shape[1], dtype=np.float64)
    x_full_te = np.linspace(0.0, 1.0, te.shape[1], dtype=np.float64)
    ne_middle_on_knots = np.vstack(
        [np.interp(ne_x, x_full_ne, np.asarray(row, dtype=np.float64)) for row in ne[keep_mask]]
    ).astype(np.float32)
    te_middle_on_knots = np.vstack(
        [np.interp(te_x, x_full_te, np.asarray(row, dtype=np.float64)) for row in te[keep_mask]]
    ).astype(np.float32)

    fig_ne_m, ax_ne_m = plt.subplots(figsize=(8.5, 5.0))
    for row in ne_middle_on_knots:
        ax_ne_m.plot(ne_x, np.asarray(row, dtype=np.float64), color="#1f77b4", alpha=0.20, linewidth=1.1)
    ax_ne_m.plot(
        ne_x, np.nanmean(ne_middle_on_knots, axis=0), color="black", linewidth=2.4, label="mean middle profile"
    )
    ax_ne_m.set_title("NE middle profiles projected to spline-knot domain")
    ax_ne_m.set_xlabel("knot coordinate (xknots)")
    ax_ne_m.set_ylabel("profile value")
    ax_ne_m.grid(alpha=0.25)
    ax_ne_m.legend()
    fig_ne_m.tight_layout()
    fig_ne_m.savefig(ne_middle_knot_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_ne_m)

    fig_te_m, ax_te_m = plt.subplots(figsize=(8.5, 5.0))
    for row in te_middle_on_knots:
        ax_te_m.plot(te_x, np.asarray(row, dtype=np.float64), color="#d62728", alpha=0.20, linewidth=1.1)
    ax_te_m.plot(
        te_x, np.nanmean(te_middle_on_knots, axis=0), color="black", linewidth=2.4, label="mean middle profile"
    )
    ax_te_m.set_title("TE middle profiles projected to spline-knot domain")
    ax_te_m.set_xlabel("knot coordinate (xknots)")
    ax_te_m.set_ylabel("profile value")
    ax_te_m.grid(alpha=0.25)
    ax_te_m.legend()
    fig_te_m.tight_layout()
    fig_te_m.savefig(te_middle_knot_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_te_m)

    return {
        "ne_anchor_path": str(ne_anchor_path),
        "te_anchor_path": str(te_anchor_path),
        "fit_summary_path": str(fit_summary_path),
        "fit_spec_path": str(fit_spec_path),
        "ne_anchor_plot_path": str(ne_anchor_plot_path),
        "te_anchor_plot_path": str(te_anchor_plot_path),
        "ne_middle_knot_plot_path": str(ne_middle_knot_plot_path),
        "te_middle_knot_plot_path": str(te_middle_knot_plot_path),
        "n_fit_kept": int(len(pulses_kept)),
        "n_fit_dropped": int(ne.shape[0] - len(pulses_kept)),
    }

