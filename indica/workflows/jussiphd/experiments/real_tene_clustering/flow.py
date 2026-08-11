"""Prefect flow to read and visualize real ST40 Te/Ne profiles by pulse."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from prefect import flow, task

from indica.workflows.jussiphd.components.data.real_brightness_dataset_generation import (
    generate_and_save_real_multipulse_brightness_dataset,
)
from indica.workflows.jussiphd.components.filtering import (
    filter_paired_profile_outliers,
)
from indica.workflows.jussiphd.components.preprocessing import (
    fit_profiles_to_anchor_space,
    load_monospline_anchor_spec,
)

TS_NE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:NE"
TS_TE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:TE"
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_TSTART = 0.04
DEFAULT_TEND = 0.15


def _load_csv_matrix(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float32)


def _load_meta_rows(meta_path: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with Path(meta_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["pulse"]), float(row["t"])))
    return rows


def _align_by_pulse(
    ne_rows: np.ndarray,
    ne_meta: list[tuple[int, float]],
    te_rows: np.ndarray,
    te_meta: list[tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, str]]]:
    """Align middle-time NE/TE rows by pulse using first-in-first-out matching."""
    if ne_rows.shape[0] != len(ne_meta):
        raise ValueError(
            f"NE row/meta mismatch: rows={ne_rows.shape[0]} meta={len(ne_meta)}"
        )
    if te_rows.shape[0] != len(te_meta):
        raise ValueError(
            f"TE row/meta mismatch: rows={te_rows.shape[0]} meta={len(te_meta)}"
        )

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
        raise RuntimeError(
            "No matching pulses between NE and TE reads after plasma gating."
        )

    return (
        np.asarray(matched_ne, dtype=np.float32),
        np.asarray(matched_te, dtype=np.float32),
        matched_pulses,
        dropped,
    )


@task(name="build_real_node_profile_dataset")
def build_real_node_profile_dataset_task(
    pulses: list[int],
    node: str,
    output_dir: str,
    profile_filename: str,
    meta_filename: str,
    tstart: float,
    tend: float,
    dt: float,
    read_verbose: bool,
    min_finite_fraction: float,
) -> dict[str, Any]:
    return generate_and_save_real_multipulse_brightness_dataset(
        pulses=pulses,
        instrument="blom_rz1",
        tstart=tstart,
        tend=tend,
        dt=dt,
        output_dir=output_dir,
        b_filename=profile_filename,
        meta_filename=meta_filename,
        use_all_timepoints=False,
        node=node,
        generate_new_data=True,
        verbose=read_verbose,
        apply_basic_quality_filter=True,
        min_finite_fraction=min_finite_fraction,
        min_nonzero_fraction=0.0,
        nonzero_threshold=0.0,
        require_plasma_summary=True,
    )


@task(name="align_and_plot_te_ne_profiles")
def align_and_plot_te_ne_profiles_task(
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
    ne_rows = _load_csv_matrix(ne_dataset["b_path"])
    te_rows = _load_csv_matrix(te_dataset["b_path"])
    ne_meta = _load_meta_rows(ne_dataset["meta_path"])
    te_meta = _load_meta_rows(te_dataset["meta_path"])

    ne_aligned, te_aligned, matched_pulses, dropped = _align_by_pulse(
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
    for i, pulse in enumerate(matched_pulses):
        ne = np.asarray(ne_aligned[i], dtype=np.float64)
        te = np.asarray(te_aligned[i], dtype=np.float64)
        ax_ne.plot(
            np.linspace(0.0, 1.0, ne.size, dtype=np.float64),
            ne,
            alpha=0.28,
            linewidth=1.2,
            label=str(pulse),
        )
        ax_te.plot(
            np.linspace(0.0, 1.0, te.size, dtype=np.float64),
            te,
            alpha=0.28,
            linewidth=1.2,
            label=str(pulse),
        )

    ax_ne.set_title("NE profiles (middle time index)")
    ax_ne.set_xlabel("normalized profile coordinate")
    ax_ne.set_ylabel("NE")
    ax_ne.grid(alpha=0.25)
    ax_te.set_title("TE profiles (middle time index)")
    ax_te.set_xlabel("normalized profile coordinate")
    ax_te.set_ylabel("TE")
    ax_te.grid(alpha=0.25)

    handles, labels = ax_ne.get_legend_handles_labels()
    if labels:
        keep = min(12, len(labels))
        fig.legend(
            handles[:keep],
            labels[:keep],
            title="Pulse",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=max(1, min(6, keep)),
            fontsize=8,
        )
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


@task(name="fit_te_ne_to_spline_anchor_space")
def fit_te_ne_to_spline_anchor_space_task(
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
) -> dict[str, Any]:
    ne = _load_csv_matrix(ne_aligned_path)
    te = _load_csv_matrix(te_aligned_path)
    if ne.shape[0] != te.shape[0]:
        raise ValueError(
            f"Aligned NE/TE row mismatch: {ne.shape[0]} vs {te.shape[0]}."
        )

    pulses: list[int] = []
    with Path(matched_meta_path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pulses.append(int(row["pulse"]))
    if len(pulses) != ne.shape[0]:
        raise ValueError(
            f"Matched meta rows ({len(pulses)}) do not match aligned profiles ({ne.shape[0]})."
        )

    ne_spec = load_monospline_anchor_spec(
        profile_name=ne_profile_name,
        config_name=config_name,
    )
    te_spec = load_monospline_anchor_spec(
        profile_name=te_profile_name,
        config_name=config_name,
    )

    ne_fit = fit_profiles_to_anchor_space(
        ne,
        xknots=np.asarray(ne_spec["xknots"], dtype=np.float64),
        fixed_start=float(ne_spec["fixed_start"]),
        fixed_end=float(ne_spec["fixed_end"]),
    )
    te_fit = fit_profiles_to_anchor_space(
        te,
        xknots=np.asarray(te_spec["xknots"], dtype=np.float64),
        fixed_start=float(te_spec["fixed_start"]),
        fixed_end=float(te_spec["fixed_end"]),
    )

    keep_mask = np.asarray(ne_fit["ok_mask"], dtype=bool) & np.asarray(te_fit["ok_mask"], dtype=bool)
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

    np.savetxt(ne_anchor_path, ne_anchors, delimiter=",")
    np.savetxt(te_anchor_path, te_anchors, delimiter=",")

    with fit_summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "pulse", "ne_rmse", "te_rmse"])
        for i, pulse in enumerate(pulses_kept):
            writer.writerow([int(i), int(pulse), float(ne_rmse[i]), float(te_rmse[i])])

    fit_spec = {
        "config_name": config_name,
        "ne_spec": {
            "profile_name": ne_spec["profile_name"],
            "xknots": [float(v) for v in np.asarray(ne_spec["xknots"]).reshape(-1)],
            "fixed_start": float(ne_spec["fixed_start"]),
            "fixed_end": float(ne_spec["fixed_end"]),
            "fixed_start_param": str(ne_spec["fixed_start_param"]),
            "fixed_end_param": str(ne_spec["fixed_end_param"]),
            "n_anchors": int(ne_spec["n_anchors"]),
        },
        "te_spec": {
            "profile_name": te_spec["profile_name"],
            "xknots": [float(v) for v in np.asarray(te_spec["xknots"]).reshape(-1)],
            "fixed_start": float(te_spec["fixed_start"]),
            "fixed_end": float(te_spec["fixed_end"]),
            "fixed_start_param": str(te_spec["fixed_start_param"]),
            "fixed_end_param": str(te_spec["fixed_end_param"]),
            "n_anchors": int(te_spec["n_anchors"]),
        },
        "n_input": int(ne.shape[0]),
        "n_fit_kept": int(len(pulses_kept)),
        "n_fit_dropped": int(ne.shape[0] - len(pulses_kept)),
    }
    fit_spec_path.write_text(json.dumps(fit_spec, indent=2))

    return {
        "ne_anchor_path": str(ne_anchor_path),
        "te_anchor_path": str(te_anchor_path),
        "fit_summary_path": str(fit_summary_path),
        "fit_spec_path": str(fit_spec_path),
        "n_fit_kept": int(len(pulses_kept)),
        "n_fit_dropped": int(ne.shape[0] - len(pulses_kept)),
    }


@flow(name="real_tene_clustering")
def real_tene_clustering(
    pulses: Sequence[int] | None = None,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = 0.01,
    read_verbose: bool = False,
    ne_node: str = TS_NE_NODE,
    te_node: str = TS_TE_NODE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ne_filename: str = "ne_middle_profiles_real_raw.csv",
    te_filename: str = "te_middle_profiles_real_raw.csv",
    ne_meta_filename: str = "ne_middle_profiles_meta.csv",
    te_meta_filename: str = "te_middle_profiles_meta.csv",
    plot_filename: str = "te_ne_middle_profiles_overlay.png",
    ne_aligned_filename: str = "ne_middle_profiles_real_aligned.csv",
    te_aligned_filename: str = "te_middle_profiles_real_aligned.csv",
    matched_meta_filename: str = "te_ne_matched_meta.csv",
    outlier_report_filename: str = "te_ne_outlier_report.csv",
    min_finite_fraction: float = 0.90,
    apply_outlier_filter: bool = True,
    outlier_point_z_threshold: float = 8.0,
    outlier_extreme_point_z_threshold: float = 15.0,
    outlier_min_bad_points: int = 2,
    outlier_bad_fraction_threshold: float = 0.08,
    spline_config_name: str = "baseline_spline_tene",
    spline_ne_profile_name: str = "electron_density",
    spline_te_profile_name: str = "electron_temperature",
    ne_anchor_filename: str = "ne_spline_anchor_vectors.csv",
    te_anchor_filename: str = "te_spline_anchor_vectors.csv",
    spline_fit_summary_filename: str = "te_ne_spline_fit_summary.csv",
    spline_fit_spec_filename: str = "te_ne_spline_fit_spec.json",
) -> dict[str, Any]:
    """Read real TS Te/Ne profiles, plasma-gated, and plot middle-time overlays."""
    if tstart < DEFAULT_TSTART or tend > DEFAULT_TEND:
        raise ValueError(
            f"Requested time window [{tstart}, {tend}] is outside allowed "
            f"[{DEFAULT_TSTART}, {DEFAULT_TEND}] s."
        )
    if tstart >= tend:
        raise ValueError(
            f"Invalid time window: tstart={tstart} must be smaller than tend={tend}."
        )

    pulse_list = [int(p) for p in (pulses or [])]
    if not pulse_list:
        pulse_list = [13622]

    ne_dataset = build_real_node_profile_dataset_task(
        pulses=pulse_list,
        node=ne_node,
        output_dir=output_dir,
        profile_filename=ne_filename,
        meta_filename=ne_meta_filename,
        tstart=tstart,
        tend=tend,
        dt=dt,
        read_verbose=read_verbose,
        min_finite_fraction=min_finite_fraction,
    )
    te_dataset = build_real_node_profile_dataset_task(
        pulses=pulse_list,
        node=te_node,
        output_dir=output_dir,
        profile_filename=te_filename,
        meta_filename=te_meta_filename,
        tstart=tstart,
        tend=tend,
        dt=dt,
        read_verbose=read_verbose,
        min_finite_fraction=min_finite_fraction,
    )

    outputs = align_and_plot_te_ne_profiles_task(
        ne_dataset=ne_dataset,
        te_dataset=te_dataset,
        output_dir=output_dir,
        plot_filename=plot_filename,
        ne_aligned_filename=ne_aligned_filename,
        te_aligned_filename=te_aligned_filename,
        matched_meta_filename=matched_meta_filename,
        outlier_report_filename=outlier_report_filename,
        apply_outlier_filter=apply_outlier_filter,
        outlier_point_z_threshold=outlier_point_z_threshold,
        outlier_extreme_point_z_threshold=outlier_extreme_point_z_threshold,
        outlier_min_bad_points=outlier_min_bad_points,
        outlier_bad_fraction_threshold=outlier_bad_fraction_threshold,
    )
    spline_fit = fit_te_ne_to_spline_anchor_space_task(
        ne_aligned_path=outputs["ne_aligned_path"],
        te_aligned_path=outputs["te_aligned_path"],
        matched_meta_path=outputs["matched_meta_path"],
        output_dir=output_dir,
        config_name=spline_config_name,
        ne_profile_name=spline_ne_profile_name,
        te_profile_name=spline_te_profile_name,
        ne_anchor_filename=ne_anchor_filename,
        te_anchor_filename=te_anchor_filename,
        fit_summary_filename=spline_fit_summary_filename,
        fit_spec_filename=spline_fit_spec_filename,
    )
    outputs["spline_fit"] = spline_fit

    return {
        "pulses_requested": pulse_list,
        "n_requested": int(len(pulse_list)),
        "ne_dataset": ne_dataset,
        "te_dataset": te_dataset,
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = real_tene_clustering(pulses=list(range(14500,14600)))
    print("Real Te/Ne clustering read pass complete")
    print(f"Matched pulses: {result['outputs']['num_matched']}/{result['n_requested']}")
    print(f"Outputs: {result['outputs']}")
