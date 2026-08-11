"""Prefect flow to read and visualize real ST40 Te/Ne profiles by pulse."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from prefect import flow, task

from indica.workflows.jussiphd.components.data.real_brightness_dataset_generation import (
    generate_and_save_real_multipulse_brightness_dataset,
)

TS_NE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:NE"
TS_TE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:TE"
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


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
    plot_path = out / plot_filename

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
    }


@flow(name="real_tene_clustering")
def real_tene_clustering(
    pulses: Sequence[int] | None = None,
    tstart: float = 0.04,
    tend: float = 0.1,
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
    min_finite_fraction: float = 0.90,
) -> dict[str, Any]:
    """Read real TS Te/Ne profiles, plasma-gated, and plot middle-time overlays."""
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
    )

    return {
        "pulses_requested": pulse_list,
        "n_requested": int(len(pulse_list)),
        "ne_dataset": ne_dataset,
        "te_dataset": te_dataset,
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = real_tene_clustering(pulses=list(range(14500,14700)))
    print("Real Te/Ne clustering read pass complete")
    print(f"Matched pulses: {result['outputs']['num_matched']}/{result['n_requested']}")
    print(f"Outputs: {result['outputs']}")
