"""Shared slice filtering for real-pulse visualisation/evaluation workflows."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from indica.workflows.jussiphd.components.data.data_generation import (
    build_real_model_for_pulse,
)
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset


def build_shared_visualization_slice_pool(
    b_path: str,
    eps_path: str,
    meta_path: str,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    min_valid_channels_required: int = 3,
    candidate_indices: list[int] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Build notebook-style shared filtered slice records using pulse/time metadata.

    Returns only lightweight, serializable structures suitable for orchestration.
    """
    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=meta_path)

    with open(meta_path, newline="") as f:
        rows = list(csv.reader(f))
    has_header = bool(rows) and rows[0] == ["pulse", "time_s"]
    meta_rows = rows[1:] if has_header else rows

    if len(meta_rows) != len(dataset):
        raise ValueError(
            f"meta row count ({len(meta_rows)}) != dataset length ({len(dataset)}). "
            "Rebuild aligned CSV/meta first."
        )

    if candidate_indices is None:
        candidate_indices = list(range(len(dataset)))

    pulse_cache: dict[int, dict[str, Any]] = {}

    def load_pulse_bundle(pulse: int) -> dict[str, Any]:
        pulse = int(pulse)
        if pulse not in pulse_cache:
            model = build_real_model_for_pulse(
                pulse=pulse,
                machine=machine,
                instrument=instrument,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
            bckc, emissivity = model(return_emissivity=True)
            pulse_cache[pulse] = {
                "brightness": bckc["brightness"],
                "emissivity": emissivity,
            }
        return pulse_cache[pulse]

    stats = {
        "attempted": 0,
        "kept": 0,
        "skipped_bad_meta": 0,
        "skipped_load_fail": 0,
        "skipped_empty_time": 0,
        "skipped_invalid_channels": 0,
    }

    visualization_slice_pool: list[dict[str, Any]] = []

    for idx in candidate_indices:
        stats["attempted"] += 1

        if idx < 0 or idx >= len(meta_rows):
            stats["skipped_bad_meta"] += 1
            continue

        try:
            pulse = int(float(meta_rows[idx][0]))
            t_target = float(meta_rows[idx][1])
        except Exception:
            stats["skipped_bad_meta"] += 1
            continue

        try:
            bundle = load_pulse_bundle(pulse)
        except Exception:
            stats["skipped_load_fail"] += 1
            continue

        brightness = bundle["brightness"]
        t_values = np.asarray(brightness.t.values, dtype=float)
        if t_values.size == 0:
            stats["skipped_empty_time"] += 1
            continue

        t_idx = int(np.argmin(np.abs(t_values - t_target)))
        b_vec = brightness.isel(t=t_idx).values.astype(np.float32)
        n_valid_channels = int(np.sum(np.isfinite(b_vec) & (b_vec > 0.0)))

        if n_valid_channels < int(min_valid_channels_required):
            stats["skipped_invalid_channels"] += 1
            continue

        visualization_slice_pool.append(
            {
                "idx": int(idx),
                "pulse": int(pulse),
                "t_target": float(t_target),
                "t_idx": int(t_idx),
                "n_valid_channels": n_valid_channels,
            }
        )

        if max_samples is not None and len(visualization_slice_pool) >= int(max_samples):
            break

    stats["kept"] = len(visualization_slice_pool)

    pulse_to_t_indices: dict[int, list[int]] = {}
    for rec in visualization_slice_pool:
        pulse_to_t_indices.setdefault(int(rec["pulse"]), []).append(int(rec["t_idx"]))

    pulse_to_t_indices = {
        pulse: sorted(set(indices)) for pulse, indices in pulse_to_t_indices.items()
    }

    return {
        "visualization_slice_pool": visualization_slice_pool,
        "visualization_pulse_to_t_indices": pulse_to_t_indices,
        "filter_stats": stats,
        "num_pulses": int(len(pulse_to_t_indices)),
        "min_valid_channels_required": int(min_valid_channels_required),
    }
