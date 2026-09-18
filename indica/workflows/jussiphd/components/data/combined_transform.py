"""Helpers to build a single LOS transform from multiple instruments."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from indica.converters.line_of_sight import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects


def _vec(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty LOS vector for '{label}'.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite LOS values in '{label}'.")
    return arr


def build_combined_los_transform(
    machine: str,
    instruments: Sequence[str],
    *,
    combined_name: str = "combined_los",
) -> tuple[LineOfSightTransform, list[dict[str, int | str]]]:
    """Concatenate LOS geometry from multiple instruments into one transform."""
    ordered = [str(i) for i in instruments]
    if not ordered:
        raise ValueError("instruments must contain at least one instrument name.")

    transforms = load_default_objects(machine, "geometry")
    missing = [name for name in ordered if name not in transforms]
    if missing:
        raise KeyError(f"Unknown instrument(s) for machine='{machine}': {missing}")

    base = deepcopy(transforms[ordered[0]])
    machine_dims = getattr(base, "_machine_dims", ((1.83, 3.9), (-1.75, 2.0)))
    dl = float(getattr(base, "dl", 0.01))
    passes = int(getattr(base, "passes", 1))
    beamlets_method = str(getattr(base, "beamlets_method", "simple"))
    n_beamlets = int(getattr(base, "beamlets", 1))
    spot_width = float(getattr(base, "spot_width", 0.0))
    spot_height = float(getattr(base, "spot_height", 0.0))
    spot_shape = str(getattr(base, "spot_shape", "square"))
    focal_length = float(getattr(base, "focal_length", -1000.0))

    ox_parts: list[np.ndarray] = []
    oy_parts: list[np.ndarray] = []
    oz_parts: list[np.ndarray] = []
    dx_parts: list[np.ndarray] = []
    dy_parts: list[np.ndarray] = []
    dz_parts: list[np.ndarray] = []
    channel_map: list[dict[str, int | str]] = []

    combined_idx = 0
    for instrument in ordered:
        tr = transforms[instrument]
        ox = _vec(getattr(tr, "origin_x"), f"{instrument}.origin_x")
        oy = _vec(getattr(tr, "origin_y"), f"{instrument}.origin_y")
        oz = _vec(getattr(tr, "origin_z"), f"{instrument}.origin_z")
        dx = _vec(getattr(tr, "direction_x"), f"{instrument}.direction_x")
        dy = _vec(getattr(tr, "direction_y"), f"{instrument}.direction_y")
        dz = _vec(getattr(tr, "direction_z"), f"{instrument}.direction_z")

        n_ch = int(ox.size)
        for vec in (oy, oz, dx, dy, dz):
            if int(vec.size) != n_ch:
                raise ValueError(f"Channel count mismatch inside instrument '{instrument}'.")

        ox_parts.append(ox)
        oy_parts.append(oy)
        oz_parts.append(oz)
        dx_parts.append(dx)
        dy_parts.append(dy)
        dz_parts.append(dz)

        for src_idx in range(n_ch):
            channel_map.append(
                {
                    "combined_channel_index": int(combined_idx),
                    "source_instrument": str(instrument),
                    "source_channel_index": int(src_idx),
                }
            )
            combined_idx += 1

    combined = LineOfSightTransform(
        origin_x=np.concatenate(ox_parts),
        origin_y=np.concatenate(oy_parts),
        origin_z=np.concatenate(oz_parts),
        direction_x=np.concatenate(dx_parts),
        direction_y=np.concatenate(dy_parts),
        direction_z=np.concatenate(dz_parts),
        name=combined_name,
        machine_dimensions=machine_dims,
        dl=dl,
        passes=passes,
        beamlets_method=beamlets_method,
        n_beamlets=max(1, n_beamlets),
        spot_width=spot_width,
        spot_height=spot_height,
        spot_shape=spot_shape,
        focal_length=focal_length,
        plot_beamlets=False,
    )
    return combined, channel_map


def save_combined_channel_map(
    channel_map: Sequence[dict[str, int | str]],
    output_dir: str,
    filename: str,
) -> str:
    """Persist combined-channel mapping as CSV for traceability."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / filename
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "combined_channel_index",
                "source_instrument",
                "source_channel_index",
            ],
        )
        writer.writeheader()
        writer.writerows(channel_map)
    return str(csv_path)

