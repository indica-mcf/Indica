"""Build fixed-length equilibrium-boundary datasets from real pulse lists."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from indica.workflows.jussiphd.components.data.expanded_equilibria_generation import (
    equilibrium_time_grid,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}.")


def load_non_outlier_pulses_from_report(
    outlier_report_path: str,
    deduplicate: bool = True,
) -> list[int]:
    """Return pulse list where `removed_as_outlier` is False in report CSV."""
    report = Path(outlier_report_path)
    if not report.exists():
        raise FileNotFoundError(f"Outlier report not found: {report}")

    pulses: list[int] = []
    with report.open(newline="") as f:
        reader = csv.DictReader(f)
        if "pulse" not in (reader.fieldnames or []):
            raise ValueError("Outlier report missing required 'pulse' column.")
        if "removed_as_outlier" not in (reader.fieldnames or []):
            raise ValueError("Outlier report missing required 'removed_as_outlier' column.")
        for row in reader:
            removed = _to_bool(row["removed_as_outlier"])
            if not removed:
                pulses.append(int(row["pulse"]))

    if deduplicate:
        unique: list[int] = []
        seen: set[int] = set()
        for pulse in pulses:
            if pulse in seen:
                continue
            seen.add(pulse)
            unique.append(int(pulse))
        pulses = unique
    return pulses


def _resample_closed_boundary(
    r: np.ndarray,
    z: np.ndarray,
    n_boundary_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an R-Z boundary to a fixed number of points by arclength."""
    rr = np.asarray(r, dtype=np.float64).reshape(-1)
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    valid = np.isfinite(rr) & np.isfinite(zz)
    rr = rr[valid]
    zz = zz[valid]
    if rr.size < 4:
        raise ValueError("Boundary has too few valid points.")

    if not (np.isclose(rr[0], rr[-1]) and np.isclose(zz[0], zz[-1])):
        rr = np.concatenate([rr, rr[:1]])
        zz = np.concatenate([zz, zz[:1]])

    dr = np.diff(rr)
    dz = np.diff(zz)
    ds = np.sqrt(dr * dr + dz * dz)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Boundary arclength is non-positive.")

    # Start at outboard-most point to keep start index stable across samples.
    x = np.linspace(0.0, total, int(n_boundary_points), endpoint=False, dtype=np.float64)
    r_interp = np.interp(x, s, rr)
    z_interp = np.interp(x, s, zz)
    start_idx = int(np.argmax(r_interp))
    r_interp = np.roll(r_interp, -start_idx)
    z_interp = np.roll(z_interp, -start_idx)

    # Enforce a consistent orientation (counter-clockwise).
    area2 = float(np.sum(r_interp * np.roll(z_interp, -1) - np.roll(r_interp, -1) * z_interp))
    if area2 < 0.0:
        r_interp = r_interp[::-1]
        z_interp = z_interp[::-1]
    return r_interp.astype(np.float32), z_interp.astype(np.float32)


def build_and_save_equilibrium_boundary_dataset(
    pulses: Sequence[int],
    output_dir: str,
    features_filename: str = "equilibrium_boundary_features.csv",
    meta_filename: str = "equilibrium_boundary_meta.csv",
    generate_new_data: bool = True,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    n_timepoints_per_equilibrium: int = 6,
    n_boundary_points: int = 128,
    verbose: bool = False,
    skip_failed_pulses: bool = True,
) -> dict[str, Any]:
    """Build clustered-feature matrix from equilibrium boundaries across pulses/timepoints."""
    if float(tstart) >= float(tend):
        raise ValueError(f"Invalid time window: tstart={tstart} must be smaller than tend={tend}.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_path = out / features_filename
    meta_path = out / meta_filename

    if not generate_new_data:
        if not features_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but expected files are missing: "
                f"{features_path}, {meta_path}"
            )
        features = np.loadtxt(features_path, delimiter=",", dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        with meta_path.open(newline="") as f:
            meta_rows = list(csv.DictReader(f))
        return {
            "features_path": str(features_path),
            "meta_path": str(meta_path),
            "features_shape": tuple(features.shape),
            "num_snapshots": int(features.shape[0]),
            "num_pulses_used": int(len({int(r["pulse"]) for r in meta_rows})) if meta_rows else 0,
            "generated_new_data": False,
            "n_boundary_points": int(n_boundary_points),
        }

    pulse_list = [int(p) for p in pulses]
    if not pulse_list:
        raise ValueError("No pulses provided for equilibrium boundary dataset generation.")

    feature_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    failed_pulses: list[int] = []
    used_pulses: list[int] = []

    for pulse in pulse_list:
        try:
            equilibrium = load_real_equilibrium_from_pulse(
                pulse=int(pulse),
                tstart=float(tstart),
                tend=float(tend),
                dt=float(dt),
                verbose=bool(verbose),
            )
            target_t = equilibrium_time_grid(
                equilibrium=equilibrium,
                tstart=float(tstart),
                tend=float(tend),
                dt=float(dt),
                n_timepoints=int(n_timepoints_per_equilibrium),
            )
        except Exception:
            failed_pulses.append(int(pulse))
            if not skip_failed_pulses:
                raise
            continue

        n_added = 0
        for tidx, t_val in enumerate(np.asarray(target_t, dtype=float).reshape(-1)):
            try:
                rb = np.asarray(equilibrium.rbnd.interp(t=float(t_val), method="nearest").values)
                zb = np.asarray(equilibrium.zbnd.interp(t=float(t_val), method="nearest").values)
                rr, zz = _resample_closed_boundary(
                    rb,
                    zb,
                    n_boundary_points=int(n_boundary_points),
                )
            except Exception:
                continue

            row = np.concatenate([rr, zz]).astype(np.float32)
            feature_rows.append(row)
            meta_rows.append(
                {
                    "sample_idx": int(len(meta_rows)),
                    "pulse": int(pulse),
                    "t_s": float(t_val),
                    "timepoint_index": int(tidx),
                    "n_timepoints_for_pulse": int(len(target_t)),
                }
            )
            n_added += 1

        if n_added > 0:
            used_pulses.append(int(pulse))
        else:
            failed_pulses.append(int(pulse))

    if not feature_rows:
        raise RuntimeError("No equilibrium boundary snapshots were created from the given pulses.")

    features = np.asarray(feature_rows, dtype=np.float32)
    np.savetxt(features_path, features, delimiter=",")

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "pulse", "t_s", "timepoint_index", "n_timepoints_for_pulse"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    return {
        "features_path": str(features_path),
        "meta_path": str(meta_path),
        "features_shape": tuple(features.shape),
        "num_snapshots": int(features.shape[0]),
        "num_pulses_requested": int(len(pulse_list)),
        "num_pulses_used": int(len(set(used_pulses))),
        "num_failed_pulses": int(len(set(failed_pulses))),
        "failed_pulses": sorted(set(int(p) for p in failed_pulses)),
        "generated_new_data": True,
        "tstart": float(tstart),
        "tend": float(tend),
        "dt": float(dt),
        "n_timepoints_per_equilibrium": int(n_timepoints_per_equilibrium),
        "n_boundary_points": int(n_boundary_points),
    }

