"""Prefect flow: expand synthetic brightness with multiple real equilibria."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, task
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.data_generation import (
    _normalise_transform_beamlets,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.datasets.paths import MULTIPULSE_SYNTHETIC_DATA_DIR
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR,
)


DEFAULT_INPUT_EPS_PATH = str(MULTIPULSE_SYNTHETIC_DATA_DIR / "eps_slices_multipulse_synthetic.csv")
DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR


def _load_eps_matrix(eps_path: str) -> np.ndarray:
    eps = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
    if eps.ndim == 1:
        eps = eps[None, :]
    return eps


def _equilibrium_time_grid(equilibrium: Any, tstart: float, tend: float, dt: float) -> np.ndarray:
    raw = getattr(equilibrium, "t", None)
    if raw is None and hasattr(equilibrium, "rhop") and hasattr(equilibrium.rhop, "t"):
        raw = equilibrium.rhop.t
    if raw is None:
        # Conservative fallback if equilibrium has no explicit time coordinate.
        n_steps = int(round((float(tend) - float(tstart)) / float(dt))) + 1
        return np.linspace(float(tstart), float(tend), max(1, n_steps), dtype=np.float32)

    eq_t = raw.values if hasattr(raw, "values") else raw
    eq_t = np.asarray(eq_t, dtype=float).reshape(-1)
    eq_t = eq_t[np.isfinite(eq_t)]
    target_t = eq_t[(eq_t >= float(tstart)) & (eq_t <= float(tend))]
    if target_t.size == 0:
        raise ValueError(
            "No equilibrium times in requested interval "
            f"[{float(tstart):.6f}, {float(tend):.6f}]"
        )
    return target_t.astype(np.float32)


def _brightness_matrix_from_emissivity(
    transform: Any,
    eps_row: np.ndarray,
    target_t: np.ndarray,
    rhop: np.ndarray,
) -> np.ndarray:
    emissivity = DataArray(
        np.repeat(eps_row[None, :], int(target_t.size), axis=0),
        coords=[("t", target_t), ("rhop", rhop)],
    )
    brightness = transform.integrate_on_los(emissivity, t=emissivity.t)
    b_arr = np.asarray(brightness.values, dtype=np.float32)
    if b_arr.ndim == 1:
        b_arr = b_arr[None, :]
    elif b_arr.ndim > 2:
        # Should be avoided by beamlet normalization, but keep robust fallback.
        b_arr = b_arr.reshape(b_arr.shape[0], -1)
    return b_arr


@task(name="expand_brightness_with_equilibria")
def expand_brightness_with_equilibria_task(
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_out_path = output_path / eps_filename
    meta_path = output_path / meta_filename

    if not generate_new_data:
        if not b_path.exists() or not eps_out_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but output files do not exist: "
                f"{b_path}, {eps_out_path}"
            )
        b_arr = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        eps_arr = np.loadtxt(eps_out_path, delimiter=",", dtype=np.float32)
        if b_arr.ndim == 1:
            b_arr = b_arr[None, :]
        if eps_arr.ndim == 1:
            eps_arr = eps_arr[None, :]
        return {
            "b_path": str(b_path),
            "eps_path": str(eps_out_path),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "b_shape": tuple(b_arr.shape),
            "eps_shape": tuple(eps_arr.shape),
            "num_pairs": int(b_arr.shape[0]),
            "generated_new_data": False,
        }

    eps = _load_eps_matrix(eps_path)
    n_samples, n_rhop = int(eps.shape[0]), int(eps.shape[1])
    rhop = np.linspace(0.0, 1.0, n_rhop, dtype=np.float32)

    equilibrium_specs = [
        {"pulse": 11419, "tstart": 0.020, "tend": 0.160, "dt": 0.010, "label": "p11419_20to160ms"},
        {"pulse": 14606, "tstart": 0.015, "tend": 0.080, "dt": 0.010, "label": "p14606_15to80ms"},
    ]

    transforms = load_default_objects(machine, "geometry")
    transform = transforms[instrument]
    transform.spot_shape = "square"
    transform.focal_length = -1000.0
    _normalise_transform_beamlets(transform)

    b_rows: list[np.ndarray] = []
    eps_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for eq_idx, spec in enumerate(equilibrium_specs):
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=int(spec["pulse"]),
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            verbose=False,
        )
        transform.set_equilibrium(equilibrium, force=True)
        target_t = _equilibrium_time_grid(
            equilibrium=equilibrium,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
        )

        for eps_idx in range(n_samples):
            eps_row = eps[eps_idx]
            b_matrix = _brightness_matrix_from_emissivity(
                transform=transform,
                eps_row=eps_row,
                target_t=target_t,
                rhop=rhop,
            )
            for tidx, t_value in enumerate(target_t):
                b_rows.append(b_matrix[tidx].astype(np.float32))
                eps_rows.append(eps_row.astype(np.float32))
                meta_rows.append(
                    {
                        "expanded_index": len(meta_rows),
                        "source_eps_index": int(eps_idx),
                        "equilibrium_index": int(eq_idx),
                        "equilibrium_label": str(spec["label"]),
                        "pulse": int(spec["pulse"]),
                        "t_s": float(t_value),
                    }
                )

    b_arr = np.asarray(b_rows, dtype=np.float32)
    eps_arr = np.asarray(eps_rows, dtype=np.float32)

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_out_path, eps_arr, delimiter=",")

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "expanded_index",
                "source_eps_index",
                "equilibrium_index",
                "equilibrium_label",
                "pulse",
                "t_s",
            ],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_out_path),
        "meta_path": str(meta_path),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "num_pairs": int(b_arr.shape[0]),
        "num_source_eps": int(n_samples),
        "num_equilibrium_specs": int(len(equilibrium_specs)),
        "generated_new_data": True,
    }


@flow(name="build_multipulse_synthetic_expanded_equilibria_dataset")
def build_multipulse_synthetic_expanded_equilibria_dataset(
    eps_path: str = DEFAULT_INPUT_EPS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria.csv",
    meta_filename: str = "sample_meta_multipulse_synthetic_expanded_equilibria.csv",
    generate_new_data: bool = True,
) -> dict[str, Any]:
    """Expand brightness by projecting fixed synthetic eps over multiple equilibria."""
    return expand_brightness_with_equilibria_task(
        eps_path=eps_path,
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
    )


if __name__ == "__main__":
    result = build_multipulse_synthetic_expanded_equilibria_dataset()
    print("Expanded-equilibria synthetic dataset complete")
    print(result)

