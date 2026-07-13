"""Shared utilities for expanded-equilibria synthetic dataset generation."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.workflows.jussiphd.components.data.data_generation import (
    _normalise_transform_beamlets,
    sample_plasma,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)


DEFAULT_EQUILIBRIUM_SPECS: tuple[dict[str, Any], ...] = (
    {"pulse": 11419, "tstart": 0.020, "tend": 0.160, "dt": 0.010, "label": "p11419_20to160ms"},
    {"pulse": 14606, "tstart": 0.015, "tend": 0.080, "dt": 0.010, "label": "p14606_15to80ms"},
)


def load_eps_matrix(eps_path: str) -> np.ndarray:
    eps = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
    if eps.ndim == 1:
        eps = eps[None, :]
    return eps


def equilibrium_time_grid(
    equilibrium: Any,
    tstart: float,
    tend: float,
    dt: float,
    n_timepoints: int | None = None,
) -> np.ndarray:
    raw = getattr(equilibrium, "t", None)
    if raw is None and hasattr(equilibrium, "rhop") and hasattr(equilibrium.rhop, "t"):
        raw = equilibrium.rhop.t
    if raw is None:
        n_steps = int(round((float(tend) - float(tstart)) / float(dt))) + 1
        return np.linspace(float(tstart), float(tend), max(1, n_steps), dtype=np.float64)

    eq_t = raw.values if hasattr(raw, "values") else raw
    eq_t = np.asarray(eq_t, dtype=float).reshape(-1)
    eq_t = eq_t[np.isfinite(eq_t)]
    target_t = eq_t[(eq_t >= float(tstart)) & (eq_t <= float(tend))]
    if target_t.size == 0:
        raise ValueError(
            "No equilibrium times in requested interval "
            f"[{float(tstart):.6f}, {float(tend):.6f}]"
        )
    if n_timepoints is not None and int(n_timepoints) > 0 and target_t.size > int(n_timepoints):
        idx = np.linspace(0, target_t.size - 1, int(n_timepoints), dtype=int)
        target_t = target_t[np.unique(idx)]
    return target_t


def _prepare_output(
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
) -> tuple[Path, Path, Path, dict[str, Any] | None]:
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
        return b_path, eps_out_path, meta_path, {
            "b_path": str(b_path),
            "eps_path": str(eps_out_path),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "b_shape": tuple(b_arr.shape),
            "eps_shape": tuple(eps_arr.shape),
            "num_pairs": int(b_arr.shape[0]),
            "generated_new_data": False,
        }

    return b_path, eps_out_path, meta_path, None


def _write_outputs(
    b_path: Path,
    eps_out_path: Path,
    meta_path: Path,
    b_rows: list[np.ndarray],
    eps_rows: list[np.ndarray],
    meta_rows: list[dict[str, Any]],
    meta_fields: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    b_arr = np.asarray(b_rows, dtype=np.float32)
    eps_arr = np.asarray(eps_rows, dtype=np.float32)

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_out_path, eps_arr, delimiter=",")

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_fields)
        writer.writeheader()
        writer.writerows(meta_rows)
    return b_arr, eps_arr


def build_equilibrium_contexts(
    machine: str,
    instrument: str,
    equilibrium_specs: list[dict[str, Any]],
    n_timepoints_per_equilibrium: int,
) -> list[dict[str, Any]]:
    transforms = load_default_objects(machine, "geometry")
    base_transform = transforms[instrument]
    base_transform.spot_shape = "square"
    base_transform.focal_length = -1000.0
    _normalise_transform_beamlets(base_transform)

    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    contexts: list[dict[str, Any]] = []
    for eq_idx, spec in enumerate(equilibrium_specs):
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=int(spec["pulse"]),
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            verbose=False,
        )
        transform = deepcopy(base_transform)
        transform.set_equilibrium(equilibrium, force=True)
        target_t = equilibrium_time_grid(
            equilibrium=equilibrium,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            n_timepoints=int(n_timepoints_per_equilibrium),
        )
        model = PinholeCamera(instrument, power_loss=power_loss)
        model.set_transform(transform)
        contexts.append(
            {
                "eq_idx": int(eq_idx),
                "spec": spec,
                "equilibrium": equilibrium,
                "transform": transform,
                "target_t": target_t,
                "model": model,
            }
        )
    return contexts


def brightness_matrix_from_emissivity(
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
        b_arr = b_arr.reshape(b_arr.shape[0], -1)
    return b_arr


def expand_brightness_with_equilibria(
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    n_timepoints_per_equilibrium: int,
    equilibrium_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    specs = list(equilibrium_specs) if equilibrium_specs is not None else list(DEFAULT_EQUILIBRIUM_SPECS)
    b_path, eps_out_path, meta_path, reuse = _prepare_output(
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
    )
    if reuse is not None:
        return reuse

    eps = load_eps_matrix(eps_path)
    n_samples, n_rhop = int(eps.shape[0]), int(eps.shape[1])
    rhop = np.linspace(0.0, 1.0, n_rhop, dtype=np.float32)
    contexts = build_equilibrium_contexts(
        machine=machine,
        instrument=instrument,
        equilibrium_specs=specs,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
    )

    b_rows: list[np.ndarray] = []
    eps_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for ctx in contexts:
        transform = ctx["transform"]
        spec = ctx["spec"]
        target_t = ctx["target_t"]
        eq_idx = int(ctx["eq_idx"])

        for eps_idx in range(n_samples):
            eps_row = eps[eps_idx]
            b_matrix = brightness_matrix_from_emissivity(
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
                        "equilibrium_index": eq_idx,
                        "equilibrium_label": str(spec["label"]),
                        "pulse": int(spec["pulse"]),
                        "t_s": float(t_value),
                    }
                )

    b_arr, eps_arr = _write_outputs(
        b_path=b_path,
        eps_out_path=eps_out_path,
        meta_path=meta_path,
        b_rows=b_rows,
        eps_rows=eps_rows,
        meta_rows=meta_rows,
        meta_fields=[
            "expanded_index",
            "source_eps_index",
            "equilibrium_index",
            "equilibrium_label",
            "pulse",
            "t_s",
        ],
    )

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_out_path),
        "meta_path": str(meta_path),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "num_pairs": int(b_arr.shape[0]),
        "num_source_eps": int(n_samples),
        "num_equilibrium_specs": int(len(specs)),
        "n_timepoints_per_equilibrium": int(n_timepoints_per_equilibrium),
        "generated_new_data": True,
    }


def _interp_plasma_fz_to_times(plasma: Any, target_t: np.ndarray) -> dict[Any, Any]:
    if not hasattr(plasma, "fz") or plasma.fz is None:
        return {}
    aligned: dict[Any, Any] = {}
    for elem, fz_da in plasma.fz.items():
        aligned[elem] = fz_da.interp(t=target_t, method="nearest") if hasattr(fz_da, "interp") else fz_da
    return aligned


def build_sampled_plasma_expanded_equilibria_dataset(
    output_dir: str,
    machine: str,
    instrument: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    n_timepoints_per_equilibrium: int,
    n_generations: int,
    config_name: str,
    config_overrides: list[str] | None,
    equilibrium_specs: list[dict[str, Any]] | None = None,
    impurity_concentrations: dict[str, float] | None = None,
    impurity_flat_zeff: bool = True,
) -> dict[str, Any]:
    specs = list(equilibrium_specs) if equilibrium_specs is not None else list(DEFAULT_EQUILIBRIUM_SPECS)
    b_path, eps_out_path, meta_path, reuse = _prepare_output(
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
    )
    if reuse is not None:
        return reuse

    contexts = build_equilibrium_contexts(
        machine=machine,
        instrument=instrument,
        equilibrium_specs=specs,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
    )
    sampling_overrides = (
        list(config_overrides)
        if config_overrides is not None
        else ["plasma.settings.n_rad=41", "tstart=0.015", "tend=0.160", "dt=0.005"]
    )
    reference_model = contexts[0]["model"]
    reference_transform = contexts[0]["transform"]

    b_rows: list[np.ndarray] = []
    eps_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for src_idx in range(int(n_generations)):
        plasma = sample_plasma(
            model=reference_model,
            transform=reference_transform,
            config_name=config_name,
            overrides=sampling_overrides,
        )
        if impurity_concentrations is not None:
            for element, concentration in impurity_concentrations.items():
                plasma.set_impurity_concentration(
                    element=str(element),
                    concentration=float(concentration),
                    flat_zeff=bool(impurity_flat_zeff),
                )
        base_fz = {elem: fz_da.copy(deep=True) for elem, fz_da in plasma.fz.items()}

        for ctx in contexts:
            model = ctx["model"]
            target_t = ctx["target_t"]
            spec = ctx["spec"]
            # `plasma.fz` is a property without a setter; update element entries in-place.
            for elem, fz_da in base_fz.items():
                plasma.fz[elem] = fz_da.copy(deep=True)
            aligned_fz = _interp_plasma_fz_to_times(plasma, target_t)
            for elem, fz_da in aligned_fz.items():
                plasma.fz[elem] = fz_da
            model.set_plasma(plasma)
            bckc, emissivity = model(t=target_t, return_emissivity=True)
            brightness = bckc["brightness"]

            for tidx, t_value in enumerate(target_t):
                b_rows.append(brightness.isel(t=tidx).values.astype(np.float32).reshape(-1))
                eps_rows.append(emissivity.isel(t=tidx).values.astype(np.float32).reshape(-1))
                meta_rows.append(
                    {
                        "expanded_index": len(meta_rows),
                        "source_sample_index": int(src_idx),
                        "equilibrium_index": int(ctx["eq_idx"]),
                        "equilibrium_label": str(spec["label"]),
                        "pulse": int(spec["pulse"]),
                        "t_s": float(t_value),
                    }
                )

    b_arr, eps_arr = _write_outputs(
        b_path=b_path,
        eps_out_path=eps_out_path,
        meta_path=meta_path,
        b_rows=b_rows,
        eps_rows=eps_rows,
        meta_rows=meta_rows,
        meta_fields=[
            "expanded_index",
            "source_sample_index",
            "equilibrium_index",
            "equilibrium_label",
            "pulse",
            "t_s",
        ],
    )

    result: dict[str, Any] = {
        "b_path": str(b_path),
        "eps_path": str(eps_out_path),
        "meta_path": str(meta_path),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "num_pairs": int(b_arr.shape[0]),
        "num_source_samples": int(n_generations),
        "num_equilibrium_specs": int(len(specs)),
        "n_timepoints_per_equilibrium": int(n_timepoints_per_equilibrium),
        "generated_new_data": True,
    }
    if impurity_concentrations is not None:
        result["impurity_concentrations"] = {
            str(k): float(v) for k, v in impurity_concentrations.items()
        }
        result["impurity_flat_zeff"] = bool(impurity_flat_zeff)
    return result


def save_equilibrium_plots(
    output_dir: str,
    n_timepoints_per_equilibrium: int,
    equilibrium_specs: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    specs = list(equilibrium_specs) if equilibrium_specs is not None else list(DEFAULT_EQUILIBRIUM_SPECS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for spec in specs:
        pulse = int(spec["pulse"])
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=pulse,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            verbose=False,
        )
        t_values = equilibrium_time_grid(
            equilibrium=equilibrium,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            n_timepoints=int(n_timepoints_per_equilibrium),
        )
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_values)))
        for idx, t in enumerate(t_values):
            rb = equilibrium.rbnd.interp(t=float(t), method="nearest")
            zb = equilibrium.zbnd.interp(t=float(t), method="nearest")
            ax.plot(rb.values, zb.values, color=colors[idx], linewidth=1.8, label=f"t={float(t):.3f}s")
        ax.set_title(f"Equilibrium Boundaries (pulse {pulse})")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out / f"equilibrium_pulse_{pulse}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved[f"pulse_{pulse}"] = str(path)
    return saved
