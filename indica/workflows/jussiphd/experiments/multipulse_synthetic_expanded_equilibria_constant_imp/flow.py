"""Prefect flow: expanded-equilibria synthetic dataset with fixed impurities."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.workflows.jussiphd.components.data.data_generation import (
    _normalise_transform_beamlets,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR,
)
from indica.workflows.jussiphd.plasma_profiler_init import (
    build_plasma_profiler,
    load_bda_config,
    sample_prior_parameters,
)


DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR
EQUILIBRIUM_SPECS = [
    {"pulse": 11419, "tstart": 0.020, "tend": 0.160, "dt": 0.010, "label": "p11419_20to160ms"},
    {"pulse": 14606, "tstart": 0.015, "tend": 0.080, "dt": 0.010, "label": "p14606_15to80ms"},
]


def _equilibrium_time_grid(
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
    target_t = target_t.astype(np.float32)
    if n_timepoints is not None and int(n_timepoints) > 0 and target_t.size > int(n_timepoints):
        idx = np.linspace(0, target_t.size - 1, int(n_timepoints), dtype=int)
        idx = np.unique(idx)
        target_t = target_t[idx]
    return target_t


@task(name="save_equilibrium_plots")
def save_equilibrium_plots_task(
    output_dir: str,
    n_timepoints_per_equilibrium: int,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for spec in EQUILIBRIUM_SPECS:
        pulse = int(spec["pulse"])
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=pulse,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            verbose=False,
        )
        t_values = _equilibrium_time_grid(
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


@task(name="build_expanded_equilibria_constant_imp_dataset")
def build_expanded_equilibria_constant_imp_dataset_task(
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
    c_concentration: float,
    ar_concentration: float,
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

    transforms = load_default_objects(machine, "geometry")
    base_transform = transforms[instrument]
    base_transform.spot_shape = "square"
    base_transform.focal_length = -1000.0
    _normalise_transform_beamlets(base_transform)

    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    eq_contexts: list[dict[str, Any]] = []
    for eq_idx, spec in enumerate(EQUILIBRIUM_SPECS):
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=int(spec["pulse"]),
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            verbose=False,
        )
        transform = deepcopy(base_transform)
        transform.set_equilibrium(equilibrium, force=True)
        target_t = _equilibrium_time_grid(
            equilibrium=equilibrium,
            tstart=float(spec["tstart"]),
            tend=float(spec["tend"]),
            dt=float(spec["dt"]),
            n_timepoints=int(n_timepoints_per_equilibrium),
        )
        model = PinholeCamera(instrument, power_loss=power_loss)
        model.set_transform(transform)
        eq_contexts.append(
            {
                "eq_idx": int(eq_idx),
                "spec": spec,
                "target_t": target_t,
                "model": model,
            }
        )

    cfg = load_bda_config(
        config_name=config_name,
        overrides=(
            list(config_overrides)
            if config_overrides is not None
            else [
                "plasma.settings.n_rad=41",
                "tstart=0.015",
                "tend=0.160",
                "dt=0.005",
            ]
        ),
    )
    plasma_profiler = build_plasma_profiler(cfg)

    b_rows: list[np.ndarray] = []
    eps_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for src_idx in range(int(n_generations)):
        sampled_params = sample_prior_parameters(cfg)
        plasma_profiler(sampled_params)
        plasma = plasma_profiler.plasma

        plasma.set_impurity_concentration(
            element="c",
            concentration=float(c_concentration),
            flat_zeff=True,
        )
        plasma.set_impurity_concentration(
            element="ar",
            concentration=float(ar_concentration),
            flat_zeff=True,
        )

        for ctx in eq_contexts:
            model = ctx["model"]
            target_t = ctx["target_t"]
            spec = ctx["spec"]
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

    b_arr = np.asarray(b_rows, dtype=np.float32)
    eps_arr = np.asarray(eps_rows, dtype=np.float32)

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_out_path, eps_arr, delimiter=",")

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "expanded_index",
                "source_sample_index",
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
        "num_source_samples": int(n_generations),
        "num_equilibrium_specs": int(len(EQUILIBRIUM_SPECS)),
        "n_timepoints_per_equilibrium": int(n_timepoints_per_equilibrium),
        "c_concentration": float(c_concentration),
        "ar_concentration": float(ar_concentration),
        "generated_new_data": True,
    }


@flow(name="build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset")
def build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    meta_filename: str = "sample_meta_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    generate_new_data: bool = True,
    n_timepoints_per_equilibrium: int = 5,
    n_generations: int = 3000,
    config_name: str = "ion_temperature_phantom_run_all_params",
    config_overrides: list[str] | None = None,
    c_concentration: float = 0.05,
    ar_concentration: float = 0.01,
    save_equilibrium_plots: bool = True,
    equilibrium_plots_dir: str | None = None,
) -> dict[str, Any]:
    """
    Build expanded-equilibria dataset with fixed impurity concentrations.

    For each sampled plasma:
      1) enforce C=5% and Ar=1% using `set_impurity_concentration(..., flat_zeff=True)`
      2) run LOS forward model across multiple equilibria/timepoints
      3) store one eps with multiple b observations.
    """
    dataset_result = build_expanded_equilibria_constant_imp_dataset_task(
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        n_generations=n_generations,
        config_name=config_name,
        config_overrides=config_overrides,
        c_concentration=c_concentration,
        ar_concentration=ar_concentration,
    )
    plots_result = None
    if save_equilibrium_plots:
        plots_result = save_equilibrium_plots_task(
            output_dir=(
                equilibrium_plots_dir
                if equilibrium_plots_dir is not None
                else str(Path(output_dir) / "equilibrium_plots")
            ),
            n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        )
    return {
        "dataset": dataset_result,
        "equilibrium_plots": plots_result,
    }


if __name__ == "__main__":
    result = build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset()
    print("Constant-impurity expanded-equilibria synthetic dataset complete")
    print(result)
