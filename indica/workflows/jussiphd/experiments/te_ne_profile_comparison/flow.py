"""Prefect flow to compare sampled Te/ne and model emissivity profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.data_generation import (
    generate_plasma_sample,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def _extract_profile_middle_slice(data: Any, label: str) -> np.ndarray:
    profile = data
    for dim in list(profile.dims):
        if dim == "rhop":
            continue
        size = int(profile.sizes[dim])
        if size <= 0:
            raise ValueError(f"{label} has empty dimension '{dim}'.")
        profile = profile.isel({dim: size // 2})

    values = np.asarray(profile.values, dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{label} did not reduce to 1D rhop profile; got shape {values.shape}.")
    return values


def _as_sample_matrix(values: np.ndarray, rhop: np.ndarray, label: str) -> np.ndarray:
    """Normalize sampled profile arrays to shape (n_samples, n_rhop)."""
    arr = np.asarray(values, dtype=np.float64)
    nr = int(np.asarray(rhop).size)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        if arr.size != nr:
            raise ValueError(
                f"{label} has shape {arr.shape}, expected length {nr}."
            )
        return arr[None, :]

    if arr.ndim == 2:
        if arr.shape[1] == nr:
            return arr
        if arr.shape[0] == nr:
            return arr.T
        raise ValueError(
            f"{label} has shape {arr.shape}, cannot align with rhop length {nr}."
        )

    # Handle any residual singleton axes robustly.
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == nr:
            return arr
        raise ValueError(
            f"{label} has shape {values.shape} (squeezed to {arr.shape}), "
            f"cannot align with rhop length {nr}."
        )

    raise ValueError(f"{label} has unsupported shape {arr.shape}.")


@task(name="sample_profiles_for_config")
def sample_profiles_for_config_task(
    config_name: str,
    n_samples: int,
    seed: int,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
) -> dict[str, Any]:
    # Keep runs deterministic and isolate config streams from each other.
    np.random.seed(int(seed))

    transforms = load_default_objects(machine, "geometry")
    transform = transforms[instrument]
    if use_real_equilibrium:
        equilibrium = load_real_equilibrium_from_pulse(
            pulse=real_equilibrium_pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=real_equilibrium_verbose,
        )
    else:
        equilibrium = load_default_objects(machine, "equilibrium")

    rhop = None
    te_samples: list[np.ndarray] = []
    ne_samples: list[np.ndarray] = []
    eps_samples: list[np.ndarray] = []

    for _ in range(int(n_samples)):
        sample = generate_plasma_sample(
            machine=machine,
            instrument=instrument,
            transform=transform,
            equilibrium=equilibrium,
            config_name=config_name,
        )
        plasma = sample["plasma"]
        emissivity = sample["emissivity"]

        te_profile = _extract_profile_middle_slice(plasma.electron_temperature, "electron_temperature")
        ne_profile = _extract_profile_middle_slice(plasma.electron_density, "electron_density")
        eps_profile = _extract_profile_middle_slice(emissivity, "model_emissivity")

        if rhop is None:
            rhop = np.asarray(plasma.rhop.values, dtype=np.float64).squeeze()
        te_samples.append(te_profile)
        ne_samples.append(ne_profile)
        eps_samples.append(eps_profile)

    if rhop is None:
        raise ValueError(f"No samples produced for config '{config_name}'.")

    return {
        "config_name": config_name,
        "n_samples": int(n_samples),
        "rhop": rhop,
        "electron_temperature": np.asarray(te_samples, dtype=np.float64),
        "electron_density": np.asarray(ne_samples, dtype=np.float64),
        "model_emissivity": np.asarray(eps_samples, dtype=np.float64),
    }


def _plot_overlay(
    rhop: np.ndarray,
    gaussian_samples: np.ndarray,
    spline_samples: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for curve in gaussian_samples:
        ax.plot(rhop, curve, color="#4C78A8", alpha=0.20, linewidth=1.1)
    for curve in spline_samples:
        ax.plot(rhop, curve, color="#F58518", alpha=0.20, linewidth=1.1)
    ax.plot([], [], color="#4C78A8", linewidth=2, label="Gaussian")
    ax.plot([], [], color="#F58518", linewidth=2, label="Spline")
    ax.set_xlabel("rhop")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_percentile_band(
    rhop: np.ndarray,
    gaussian_samples: np.ndarray,
    spline_samples: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    g_p10, g_p50, g_p90 = np.percentile(gaussian_samples, [10, 50, 90], axis=0)
    s_p10, s_p50, s_p90 = np.percentile(spline_samples, [10, 50, 90], axis=0)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.fill_between(rhop, g_p10, g_p90, color="#4C78A8", alpha=0.22, label="Gaussian p10-p90")
    ax.fill_between(rhop, s_p10, s_p90, color="#F58518", alpha=0.22, label="Spline p10-p90")
    ax.plot(rhop, g_p50, color="#4C78A8", linewidth=2.3, label="Gaussian median")
    ax.plot(rhop, s_p50, color="#F58518", linewidth=2.3, label="Spline median")
    ax.set_xlabel("rhop")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@task(name="save_te_ne_comparison_plots")
def save_te_ne_comparison_plots_task(
    gaussian_samples: dict[str, Any],
    spline_samples: dict[str, Any],
    output_dir: str,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rhop = np.asarray(gaussian_samples["rhop"], dtype=np.float64)
    if rhop.shape != np.asarray(spline_samples["rhop"]).shape or not np.allclose(
        rhop, np.asarray(spline_samples["rhop"], dtype=np.float64)
    ):
        raise ValueError("Gaussian and spline runs used different rhop grids.")

    te_overlay_path = out / "te_profiles_overlay.png"
    te_band_path = out / "te_profiles_median_p10_p90.png"
    ne_overlay_path = out / "ne_profiles_overlay.png"
    ne_band_path = out / "ne_profiles_median_p10_p90.png"
    eps_overlay_path = out / "eps_profiles_overlay.png"
    eps_band_path = out / "eps_profiles_median_p10_p90.png"

    g_te = _as_sample_matrix(gaussian_samples["electron_temperature"], rhop, "gaussian Te")
    s_te = _as_sample_matrix(spline_samples["electron_temperature"], rhop, "spline Te")
    g_ne = _as_sample_matrix(gaussian_samples["electron_density"], rhop, "gaussian ne")
    s_ne = _as_sample_matrix(spline_samples["electron_density"], rhop, "spline ne")
    g_eps = _as_sample_matrix(gaussian_samples["model_emissivity"], rhop, "gaussian model_emissivity")
    s_eps = _as_sample_matrix(spline_samples["model_emissivity"], rhop, "spline model_emissivity")

    _plot_overlay(
        rhop=rhop,
        gaussian_samples=g_te,
        spline_samples=s_te,
        ylabel="Electron Temperature",
        title="Electron Temperature Profiles: Gaussian vs Spline",
        output_path=te_overlay_path,
    )
    _plot_percentile_band(
        rhop=rhop,
        gaussian_samples=g_te,
        spline_samples=s_te,
        ylabel="Electron Temperature",
        title="Electron Temperature Median + p10-p90",
        output_path=te_band_path,
    )
    _plot_overlay(
        rhop=rhop,
        gaussian_samples=g_ne,
        spline_samples=s_ne,
        ylabel="Electron Density",
        title="Electron Density Profiles: Gaussian vs Spline",
        output_path=ne_overlay_path,
    )
    _plot_percentile_band(
        rhop=rhop,
        gaussian_samples=g_ne,
        spline_samples=s_ne,
        ylabel="Electron Density",
        title="Electron Density Median + p10-p90",
        output_path=ne_band_path,
    )
    _plot_overlay(
        rhop=rhop,
        gaussian_samples=g_eps,
        spline_samples=s_eps,
        ylabel="Model Emissivity",
        title="Model Emissivity Profiles: Gaussian vs Spline",
        output_path=eps_overlay_path,
    )
    _plot_percentile_band(
        rhop=rhop,
        gaussian_samples=g_eps,
        spline_samples=s_eps,
        ylabel="Model Emissivity",
        title="Model Emissivity Median + p10-p90",
        output_path=eps_band_path,
    )

    return {
        "te_overlay": str(te_overlay_path),
        "te_median_p10_p90": str(te_band_path),
        "ne_overlay": str(ne_overlay_path),
        "ne_median_p10_p90": str(ne_band_path),
        "eps_overlay": str(eps_overlay_path),
        "eps_median_p10_p90": str(eps_band_path),
    }


@flow(name="compare_te_ne_profile_sampling")
def compare_te_ne_profile_sampling(
    n_samples: int = 50,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    gaussian_config_name: str = "ion_temperature_phantom_run_all_params",
    spline_config_name: str = "baseline_spline_tene",
    seed: int = 0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Sample Te/ne profiles from two configs and save side-by-side visualisations."""
    gaussian_samples = sample_profiles_for_config_task(
        config_name=gaussian_config_name,
        n_samples=n_samples,
        seed=seed,
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        use_real_equilibrium=use_real_equilibrium,
        real_equilibrium_pulse=real_equilibrium_pulse,
        real_equilibrium_verbose=real_equilibrium_verbose,
    )
    spline_samples = sample_profiles_for_config_task(
        config_name=spline_config_name,
        n_samples=n_samples,
        seed=seed + 1,
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        use_real_equilibrium=use_real_equilibrium,
        real_equilibrium_pulse=real_equilibrium_pulse,
        real_equilibrium_verbose=real_equilibrium_verbose,
    )
    outputs = save_te_ne_comparison_plots_task(
        gaussian_samples=gaussian_samples,
        spline_samples=spline_samples,
        output_dir=output_dir,
    )

    return {
        "n_samples": int(n_samples),
        "gaussian_config_name": gaussian_config_name,
        "spline_config_name": spline_config_name,
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = compare_te_ne_profile_sampling()
    print("Te/ne profile comparison complete")
    print(result["outputs"])
