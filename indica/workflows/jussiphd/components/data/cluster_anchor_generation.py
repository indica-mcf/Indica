"""Synthetic dataset generation from anchor-cluster Gaussian parameters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data

from indica.workflows.jussiphd.components.data.data_generation import (
    DEFAULT_BDA_OVERRIDES,
    PlasmaGenerator,
    _normalise_transform_beamlets,
)


def _load_gaussian_params(npz_path: str) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "clusters": np.asarray(data["clusters"], dtype=int).reshape(-1),
        "counts": np.asarray(data["counts"], dtype=int).reshape(-1),
        "means": np.asarray(data["means"], dtype=np.float64),
        "covariances": np.asarray(data["covariances"], dtype=np.float64),
    }


def _safe_mvn_draw(rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(rng.multivariate_normal(mean, cov, check_valid="ignore"), dtype=np.float64)
    except Exception:
        evals, evecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
        evals = np.clip(evals, 1e-12, None)
        cov_psd = (evecs * evals) @ evecs.T
        return np.asarray(rng.multivariate_normal(mean, cov_psd, check_valid="ignore"), dtype=np.float64)


def _select_time_indices(
    t_size: int,
    use_all_timepoints: bool,
    single_timepoint_mode: str,
    rng: np.random.Generator,
) -> list[int]:
    if use_all_timepoints:
        return list(range(int(t_size)))
    if single_timepoint_mode == "middle":
        return [int(t_size // 2)]
    if single_timepoint_mode == "random":
        return [int(rng.integers(0, max(1, int(t_size))))]
    raise ValueError("single_timepoint_mode must be 'middle' or 'random'.")


def generate_and_save_dataset_from_anchor_cluster_gaussians(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    ne_gaussian_params_path: str,
    te_gaussian_params_path: str,
    ne_xknots: Sequence[float],
    te_xknots: Sequence[float],
    n_generations: int,
    use_all_timepoints: bool,
    single_timepoint_mode: str,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    config_name: str = "ion_temperature_phantom_run_all_params",
    config_overrides: Sequence[str] | None = None,
    seed: int = 0,
    sample_weight_by_cluster_counts: bool = True,
    enforce_nonnegative_profiles: bool = True,
    enforce_strictly_positive_profiles: bool = False,
    positive_profile_floor: float = 1e-12,
    impurity_concentrations: dict[str, float] | None = None,
    impurity_flat_zeff: bool = True,
) -> dict[str, Any]:
    """Generate (brightness, emissivity) pairs by sampling TE/NE anchors per cluster family."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_path = output_path / eps_filename
    meta_path = output_path / meta_filename

    if not generate_new_data:
        if not b_path.exists() or not eps_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but dataset files do not exist: "
                f"{b_path}, {eps_path}, {meta_path}"
            )
        b_arr = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        eps_arr = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
        if b_arr.ndim == 1:
            b_arr = b_arr[None, :]
        if eps_arr.ndim == 1:
            eps_arr = eps_arr[None, :]
        return {
            "b_path": str(b_path),
            "eps_path": str(eps_path),
            "meta_path": str(meta_path),
            "num_pairs": int(len(b_arr)),
            "b_shape": tuple(b_arr.shape),
            "eps_shape": tuple(eps_arr.shape),
            "generated_new_data": False,
        }

    ne_params = _load_gaussian_params(ne_gaussian_params_path)
    te_params = _load_gaussian_params(te_gaussian_params_path)
    k = int(min(ne_params["means"].shape[0], te_params["means"].shape[0]))
    if k <= 0:
        raise ValueError("No clusters available in Gaussian parameter files.")

    ne_x = np.asarray(ne_xknots, dtype=np.float64).reshape(-1)
    te_x = np.asarray(te_xknots, dtype=np.float64).reshape(-1)
    if ne_params["means"].shape[1] != ne_x.size:
        raise ValueError(
            f"NE anchor dimension mismatch: means dim {ne_params['means'].shape[1]} vs xknots {ne_x.size}"
        )
    if te_params["means"].shape[1] != te_x.size:
        raise ValueError(
            f"TE anchor dimension mismatch: means dim {te_params['means'].shape[1]} vs xknots {te_x.size}"
        )

    ne_counts = ne_params["counts"][:k].astype(np.float64)
    te_counts = te_params["counts"][:k].astype(np.float64)
    if sample_weight_by_cluster_counts and np.all((ne_counts + te_counts) > 0):
        w = 0.5 * (ne_counts + te_counts)
        w = w / np.sum(w)
    else:
        w = np.full(k, 1.0 / float(k), dtype=np.float64)

    transform.set_equilibrium(equilibrium)
    transform.spot_shape = "square"
    transform.focal_length = -1000.0
    _normalise_transform_beamlets(transform)
    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    model = PinholeCamera(instrument, power_loss=power_loss)
    model.set_transform(transform)

    overrides = (
        list(config_overrides)
        if config_overrides is not None
        else list(DEFAULT_BDA_OVERRIDES)
    )
    generator = PlasmaGenerator(
        model=model,
        transform=transform,
        config_name=config_name,
        overrides=overrides,
    )

    rng = np.random.default_rng(int(seed))
    b_slices: list[np.ndarray] = []
    eps_slices: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []

    for gen_idx in range(int(n_generations)):
        fam_idx = int(rng.choice(k, p=w))
        ne_anchor = _safe_mvn_draw(rng, ne_params["means"][fam_idx], ne_params["covariances"][fam_idx])
        te_anchor = _safe_mvn_draw(rng, te_params["means"][fam_idx], te_params["covariances"][fam_idx])
        plasma = generator.generate()
        if impurity_concentrations is not None:
            for element, concentration in impurity_concentrations.items():
                plasma.set_impurity_concentration(
                    element=str(element),
                    concentration=float(concentration),
                    flat_zeff=bool(impurity_flat_zeff),
                )
        rhop = np.asarray(plasma.rhop.values, dtype=np.float64).reshape(-1)
        ne_profile = np.interp(rhop, ne_x, ne_anchor).astype(np.float64)
        te_profile = np.interp(rhop, te_x, te_anchor).astype(np.float64)
        if enforce_nonnegative_profiles:
            ne_profile = np.maximum(ne_profile, 0.0)
            te_profile = np.maximum(te_profile, 0.0)
        if enforce_strictly_positive_profiles:
            floor = float(positive_profile_floor)
            if floor <= 0.0:
                raise ValueError(
                    "positive_profile_floor must be > 0 when "
                    "enforce_strictly_positive_profiles=True."
                )
            ne_profile = np.maximum(ne_profile, floor)
            te_profile = np.maximum(te_profile, floor)
        plasma.electron_density.loc[dict(t=plasma.t)] = np.repeat(ne_profile[None, :], int(plasma.t.size), axis=0)
        plasma.electron_temperature.loc[dict(t=plasma.t)] = np.repeat(
            te_profile[None, :], int(plasma.t.size), axis=0
        )
        measurements, emissivity = generator.run_model(target_plasma=plasma)
        t_indices = _select_time_indices(
            t_size=int(measurements.sizes["t"]),
            use_all_timepoints=bool(use_all_timepoints),
            single_timepoint_mode=single_timepoint_mode,
            rng=rng,
        )
        for t_idx in t_indices:
            b_slices.append(measurements.isel(t=t_idx).values.astype(np.float32).reshape(-1))
            eps_slices.append(emissivity.isel(t=t_idx).values.astype(np.float32).reshape(-1))
            t_val = float(np.asarray(measurements.t.values, dtype=np.float64).reshape(-1)[t_idx])
            meta_rows.append(
                {
                    "sample_index": int(len(meta_rows)),
                    "generation_index": int(gen_idx),
                    "cluster_family_index": int(fam_idx),
                    "ne_cluster": int(ne_params["clusters"][fam_idx]),
                    "te_cluster": int(te_params["clusters"][fam_idx]),
                    "t_idx": int(t_idx),
                    "t_s": t_val,
                }
            )

    b_arr = np.asarray(b_slices, dtype=np.float32)
    eps_arr = np.asarray(eps_slices, dtype=np.float32)
    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_path, eps_arr, delimiter=",")
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "generation_index",
                "cluster_family_index",
                "ne_cluster",
                "te_cluster",
                "t_idx",
                "t_s",
            ],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    result = {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "meta_path": str(meta_path),
        "num_pairs": int(len(b_arr)),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "n_generations": int(n_generations),
        "num_cluster_families": int(k),
        "generated_new_data": True,
        "single_timepoint_mode": single_timepoint_mode,
        "enforce_nonnegative_profiles": bool(enforce_nonnegative_profiles),
        "enforce_strictly_positive_profiles": bool(enforce_strictly_positive_profiles),
        "positive_profile_floor": float(positive_profile_floor),
        "source_ne_gaussian_params": str(ne_gaussian_params_path),
        "source_te_gaussian_params": str(te_gaussian_params_path),
    }
    if impurity_concentrations is not None:
        result["impurity_concentrations"] = {
            str(k): float(v) for k, v in impurity_concentrations.items()
        }
        result["impurity_flat_zeff"] = bool(impurity_flat_zeff)
    return result
