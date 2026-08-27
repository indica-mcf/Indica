"""Contextual comparison with full-dataset noisy brightness on constant-imp expanded data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from prefect import flow, task

from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)
from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset
from indica.workflows.jussiphd.components.visualisations.vae_contextual_comparison import (
    generate_contextual_vae_vs_naive_visualisations,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR,
)
from indica.workflows.jussiphd.experiments.multipulse_synthetic.flow import (
    bolometry_inversion_multipulse_synthetic,
)


DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR
DEFAULT_VAE_DIR = str(
    Path(__file__).resolve().parents[2]
    / "components"
    / "ml"
    / "flow_data"
    / "multipulse_synthetic_expanded_equilibria_constant_imp_compare"
)
DEFAULT_VIS_DIR = str(Path(__file__).resolve().parent / "outputs")


def _load_vae(model_path: str) -> CVAENetwork:
    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_scaling=int(ckpt.get("hidden_scaling", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@task(name="noise_full_brightness_dataset")
def noise_full_brightness_dataset_task(
    b_path: str,
    output_dir: str,
    noisy_b_filename: str,
    count_level: float,
    scale_percentile: float,
    seed: int | None,
) -> dict[str, Any]:
    b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
    if b.ndim == 1:
        b = b[None, :]
    scale = float(np.percentile(np.clip(b, a_min=0.0, a_max=None), float(scale_percentile)))
    if scale <= 0:
        scale = 1.0
    rng = np.random.default_rng(seed)
    b_noisy = add_poisson_noise_with_counts(
        values=b,
        count_level=float(count_level),
        scale_value=scale,
        rng=rng,
    ).astype(np.float32)

    out_path = Path(output_dir) / noisy_b_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, b_noisy, delimiter=",")
    return {
        "b_noisy_path": str(out_path),
        "count_level": float(count_level),
        "scale_percentile": float(scale_percentile),
        "scale_value": float(scale),
        "seed": None if seed is None else int(seed),
    }


@task(name="compare_vae_vs_naive_on_generated_equilibrium_contexts_noisy_b")
def compare_vae_vs_naive_on_generated_equilibrium_contexts_noisy_b_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    config_name: str,
    config_overrides: list[str] | None,
    n_timepoints_per_equilibrium: int,
    n_generated_samples: int,
    c_concentration: float,
    ar_concentration: float,
    k_samples: int,
    n_examples: int,
    seed: int | None,
    noise_count_level: float,
    noise_scale_percentile: float,
    noise_seed: int | None,
) -> dict[str, Any]:
    return generate_contextual_vae_vs_naive_visualisations(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        config_name=config_name,
        config_overrides=config_overrides,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        n_generated_samples=n_generated_samples,
        c_concentration=c_concentration,
        ar_concentration=ar_concentration,
        k_samples=k_samples,
        n_examples=n_examples,
        seed=seed,
        noise_b_test=True,
        noise_count_level=noise_count_level,
        noise_scale_percentile=noise_scale_percentile,
        noise_seed=noise_seed,
    )


@task(name="evaluate_vae_uncertainty_calibration")
def evaluate_vae_uncertainty_calibration_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    output_dir: str,
    summary_filename: str,
    per_sample_filename: str,
    k_samples: int,
    max_eval_samples: int,
    seed: int | None,
    std_floor: float = 1e-8,
) -> dict[str, Any]:
    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=None)
    model = _load_vae(model_path=model_path)

    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("Dataset is empty; cannot evaluate calibration.")
    n_eval = int(min(max_eval_samples, n_total)) if max_eval_samples > 0 else int(n_total)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=n_eval, replace=False))

    cover_1sigma = []
    cover_2sigma = []
    cover_p10_p90 = []
    mean_abs_z = []
    rmse_mean = []
    mean_pred_std = []
    rows = []
    n_valid_points_total = 0

    with torch.no_grad():
        for idx in indices:
            e_norm, b_norm = dataset[int(idx)]
            e_true = (e_norm * dataset.sigma_eps + dataset.mu_eps).astype(np.float32)
            b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)

            z = torch.randn(int(k_samples), model.latent_dim)
            b_rep = b_t_norm.expand(int(k_samples), -1)
            e_samps = model.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()

            pred_mean = e_samps_un.mean(axis=0)
            pred_std = e_samps_un.std(axis=0)
            lo = np.percentile(e_samps_un, 10.0, axis=0)
            hi = np.percentile(e_samps_un, 90.0, axis=0)

            valid = np.isfinite(e_true) & np.isfinite(pred_mean) & np.isfinite(pred_std)
            if not np.any(valid):
                continue

            safe_std = np.maximum(pred_std[valid], float(std_floor))
            abs_z = np.abs((e_true[valid] - pred_mean[valid]) / safe_std)
            c1 = float(np.mean(abs_z <= 1.0))
            c2 = float(np.mean(abs_z <= 2.0))
            cp = float(np.mean((e_true[valid] >= lo[valid]) & (e_true[valid] <= hi[valid])))
            maz = float(np.mean(abs_z))
            rmean = float(np.sqrt(np.mean((pred_mean[valid] - e_true[valid]) ** 2)))
            mstd = float(np.mean(pred_std[valid]))

            cover_1sigma.append(c1)
            cover_2sigma.append(c2)
            cover_p10_p90.append(cp)
            mean_abs_z.append(maz)
            rmse_mean.append(rmean)
            mean_pred_std.append(mstd)
            n_valid_points_total += int(np.count_nonzero(valid))

            rows.append([int(idx), c1, c2, cp, maz, rmean, mstd, int(np.count_nonzero(valid))])

    if not rows:
        raise ValueError("No valid samples found during calibration evaluation.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / per_sample_filename
    summary_path = out_dir / summary_filename

    np.savetxt(
        per_sample_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header=(
            "sample_index,coverage_1sigma,coverage_2sigma,coverage_p10_p90,"
            "mean_abs_z,rmse_mean,mean_pred_std,num_valid_points"
        ),
        comments="",
    )

    summary = {
        "n_total": int(n_total),
        "n_evaluated": int(len(rows)),
        "k_samples": int(k_samples),
        "coverage_1sigma_mean": float(np.mean(cover_1sigma)),
        "coverage_2sigma_mean": float(np.mean(cover_2sigma)),
        "coverage_p10_p90_mean": float(np.mean(cover_p10_p90)),
        "mean_abs_z_mean": float(np.mean(mean_abs_z)),
        "rmse_of_mean_prediction_mean": float(np.mean(rmse_mean)),
        "mean_pred_std_mean": float(np.mean(mean_pred_std)),
        "num_valid_points_total": int(n_valid_points_total),
        "per_sample_csv": str(per_sample_path),
    }
    np.savetxt(
        summary_path,
        np.asarray([[k, v] for k, v in summary.items() if k != "per_sample_csv"], dtype=object),
        fmt="%s",
        delimiter=",",
        header="metric,value",
        comments="",
    )
    summary["summary_csv"] = str(summary_path)
    return summary


@flow(name="multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_contextual_comparison_testbed")
def multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_contextual_comparison_testbed(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    train_or_reuse_vae: bool = True,
    vae_output_dir: str = DEFAULT_VAE_DIR,
    vae_model_filename: str = "vae_multipulse_synthetic_expanded_equilibria_constant_imp.pt",
    vae_latent_dim: int = 4,
    vae_hidden_scaling: int = 8,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    visualisations_output_dir: str = DEFAULT_VIS_DIR,
    n_generated_samples: int = 200,
    n_examples: int = 8,
    k_samples: int = 30,
    n_timepoints_per_equilibrium: int = 5,
    config_name: str = "ion_temperature_phantom_run_all_params",
    config_overrides: list[str] | None = None,
    c_concentration: float = 0.05,
    ar_concentration: float = 0.01,
    seed: int | None = 7,
    noise_count_level: float = 370.0,
    noise_scale_percentile: float = 99.0,
    noise_seed: int | None = 0,
    noisy_b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria_constant_imp_noisy_poisson370.csv",
    calibration_output_dir: str | None = None,
    calibration_summary_filename: str = "vae_uncertainty_calibration_summary.csv",
    calibration_per_sample_filename: str = "vae_uncertainty_calibration_per_sample.csv",
    calibration_k_samples: int = 200,
    calibration_max_eval_samples: int = 400,
    calibration_seed: int | None = 0,
    vae_model_path: str | None = None,
) -> dict[str, Any]:
    """
    Train/reuse VAE and compare VAE vs naive on freshly generated noisy-b samples.

    Each generated comparison sample:
      1) picks one of the same equilibrium-time points used by expanded-equilibria generation
      2) samples a plasma with fixed impurities
      3) forward-models brightness with that explicit equilibrium context
      4) applies Poisson noise to the entire reference brightness dataset
      5) compares VAE reconstruction (without explicit eq input) against naive inversion
    """
    b_path = str(Path(output_dir) / b_filename)
    eps_path = str(Path(output_dir) / eps_filename)
    noisy_b = noise_full_brightness_dataset_task(
        b_path=b_path,
        output_dir=output_dir,
        noisy_b_filename=noisy_b_filename,
        count_level=noise_count_level,
        scale_percentile=noise_scale_percentile,
        seed=noise_seed,
    )
    b_noisy_path = str(noisy_b["b_noisy_path"])

    training_result = None
    model_path = vae_model_path
    if train_or_reuse_vae:
        training_result = bolometry_inversion_multipulse_synthetic(
            machine=machine,
            instrument=instrument,
            output_dir=output_dir,
            b_filename=noisy_b_filename,
            eps_filename=eps_filename,
            generate_new_data=False,
            use_all_timepoints=True,
            create_training_dataset=True,
            train_fraction=train_fraction,
            batch_size=batch_size,
            shuffle=True,
            run_vae_training=True,
            vae_output_dir=vae_output_dir,
            vae_model_filename=vae_model_filename,
            vae_latent_dim=vae_latent_dim,
            vae_hidden_scaling=vae_hidden_scaling,
            vae_n_epochs=vae_n_epochs,
            vae_lr=vae_lr,
            run_vae_metrics=False,
            run_visualisations=False,
        )
        model_path = str(training_result["vae_training"]["model_path"])

    if model_path is None:
        raise ValueError(
            "No VAE model path available. Set train_or_reuse_vae=True or provide vae_model_path."
        )

    comparison = compare_vae_vs_naive_on_generated_equilibrium_contexts_noisy_b_task(
        model_path=model_path,
        b_path=b_noisy_path,
        eps_path=eps_path,
        output_dir=visualisations_output_dir,
        machine=machine,
        instrument=instrument,
        config_name=config_name,
        config_overrides=config_overrides,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        n_generated_samples=n_generated_samples,
        c_concentration=c_concentration,
        ar_concentration=ar_concentration,
        k_samples=k_samples,
        n_examples=n_examples,
        seed=seed,
        noise_count_level=noise_count_level,
        noise_scale_percentile=noise_scale_percentile,
        noise_seed=noise_seed,
    )
    calibration = evaluate_vae_uncertainty_calibration_task(
        model_path=str(model_path),
        b_path=b_noisy_path,
        eps_path=eps_path,
        output_dir=(
            calibration_output_dir
            if calibration_output_dir is not None
            else visualisations_output_dir
        ),
        summary_filename=calibration_summary_filename,
        per_sample_filename=calibration_per_sample_filename,
        k_samples=calibration_k_samples,
        max_eval_samples=calibration_max_eval_samples,
        seed=calibration_seed,
    )

    return {
        "noisy_b_dataset": noisy_b,
        "training": training_result,
        "model_path": str(model_path),
        "comparison": comparison,
        "calibration": calibration,
    }


if __name__ == "__main__":
    result = multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_contextual_comparison_testbed()
    print(result)
