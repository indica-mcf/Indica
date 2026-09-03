"""Prefect flow: synthetic training with noisy-brightness testing set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.data_generation import (
    generate_and_save_dataset,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.components.evaluation.metrics import (
    compute_vae_diversity_and_forward_metrics,
)
from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv
from indica.workflows.jussiphd.components.visualisations.measurement_noise_visualisations import (
    plot_measurement_noise_comparison,
)
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    generate_generated_dataset_visualisations,
    generate_vae_training_progress_visualisation,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_DATA_DIR_STR,
)
from indica.workflows.jussiphd.experiments.inference_timing_synthetic.flow import (
    run_timing_benchmark_task,
    save_timing_results_task,
)

DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_DATA_DIR_STR
DEFAULT_VAE_DIR = str(
    Path(__file__).resolve().parents[2] / "components" / "ml" / "flow_data" / "multipulse_synthetic_noisy_test_b"
)
DEFAULT_VIS_DIR = str(Path(__file__).resolve().parent / "outputs")


@task(name="generate_multipulse_synthetic_dataset")
def generate_multipulse_synthetic_dataset_task(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    n_generations: int,
    use_all_timepoints: bool,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    generate_new_data: bool,
) -> dict[str, Any]:
    return generate_and_save_dataset(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=n_generations,
        use_all_timepoints=use_all_timepoints,
        single_timepoint_mode="middle",
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        generate_new_data=generate_new_data,
    )


@task(name="load_real_equilibrium")
def load_real_equilibrium_task(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    verbose: bool,
) -> Any:
    return load_real_equilibrium_from_pulse(
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        verbose=verbose,
    )


@task(name="split_train_test_dataset")
def split_train_test_dataset_task(
    b_path: str,
    eps_path: str,
    output_dir: str,
    train_fraction: float,
    seed: int,
) -> dict[str, str]:
    b = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
    eps = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
    if b.ndim == 1:
        b = b[None, :]
    if eps.ndim == 1:
        eps = eps[None, :]
    if b.shape[0] != eps.shape[0]:
        raise ValueError(f"Row mismatch: b={b.shape}, eps={eps.shape}")

    n = int(b.shape[0])
    n_train = int(np.clip(round(float(train_fraction) * n), 1, n - 1))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b_train_path = out_dir / "b_train.csv"
    eps_train_path = out_dir / "eps_train.csv"
    b_test_path = out_dir / "b_test_clean.csv"
    eps_test_path = out_dir / "eps_test.csv"
    np.savetxt(b_train_path, b[train_idx], delimiter=",")
    np.savetxt(eps_train_path, eps[train_idx], delimiter=",")
    np.savetxt(b_test_path, b[test_idx], delimiter=",")
    np.savetxt(eps_test_path, eps[test_idx], delimiter=",")

    return {
        "b_train_path": str(b_train_path),
        "eps_train_path": str(eps_train_path),
        "b_test_clean_path": str(b_test_path),
        "eps_test_path": str(eps_test_path),
    }


@task(name="noise_test_brightness")
def noise_test_brightness_task(
    b_test_clean_path: str,
    output_dir: str,
    count_level: float,
    scale_percentile: float,
    seed: int,
) -> dict[str, Any]:
    b = np.loadtxt(b_test_clean_path, delimiter=",", dtype=np.float32)
    if b.ndim == 1:
        b = b[None, :]
    scale = float(np.percentile(np.clip(b, a_min=0.0, a_max=None), float(scale_percentile)))
    if scale <= 0:
        scale = 1.0
    rng = np.random.default_rng(int(seed))
    b_noisy = add_poisson_noise_with_counts(
        values=b,
        count_level=float(count_level),
        scale_value=scale,
        rng=rng,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"b_test_noisy_poisson{int(float(count_level))}.csv"
    np.savetxt(out_path, b_noisy, delimiter=",")
    return {
        "b_test_noisy_path": str(out_path),
        "count_level": float(count_level),
        "scale_percentile": float(scale_percentile),
        "scale_value": float(scale),
    }


@task(name="train_synthetic_vae_on_clean_train")
def train_synthetic_vae_on_clean_train_task(
    b_train_path: str,
    eps_train_path: str,
    latent_dim: int,
    hidden_scaling: int,
    n_epochs: int,
    lr: float,
    kl_scaling: float,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
    output_dir: str,
    model_filename: str,
) -> dict[str, Any]:
    return train_vae_from_csv(
        b_path=b_train_path,
        eps_path=eps_train_path,
        meta_path=None,
        latent_dim=latent_dim,
        hidden_scaling=hidden_scaling,
        n_epochs=n_epochs,
        lr=lr,
        kl_scaling=kl_scaling,
        train_fraction=1.0,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        output_dir=output_dir,
        model_filename=model_filename,
    )


@task(name="compute_noisy_test_metrics")
def compute_noisy_test_metrics_task(
    model_path: str,
    b_test_noisy_path: str,
    eps_test_path: str,
    idx: int,
    k_samples: int,
    seed: int | None,
) -> dict[str, Any]:
    return compute_vae_diversity_and_forward_metrics(
        model_path=model_path,
        b_path=b_test_noisy_path,
        eps_path=eps_test_path,
        meta_path=None,
        idx=idx,
        k_samples=k_samples,
        seed=seed,
    )


@task(name="visualise_noisy_measurements")
def visualise_noisy_measurements_task(
    b_test_clean_path: str,
    b_test_noisy_path: str,
    output_dir: str,
    n_examples: int,
) -> dict[str, Any]:
    return plot_measurement_noise_comparison(
        clean_b_path=b_test_clean_path,
        noisy_b_path=b_test_noisy_path,
        output_dir=output_dir,
        n_examples=n_examples,
    )


@task(name="visualise_vae_on_noisy_test")
def visualise_vae_on_noisy_test_task(
    model_path: str,
    b_test_noisy_path: str,
    eps_test_path: str,
    transform: Any,
    output_dir: str,
    n_examples: int,
    k_samples: int,
    n_uncertainty_samples: int,
) -> dict[str, Any]:
    return generate_generated_dataset_visualisations(
        model_path=model_path,
        b_path=b_test_noisy_path,
        eps_path=eps_test_path,
        transform=transform,
        output_dir=output_dir,
        n_examples=n_examples,
        k_samples=k_samples,
        n_uncertainty_samples=n_uncertainty_samples,
    )


@task(name="visualise_vae_training_progress")
def visualise_vae_training_progress_task(
    model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    return generate_vae_training_progress_visualisation(
        model_path=model_path,
        output_dir=output_dir,
    )


@flow(name="bolometry_inversion_multipulse_synthetic_noisy_test_b")
def bolometry_inversion_multipulse_synthetic_noisy_test_b(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    b_filename: str = "b_slices_multipulse_synthetic.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic.csv",
    n_generations: int = 3000,
    generate_new_data: bool = False,
    use_all_timepoints: bool = True,
    train_fraction: float = 0.8,
    split_seed: int = 0,
    test_b_poisson_count_level: float = 200.0,
    test_b_poisson_scale_percentile: float = 99.0,
    test_b_poisson_seed: int = 0,
    run_vae_training: bool = True,
    vae_output_dir: str = DEFAULT_VAE_DIR,
    vae_model_filename: str = "vae_multipulse_synthetic_noisy_test_b.pt",
    vae_latent_dim: int = 4,
    vae_hidden_scaling: int = 8,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    vae_kl_scaling: float = 0.2,
    batch_size: int = 8,
    shuffle: bool = True,
    run_vae_metrics: bool = True,
    vae_metrics_model_path: str | None = None,
    metrics_idx: int = 10,
    metrics_k_samples: int = 100,
    run_visualisations: bool = True,
    visualisations_output_dir: str = DEFAULT_VIS_DIR,
    visualisations_n_examples: int = 6,
    visualisations_k_samples: int = 20,
    visualisations_n_uncertainty_samples: int = 200,
    run_timing_visualisation: bool = True,
    timing_n_samples: int = 100,
    timing_vae_k_samples: int = 20,
    timing_warmup_samples: int = 20,
    timing_seed: int = 0,
) -> dict[str, Any]:
    """Train on clean synthetic train split and evaluate on noisy-brightness test split."""
    transforms = load_default_objects(machine, "geometry")
    if use_real_equilibrium:
        equilibrium = load_real_equilibrium_task(
            pulse=real_equilibrium_pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=real_equilibrium_verbose,
        )
    else:
        equilibrium = load_default_objects(machine, "equilibrium")
    transform = transforms[instrument]

    generated = generate_multipulse_synthetic_dataset_task(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=n_generations,
        use_all_timepoints=use_all_timepoints,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        generate_new_data=generate_new_data,
    )

    split = split_train_test_dataset_task(
        b_path=generated["b_path"],
        eps_path=generated["eps_path"],
        output_dir=output_dir,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    noisy_test = noise_test_brightness_task(
        b_test_clean_path=split["b_test_clean_path"],
        output_dir=output_dir,
        count_level=test_b_poisson_count_level,
        scale_percentile=test_b_poisson_scale_percentile,
        seed=test_b_poisson_seed,
    )

    vae_training = None
    vae_metrics = None
    visualisations = None
    measurement_visualisation = None

    if run_vae_training:
        vae_training = train_synthetic_vae_on_clean_train_task(
            b_train_path=split["b_train_path"],
            eps_train_path=split["eps_train_path"],
            latent_dim=vae_latent_dim,
            hidden_scaling=vae_hidden_scaling,
            n_epochs=vae_n_epochs,
            lr=vae_lr,
            kl_scaling=vae_kl_scaling,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=None,
            output_dir=vae_output_dir,
            model_filename=vae_model_filename,
        )

    if run_vae_metrics:
        model_path = vae_metrics_model_path
        if model_path is None:
            if vae_training is None:
                raise ValueError(
                    "run_vae_metrics=True requires either run_vae_training=True "
                    "or explicit vae_metrics_model_path."
                )
            model_path = vae_training["model_path"]
        vae_metrics = compute_noisy_test_metrics_task(
            model_path=model_path,
            b_test_noisy_path=noisy_test["b_test_noisy_path"],
            eps_test_path=split["eps_test_path"],
            idx=metrics_idx,
            k_samples=metrics_k_samples,
            seed=None,
        )

    if run_visualisations:
        model_path = vae_metrics_model_path
        if model_path is None:
            if vae_training is None:
                raise ValueError(
                    "run_visualisations=True requires either run_vae_training=True "
                    "or explicit vae_metrics_model_path."
                )
            model_path = vae_training["model_path"]

        measurement_visualisation = visualise_noisy_measurements_task(
            b_test_clean_path=split["b_test_clean_path"],
            b_test_noisy_path=noisy_test["b_test_noisy_path"],
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
        )
        vae_visuals = visualise_vae_on_noisy_test_task(
            model_path=model_path,
            b_test_noisy_path=noisy_test["b_test_noisy_path"],
            eps_test_path=split["eps_test_path"],
            transform=transform,
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
            k_samples=visualisations_k_samples,
            n_uncertainty_samples=visualisations_n_uncertainty_samples,
        )
        training_progress = visualise_vae_training_progress_task(
            model_path=model_path,
            output_dir=visualisations_output_dir,
        )
        timing_visualisation = None
        if run_timing_visualisation:
            timing_benchmark = run_timing_benchmark_task(
                model_path=model_path,
                b_path=noisy_test["b_test_noisy_path"],
                eps_path=split["eps_test_path"],
                transform=transform,
                n_samples=timing_n_samples,
                vae_k_samples=timing_vae_k_samples,
                warmup_samples=timing_warmup_samples,
                seed=timing_seed,
            )
            timing_visualisation = save_timing_results_task(
                benchmark=timing_benchmark,
                output_dir=str(Path(visualisations_output_dir) / "timing"),
                csv_filename="timing_naive_vs_vae.csv",
                plot_filename="timing_naive_vs_vae_log.png",
            )
        visualisations = {
            "measurement_noise_visualisation": measurement_visualisation,
            "vae_on_noisy_test_visualisations": vae_visuals,
            "training_progress_visualisation": training_progress,
            "timing_visualisation": timing_visualisation,
        }

    return {
        "generated_dataset": generated,
        "split": split,
        "noisy_test": noisy_test,
        "vae_training": vae_training,
        "vae_metrics": vae_metrics,
        "visualisations": visualisations,
    }


if __name__ == "__main__":
    result = bolometry_inversion_multipulse_synthetic_noisy_test_b()
