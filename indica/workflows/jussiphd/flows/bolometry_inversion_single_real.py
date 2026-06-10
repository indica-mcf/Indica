"""Prefect flow for single-pulse real ST40 bolometry inversion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.components.data.real_dataset_generation import (
    generate_and_save_real_dataset,
    load_real_transform_from_pulse,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.components.evaluation.metrics import (
    compute_vae_diversity_and_forward_metrics,
)
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import (
    create_dataset_and_dataloaders,
)
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    generate_generated_dataset_visualisations,
    generate_vae_training_progress_visualisation,
)

DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[1] / "components" / "data" / "flow_data" / "single_real"
)
DEFAULT_VAE_DIR = str(Path(__file__).resolve().parents[1] / "components" / "ml" / "flow_data" / "single_real")
DEFAULT_VIS_DIR = str(
    Path(__file__).resolve().parents[1] / "components" / "visualisations" / "outputs" / "single_real"
)


@task(name="generate_single_real_dataset")
def build_single_real_dataset_task(
    machine: str,
    instrument: str,
    emissivity_instrument: str,
    transform: Any,
    equilibrium: Any,
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    revision: int,
    node: str | None,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    use_all_timepoints: bool,
    verbose: bool,
    generate_new_data: bool,
) -> dict[str, Any]:
    return generate_and_save_real_dataset(
        machine=machine,
        instrument=instrument,
        emissivity_instrument=emissivity_instrument,
        transform=transform,
        equilibrium=equilibrium,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        use_all_timepoints=use_all_timepoints,
        node=node,
        revision=revision,
        generate_new_data=generate_new_data,
        verbose=verbose,
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


@task(name="load_real_transform")
def load_real_transform_task(
    instrument: str,
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    revision: int,
    equilibrium: Any,
    verbose: bool,
) -> Any:
    return load_real_transform_from_pulse(
        instrument=instrument,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        equilibrium=equilibrium,
        revision=revision,
        verbose=verbose,
    )


@task(name="create_real_training_dataset")
def create_real_training_dataset_task(
    b_path: str,
    eps_path: str,
    train_fraction: float,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> dict[str, Any]:
    bundle = create_dataset_and_dataloaders(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=None,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return bundle["summary"]


@task(name="train_real_vae")
def train_real_vae_task(
    b_path: str,
    eps_path: str,
    latent_dim: int,
    hidden_scaling: int,
    n_epochs: int,
    lr: float,
    train_fraction: float,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
    output_dir: str,
    model_filename: str,
) -> dict[str, Any]:
    return train_vae_from_csv(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=None,
        latent_dim=latent_dim,
        hidden_scaling=hidden_scaling,
        n_epochs=n_epochs,
        lr=lr,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        output_dir=output_dir,
        model_filename=model_filename,
    )


@task(name="compute_real_vae_metrics")
def compute_real_vae_metrics_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    idx: int,
    k_samples: int,
    seed: int | None,
) -> dict[str, Any]:
    return compute_vae_diversity_and_forward_metrics(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        meta_path=None,
        idx=idx,
        k_samples=k_samples,
        seed=seed,
    )


@task(name="generate_real_visualisations")
def generate_real_visualisations_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    transform: Any,
    output_dir: str,
    n_examples: int,
    k_samples: int,
    n_uncertainty_samples: int,
) -> dict[str, Any]:
    return generate_generated_dataset_visualisations(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        transform=transform,
        output_dir=output_dir,
        n_examples=n_examples,
        k_samples=k_samples,
        n_uncertainty_samples=n_uncertainty_samples,
    )


@task(name="generate_real_vae_training_progress_visualisation")
def generate_real_vae_training_progress_visualisation_task(
    model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    return generate_vae_training_progress_visualisation(
        model_path=model_path,
        output_dir=output_dir,
    )


@flow(name="bolometry_inversion_single_real")
def bolometry_inversion_single_real(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    emissivity_instrument: str = "blom_rz1",
    pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    revision: int = 0,
    node: str | None = None,
    read_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    b_filename: str = "b_slices_single_real.csv",
    eps_filename: str = "eps_slices_single_real.csv",
    generate_new_data: bool = True,
    use_all_timepoints: bool = True,
    create_training_dataset: bool = True,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    run_vae_training: bool = True,
    vae_output_dir: str = DEFAULT_VAE_DIR,
    vae_model_filename: str = "vae_single_real.pt",
    vae_latent_dim: int = 4,
    vae_hidden_scaling: int = 8,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    run_vae_metrics: bool = True,
    vae_metrics_model_path: str | None = None,
    metrics_idx: int = 10,
    metrics_k_samples: int = 100,
    run_visualisations: bool = True,
    visualisations_output_dir: str = DEFAULT_VIS_DIR,
    visualisations_n_examples: int = 6,
    visualisations_k_samples: int = 20,
    visualisations_n_uncertainty_samples: int = 200,
) -> dict[str, Any]:
    """Run single-real-data workflow: read -> dataset -> VAE -> metrics -> visus."""
    equilibrium = load_real_equilibrium_task(
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        verbose=read_verbose,
    )
    transform = load_real_transform_task(
        instrument=instrument,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        revision=revision,
        equilibrium=equilibrium,
        verbose=read_verbose,
    )

    real_dataset = build_single_real_dataset_task(
        machine=machine,
        instrument=instrument,
        emissivity_instrument=emissivity_instrument,
        transform=transform,
        equilibrium=equilibrium,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        revision=revision,
        node=node,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        use_all_timepoints=use_all_timepoints,
        verbose=read_verbose,
        generate_new_data=generate_new_data,
    )

    dataset_summary = None
    vae_training = None
    vae_metrics = None
    visualisations = None

    if create_training_dataset:
        dataset_summary = create_real_training_dataset_task(
            b_path=real_dataset["b_path"],
            eps_path=real_dataset["eps_path"],
            train_fraction=train_fraction,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=None,
        )

    if run_vae_training:
        vae_training = train_real_vae_task(
            b_path=real_dataset["b_path"],
            eps_path=real_dataset["eps_path"],
            latent_dim=vae_latent_dim,
            hidden_scaling=vae_hidden_scaling,
            n_epochs=vae_n_epochs,
            lr=vae_lr,
            train_fraction=train_fraction,
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
        vae_metrics = compute_real_vae_metrics_task(
            model_path=model_path,
            b_path=real_dataset["b_path"],
            eps_path=real_dataset["eps_path"],
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

        real_visualisations = generate_real_visualisations_task(
            model_path=model_path,
            b_path=real_dataset["b_path"],
            eps_path=real_dataset["eps_path"],
            transform=transform,
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
            k_samples=visualisations_k_samples,
            n_uncertainty_samples=visualisations_n_uncertainty_samples,
        )
        training_progress_visualisation = generate_real_vae_training_progress_visualisation_task(
            model_path=model_path,
            output_dir=visualisations_output_dir,
        )
        visualisations = {
            "real_dataset_visualisations": real_visualisations,
            "training_progress_visualisation": training_progress_visualisation,
        }

    return {
        "real_dataset": real_dataset,
        "dataset_summary": dataset_summary,
        "vae_training": vae_training,
        "vae_metrics": vae_metrics,
        "visualisations": visualisations,
    }


if __name__ == "__main__":
    result = bolometry_inversion_single_real()
