"""Prefect flow for multi-pulse bolometry dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Sequence

from prefect import flow, task

from indica.workflows.jussiphd.components.filtering.dataset_filters import (
    apply_zero_and_tomo_filters,
)
from indica.workflows.jussiphd.components.evaluation.metrics import (
    compute_vae_diversity_and_forward_metrics,
)
from indica.workflows.jussiphd.components.visualisations.vae_tomo_visualisations import (
    generate_vae_tomo_visualisations,
)
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import (
    create_dataset_and_dataloaders,
)
from indica.workflows.jussiphd.components.data.data_generation import (
    generate_and_save_multipulse_real_dataset,
)

DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "components" / "data")


@task(name="generate_multipulse_real_dataset")
def generate_multipulse_real_dataset_task(
    pulses: list[int],
    machine: str,
    instrument: str,
    tstart: float,
    tend: float,
    dt: float,
    use_all_timepoints: bool,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
) -> dict[str, Any]:
    return generate_and_save_multipulse_real_dataset(
        pulses=pulses,
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        use_all_timepoints=use_all_timepoints,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
    )


@task(name="filter_multipulse_dataset")
def filter_multipulse_dataset_task(
    b_path: str,
    eps_path: str,
    meta_path: str | None,
    zero_tol: float,
    zero_slack: int,
    min_valid_channels_required: int,
    overwrite: bool,
    output_dir: str | None,
) -> dict[str, Any]:
    return apply_zero_and_tomo_filters(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        zero_tol=zero_tol,
        zero_slack=zero_slack,
        min_valid_channels_required=min_valid_channels_required,
        overwrite=overwrite,
        output_dir=output_dir,
    )


@task(name="create_training_dataset")
def create_training_dataset_task(
    b_path: str,
    eps_path: str,
    meta_path: str | None,
    zero_tol: float,
    train_fraction: float,
    batch_size: int,
    shuffle: bool,
    seed: int | None,
) -> dict[str, Any]:
    bundle = create_dataset_and_dataloaders(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        zero_tol=zero_tol,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return bundle["summary"]


@task(name="train_vae")
def train_vae_task(
    b_path: str,
    eps_path: str,
    meta_path: str | None,
    latent_dim: int,
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
        meta_path=meta_path,
        latent_dim=latent_dim,
        n_epochs=n_epochs,
        lr=lr,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        output_dir=output_dir,
        model_filename=model_filename,
    )


@task(name="compute_vae_metrics")
def compute_vae_metrics_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    meta_path: str | None,
    idx: int,
    k_samples: int,
    seed: int | None,
) -> dict[str, Any]:
    return compute_vae_diversity_and_forward_metrics(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        idx=idx,
        k_samples=k_samples,
        seed=seed,
    )


@task(name="generate_vae_tomo_visualisations")
def generate_vae_tomo_visualisations_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    meta_path: str,
    machine: str,
    instrument: str,
    tstart: float,
    tend: float,
    dt: float,
    output_dir: str,
    max_pulses_to_plot: int,
    max_test_samples: int,
    n_times_per_pulse: int,
) -> dict[str, Any]:
    return generate_vae_tomo_visualisations(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        output_dir=output_dir,
        max_pulses_to_plot=max_pulses_to_plot,
        max_test_samples=max_test_samples,
        n_times_per_pulse=n_times_per_pulse,
    )


@flow(name="bolometry_inversion")
def bolometry_inversion(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pulses: Sequence[int] | None = None,
    multipulse_use_all_timepoints: bool = True,
    multipulse_b_filename: str = "vae_firstpass/b_slices_multipulse.csv",
    multipulse_eps_filename: str = "vae_firstpass/eps_slices_multipulse.csv",
    multipulse_meta_filename: str = "vae_firstpass/sample_meta_multipulse.csv",
    multipulse_generate_new_data: bool = False,
    apply_dataset_filters: bool = True,
    zero_tol: float = 0.0,
    zero_slack: int = 30,
    min_valid_channels_required: int = 2,
    filters_overwrite: bool = True,
    filters_output_dir: str | None = None,
    create_training_dataset: bool = True,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    split_seed: int | None = None,
    run_vae_training: bool = True,
    vae_output_dir: str = str(Path(__file__).resolve().parents[1] / "components" / "ml"),
    vae_model_filename: str = "vae_model.pt",
    vae_latent_dim: int = 4,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    run_vae_metrics: bool = True,
    vae_metrics_model_path: str | None = None,
    metrics_idx: int = 10,
    metrics_k_samples: int = 100,
    run_visualisations: bool = True,
    visualisations_output_dir: str = str(
        Path(__file__).resolve().parents[1] / "components" / "visualisations" / "outputs"
    ),
    visualisations_max_pulses_to_plot: int = 3,
    visualisations_max_test_samples: int = 50,
    visualisations_n_times_per_pulse: int = 4,
) -> dict[str, Any]:
    """Use pre-saved multipulse data by default, or regenerate when pulses are given."""
    pulse_list = list(pulses) if pulses is not None else []
    if multipulse_generate_new_data and not pulse_list:
        raise ValueError(
            "multipulse_generate_new_data=True requires explicit `pulses`."
        )

    multipulse_dataset_info = generate_multipulse_real_dataset_task(
        pulses=pulse_list,
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        use_all_timepoints=multipulse_use_all_timepoints,
        output_dir=output_dir,
        b_filename=multipulse_b_filename,
        eps_filename=multipulse_eps_filename,
        meta_filename=multipulse_meta_filename,
        generate_new_data=multipulse_generate_new_data,
    )

    filtered_dataset_info = None
    dataset_summary = None
    vae_training = None
    vae_metrics = None
    visualisations = None
    if apply_dataset_filters:
        filtered_dataset_info = filter_multipulse_dataset_task(
            b_path=multipulse_dataset_info["b_path"],
            eps_path=multipulse_dataset_info["eps_path"],
            meta_path=multipulse_dataset_info.get("meta_path"),
            zero_tol=zero_tol,
            zero_slack=zero_slack,
            min_valid_channels_required=min_valid_channels_required,
            overwrite=filters_overwrite,
            output_dir=filters_output_dir,
        )

    training_source = (
        filtered_dataset_info if filtered_dataset_info is not None else multipulse_dataset_info
    )
    evaluation_source = training_source


    if create_training_dataset:
        dataset_summary = create_training_dataset_task(
            b_path=training_source["b_path"],
            eps_path=training_source["eps_path"],
            meta_path=training_source.get("meta_path"),
            zero_tol=zero_tol,
            train_fraction=train_fraction,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=split_seed,
        )

    if run_vae_training:
        vae_training = train_vae_task(
            b_path=training_source["b_path"],
            eps_path=training_source["eps_path"],
            meta_path=training_source.get("meta_path"),
            latent_dim=vae_latent_dim,
            n_epochs=vae_n_epochs,
            lr=vae_lr,
            train_fraction=train_fraction,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=split_seed,
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
        vae_metrics = compute_vae_metrics_task(
            model_path=model_path,
            b_path=evaluation_source["b_path"],
            eps_path=evaluation_source["eps_path"],
            meta_path=evaluation_source.get("meta_path"),
            idx=metrics_idx,
            k_samples=metrics_k_samples,
            seed=split_seed,
        )

    if run_visualisations:
        meta_path = evaluation_source.get("meta_path")
        if meta_path is None:
            raise ValueError("run_visualisations=True requires meta_path in evaluation source.")
        model_path = vae_metrics_model_path
        if model_path is None:
            if vae_training is None:
                raise ValueError(
                    "run_visualisations=True requires either run_vae_training=True "
                    "or explicit vae_metrics_model_path."
                )
            model_path = vae_training["model_path"]
        visualisations = generate_vae_tomo_visualisations_task(
            model_path=model_path,
            b_path=evaluation_source["b_path"],
            eps_path=evaluation_source["eps_path"],
            meta_path=meta_path,
            machine=machine,
            instrument=instrument,
            tstart=tstart,
            tend=tend,
            dt=dt,
            output_dir=visualisations_output_dir,
            max_pulses_to_plot=visualisations_max_pulses_to_plot,
            max_test_samples=visualisations_max_test_samples,
            n_times_per_pulse=visualisations_n_times_per_pulse,
        )

    return {
        "multipulse_dataset": multipulse_dataset_info,
        "filtered_dataset": filtered_dataset_info,
        "dataset_summary": dataset_summary,
        "vae_training": vae_training,
        "vae_metrics": vae_metrics,
        "visualisations": visualisations,
    }


if __name__ == "__main__":
    result = bolometry_inversion()
    print(result.get("dataset_summary", {}))
