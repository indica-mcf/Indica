"""Contextual comparison with noisy test brightness on constant-imp expanded data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

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


@flow(name="multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_contextual_comparison")
def multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_contextual_comparison(
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
    noise_count_level: float = 200.0,
    noise_scale_percentile: float = 99.0,
    noise_seed: int | None = 0,
    vae_model_path: str | None = None,
) -> dict[str, Any]:
    """
    Train/reuse VAE and compare VAE vs naive on freshly generated noisy-b samples.

    Each generated comparison sample:
      1) picks one of the same equilibrium-time points used by expanded-equilibria generation
      2) samples a plasma with fixed impurities
      3) forward-models brightness with that explicit equilibrium context
      4) applies Poisson noise to test-set brightness
      5) compares VAE reconstruction (without explicit eq input) against naive inversion
    """
    b_path = str(Path(output_dir) / b_filename)
    eps_path = str(Path(output_dir) / eps_filename)

    training_result = None
    model_path = vae_model_path
    if train_or_reuse_vae:
        training_result = bolometry_inversion_multipulse_synthetic(
            machine=machine,
            instrument=instrument,
            output_dir=output_dir,
            b_filename=b_filename,
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
        b_path=b_path,
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

    return {
        "training": training_result,
        "model_path": str(model_path),
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_contextual_comparison()
    print(result)
