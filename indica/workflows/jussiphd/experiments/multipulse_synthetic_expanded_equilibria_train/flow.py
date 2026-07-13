"""Run the original multipulse_synthetic VAE pipeline on expanded-equilibria data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow

from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR,
)
from indica.workflows.jussiphd.experiments.multipulse_synthetic.flow import (
    bolometry_inversion_multipulse_synthetic,
)


DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR
DEFAULT_VAE_DIR = str(
    Path(__file__).resolve().parents[2]
    / "components"
    / "ml"
    / "flow_data"
    / "multipulse_synthetic_expanded_equilibria"
)
DEFAULT_VIS_DIR = str(Path(__file__).resolve().parent / "outputs")


@flow(name="bolometry_inversion_multipulse_synthetic_expanded_equilibria")
def bolometry_inversion_multipulse_synthetic_expanded_equilibria(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria.csv",
    n_generations: int = 3000,
    generate_new_data: bool = False,
    use_all_timepoints: bool = True,
    create_training_dataset: bool = True,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    run_vae_training: bool = True,
    vae_output_dir: str = DEFAULT_VAE_DIR,
    vae_model_filename: str = "vae_multipulse_synthetic_expanded_equilibria.pt",
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
    """
    Use the original multipulse_synthetic flow/components on expanded-equilibria data.

    Dataset generation is disabled by default (`generate_new_data=False`) so existing
    expanded CSVs are consumed directly.
    """
    return bolometry_inversion_multipulse_synthetic(
        machine=machine,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        use_real_equilibrium=use_real_equilibrium,
        real_equilibrium_pulse=real_equilibrium_pulse,
        real_equilibrium_verbose=real_equilibrium_verbose,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        n_generations=n_generations,
        generate_new_data=generate_new_data,
        use_all_timepoints=use_all_timepoints,
        create_training_dataset=create_training_dataset,
        train_fraction=train_fraction,
        batch_size=batch_size,
        shuffle=shuffle,
        run_vae_training=run_vae_training,
        vae_output_dir=vae_output_dir,
        vae_model_filename=vae_model_filename,
        vae_latent_dim=vae_latent_dim,
        vae_hidden_scaling=vae_hidden_scaling,
        vae_n_epochs=vae_n_epochs,
        vae_lr=vae_lr,
        run_vae_metrics=run_vae_metrics,
        vae_metrics_model_path=vae_metrics_model_path,
        metrics_idx=metrics_idx,
        metrics_k_samples=metrics_k_samples,
        run_visualisations=run_visualisations,
        visualisations_output_dir=visualisations_output_dir,
        visualisations_n_examples=visualisations_n_examples,
        visualisations_k_samples=visualisations_k_samples,
        visualisations_n_uncertainty_samples=visualisations_n_uncertainty_samples,
    )


if __name__ == "__main__":
    result = bolometry_inversion_multipulse_synthetic_expanded_equilibria()
    print(result)

