"""Prefect flow for multipulse-like purely synthetic bolometry inversion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import (
    create_dataset_and_dataloaders,
)
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    generate_generated_dataset_visualisations,
    generate_vae_training_progress_visualisation,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_SPLINED_DATA_DIR_STR,
)

DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_SPLINED_DATA_DIR_STR
DEFAULT_VAE_DIR = str(
    Path(__file__).resolve().parents[2] / "components" / "ml" / "flow_data" / "multipulse_synthetic"
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
    config_name: str = "ion_temperature_phantom_run_all_params",
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
        config_name=config_name
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





@flow(name="bolometry_inversion_multipulse_synthetic_splined")
def bolometry_inversion_multipulse_synthetic_splined(
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
    n_generations: int = 100,
    generate_new_data: bool = True,
    use_all_timepoints: bool = True,
    config_name="baseline_spline_tene"

) -> dict[str, Any]:
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

    synthetic_dataset = generate_multipulse_synthetic_dataset_task(
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
        config_name=config_name
    )

    return {
        "synthetic_dataset": synthetic_dataset,
    }



if __name__ == "__main__":
    result = bolometry_inversion_multipulse_synthetic_splined()
