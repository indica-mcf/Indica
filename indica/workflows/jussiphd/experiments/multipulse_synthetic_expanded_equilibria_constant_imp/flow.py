"""Prefect flow: expanded-equilibria synthetic dataset with fixed impurities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.components.data.expanded_equilibria_generation import (
    build_sampled_plasma_expanded_equilibria_dataset,
    save_equilibrium_plots,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR,
)


DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR
DEFAULT_VIS_DIR = str(Path(__file__).resolve().parent / "outputs")


@task(name="save_equilibrium_plots")
def save_equilibrium_plots_task(
    output_dir: str,
    n_timepoints_per_equilibrium: int,
) -> dict[str, str]:
    return save_equilibrium_plots(
        output_dir=output_dir,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
    )


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
    return build_sampled_plasma_expanded_equilibria_dataset(
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
        impurity_concentrations={"c": float(c_concentration), "ar": float(ar_concentration)},
        impurity_flat_zeff=True,
    )


@flow(name="build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset")
def build_multipulse_synthetic_expanded_equilibria_constant_imp_dataset(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    meta_filename: str = "sample_meta_multipulse_synthetic_expanded_equilibria_constant_imp.csv",
    generate_new_data: bool = True,
    n_timepoints_per_equilibrium: int = 6,
    n_generations: int = 2000,
    config_name: str = "ion_temperature_phantom_run_all_params",
    config_overrides: list[str] | None = None,
    c_concentration: float = 0.05,
    ar_concentration: float = 0.01,
    save_equilibrium_plots: bool = True,
    equilibrium_plots_dir: str = DEFAULT_VIS_DIR,
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
            output_dir=equilibrium_plots_dir,
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
