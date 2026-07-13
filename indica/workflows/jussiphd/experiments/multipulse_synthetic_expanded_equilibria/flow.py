"""Prefect flow: expand synthetic brightness with multiple real equilibria."""

from __future__ import annotations

from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.components.data.expanded_equilibria_generation import (
    expand_brightness_with_equilibria,
)
from indica.workflows.jussiphd.datasets.paths import MULTIPULSE_SYNTHETIC_DATA_DIR
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR,
)


DEFAULT_INPUT_EPS_PATH = str(MULTIPULSE_SYNTHETIC_DATA_DIR / "eps_slices_multipulse_synthetic.csv")
DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_DATA_DIR_STR


@task(name="expand_brightness_with_equilibria")
def expand_brightness_with_equilibria_task(
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    n_timepoints_per_equilibrium: int,
) -> dict[str, Any]:
    return expand_brightness_with_equilibria(
        eps_path=eps_path,
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
    )


@flow(name="build_multipulse_synthetic_expanded_equilibria_dataset")
def build_multipulse_synthetic_expanded_equilibria_dataset(
    eps_path: str = DEFAULT_INPUT_EPS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    b_filename: str = "b_slices_multipulse_synthetic_expanded_equilibria.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_expanded_equilibria.csv",
    meta_filename: str = "sample_meta_multipulse_synthetic_expanded_equilibria.csv",
    generate_new_data: bool = True,
    n_timepoints_per_equilibrium: int = 5,
) -> dict[str, Any]:
    """Expand brightness by projecting fixed synthetic eps over multiple equilibria."""
    return expand_brightness_with_equilibria_task(
        eps_path=eps_path,
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
    )


if __name__ == "__main__":
    result = build_multipulse_synthetic_expanded_equilibria_dataset()
    print("Expanded-equilibria synthetic dataset complete")
    print(result)
