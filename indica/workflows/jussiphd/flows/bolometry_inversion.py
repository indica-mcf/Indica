"""Prefect flow for multi-pulse bolometry dataset generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Sequence

from prefect import flow, task

from indica.workflows.jussiphd.components.filtering.dataset_filters import (
    apply_zero_and_tomo_filters,
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
    min_valid_channels_required: int = 1,
    filters_overwrite: bool = True,
    filters_output_dir: str | None = None,
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

    return {
        "multipulse_dataset": multipulse_dataset_info,
        "filtered_dataset": filtered_dataset_info,
    }


if __name__ == "__main__":
    bolometry_inversion()
