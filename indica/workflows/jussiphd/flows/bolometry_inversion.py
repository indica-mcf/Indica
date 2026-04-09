"""Prefect flow for multi-pulse bolometry dataset generation."""

from __future__ import annotations

from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.components.data.data_generation import (
    filter_and_save_valid_pulses,
    generate_and_save_multipulse_real_dataset,
)


@task(name="filter_valid_pulses")
def filter_valid_pulses_task(
    r_start: int,
    r_end: int,
    output_dir: str,
    filename: str | None,
    recompute: bool,
) -> dict[str, Any]:
    return filter_and_save_valid_pulses(
        r_start=r_start,
        r_end=r_end,
        output_dir=output_dir,
        filename=filename,
        recompute=recompute,
    )


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


@flow(name="bolometry_inversion")
def bolometry_inversion(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    output_dir: str = ".",
    pulse_range_start: int = 13500,
    pulse_range_end: int = 14000,
    valids_filename: str | None = None,
    recompute_valids: bool = False,
    multipulse_use_all_timepoints: bool = True,
    multipulse_b_filename: str = "vae_firstpass/b_slices_multipulse.csv",
    multipulse_eps_filename: str = "vae_firstpass/eps_slices_multipulse.csv",
    multipulse_meta_filename: str = "vae_firstpass/sample_meta_multipulse.csv",
    multipulse_generate_new_data: bool = True,
) -> dict[str, Any]:
    """Generate/load valid pulses and build multi-pulse real dataset."""
    pulse_filter_info = filter_valid_pulses_task(
        r_start=pulse_range_start,
        r_end=pulse_range_end,
        output_dir=output_dir,
        filename=valids_filename,
        recompute=recompute_valids,
    )

    multipulse_dataset_info = generate_multipulse_real_dataset_task(
        pulses=pulse_filter_info["valids"],
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

    return {
        "pulse_filter": pulse_filter_info,
        "multipulse_dataset": multipulse_dataset_info,
    }


if __name__ == "__main__":
    bolometry_inversion()
