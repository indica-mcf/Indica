"""Prefect-style bolometry inversion flow assembled from modular components.

Order follows the current notebook migration:
1) Sensor geometry visualisation preview
2) Plasma generation + forward-model sample generation
"""

from __future__ import annotations

from typing import Any

from indica.defaults.load_defaults import load_default_objects
from indica.equilibrium import Equilibrium
from prefect import flow, task
from indica.converters import CoordinateTransform
from indica.workflows.jussiphd.components.data.data_generation import (
    filter_and_save_valid_pulses,
    generate_and_save_multipulse_real_dataset,
    generate_and_save_dataset,
    generate_plasma_sample,
)
from indica.workflows.jussiphd.components.visualisations.sensor_geometry import (
    preview_sensor_geometry,
)


@task(name="preview_sensor_geometry")
def preview_sensor_geometry_task(
    pulse: int,
    instrument: str,
    tstart: float,
    tend: float,
    dt: float,
    verbose: bool,
) -> Any:
    return preview_sensor_geometry(
        pulse=pulse,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        verbose=verbose,
    )


@task(name="generate_plasma_sample")
def generate_plasma_sample_task(
    machine: str,
    instrument: str,
    transform: CoordinateTransform,
    equilibrium: Equilibrium,
) -> dict[str, Any]:
    return generate_plasma_sample(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
    )


@task(name="generate_and_save_dataset")
def generate_and_save_dataset_task(
    machine: str,
    instrument: str,
    transform: CoordinateTransform,
    equilibrium: Equilibrium,
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
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        generate_new_data=generate_new_data,
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
    preview_pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
    write_dataset: bool = True,
    n_generations: int = 100,
    use_all_timepoints: bool = False,
    output_dir: str = ".",
    b_filename: str = "vae_firstpass/b_slices.csv",
    eps_filename: str = "vae_firstpass/eps_slices.csv",
    generate_new_data: bool = True,
    run_pulse_filtering: bool = True,
    pulse_range_start: int = 13500,
    pulse_range_end: int = 14000,
    valids_filename: str | None = None,
    recompute_valids: bool = False,
    write_multipulse_dataset: bool = True,
    multipulse_use_all_timepoints: bool = True,
    multipulse_b_filename: str = "vae_firstpass/b_slices_multipulse.csv",
    multipulse_eps_filename: str = "vae_firstpass/eps_slices_multipulse.csv",
    multipulse_meta_filename: str = "vae_firstpass/sample_meta_multipulse.csv",
    multipulse_generate_new_data: bool = True,
) -> dict[str, Any]:
    """Run migrated notebook steps in sequence for iterative workflow development."""
    equilibrium = load_default_objects(machine, "equilibrium")

    transform = preview_sensor_geometry_task(
        pulse=preview_pulse,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        verbose=verbose,
    )

    sample = generate_plasma_sample_task(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
    )

    dataset_info = None
    if write_dataset:
        dataset_info = generate_and_save_dataset_task(
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

    pulse_filter_info = None
    if run_pulse_filtering:
        pulse_filter_info = filter_valid_pulses_task(
            r_start=pulse_range_start,
            r_end=pulse_range_end,
            output_dir=output_dir,
            filename=valids_filename,
            recompute=recompute_valids,
        )

    multipulse_dataset_info = None
    if write_multipulse_dataset:
        if pulse_filter_info is None:
            raise ValueError(
                "write_multipulse_dataset=True requires run_pulse_filtering=True "
                "to provide pulse list."
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
        "preview_transform": transform,
        "sample": sample,
        "dataset": dataset_info,
        "pulse_filter": pulse_filter_info,
        "multipulse_dataset": multipulse_dataset_info,
    }



if __name__ == "__main__":
    rar=bolometry_inversion()
