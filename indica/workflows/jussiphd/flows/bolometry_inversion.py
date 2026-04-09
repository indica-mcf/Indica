"""Prefect-style bolometry inversion flow assembled from modular components.

Order follows the current notebook migration:
1) Sensor geometry visualisation preview
2) Plasma generation + forward-model sample generation
"""

from __future__ import annotations

from typing import Any


from indica.defaults.load_defaults import load_default_objects
from indica.equilibrium import Equilibrium
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from prefect import flow, task
from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters import CoordinateTransform
from indica.workflows.jussiphd.components.data.data_generation import PlasmaGenerator
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
    
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform.set_equilibrium(equilibrium)

    #transform.spot_shape = "square"
    #transform.focal_length = -1000.0

    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    model = PinholeCamera(instrument, power_loss=power_loss)
    model.set_transform(transform)
    model.set_plasma(plasma)

    generator = PlasmaGenerator(model=model, transform=transform)
    generated_plasma = generator.generate()
    measurements, emissivity = generator.run_model(target_plasma=generated_plasma)

    return {
        "plasma": generated_plasma,
        "transform": transform,
        "measurements": measurements,
        "emissivity": emissivity,
    }


@flow(name="bolometry_inversion")
def bolometry_inversion(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    preview_pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
) -> dict[str, Any]:
    

    """Run migrated notebook steps in sequence for iterative workflow development."""
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
    )

    return {
        "preview_transform": transform,
        **sample,
    }


if __name__ == "__main__":
    bolometry_inversion()
