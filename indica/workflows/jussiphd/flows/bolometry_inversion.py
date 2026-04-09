"""Prefect-style bolometry inversion flow assembled from modular components.

Order follows the current notebook migration:
1) Sensor geometry visualisation preview
2) Plasma generation + forward-model sample generation
"""

from __future__ import annotations

from typing import Any

from indica.defaults.load_defaults import load_default_objects
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.workflows.jussiphd.components.data.data_generation import PlasmaGenerator
from indica.workflows.jussiphd.components.visualisations.sensor_geometry import (
    preview_sensor_geometry,
)

try:
    from prefect import flow, task
except ImportError:  # pragma: no cover
    def flow(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def task(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


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
) -> dict[str, Any]:
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)
    transform.spot_shape = "square"
    transform.focal_length = -1000.0

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
    preview = preview_sensor_geometry_task(
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
        "preview_transform": preview,
        **sample,
    }


if __name__ == "__main__":
    bolometry_inversion()
