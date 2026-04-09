"""Plasma data-generation utilities extracted from the surrogate notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.workflows.jussiphd.plasma_profiler_init import (
    build_plasma_profiler,
    load_bda_config,
    sample_prior_parameters,
)


DEFAULT_BDA_OVERRIDES: tuple[str, ...] = (
    "plasma.settings.n_rad=41",
    "tstart=0.04",
    "tend=0.15",
    "dt=0.01",
)


class PlasmaGenerator:
    """Generate random plasma states and run the bolometry forward model."""

    def __init__(
        self,
        model: Any,
        transform: Any,
        config_name: str = "ion_temperature_phantom_run_all_params",
        overrides: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.cfg = load_bda_config(
            config_name=config_name,
            overrides=list(overrides) if overrides is not None else list(DEFAULT_BDA_OVERRIDES),
        )
        self.plasma_profiler = build_plasma_profiler(self.cfg)

    def generate(self) -> Any:
        """Sample prior parameters and build a plasma instance."""
        all_params = sample_prior_parameters(self.cfg)
        self.plasma_profiler(all_params)
        return self.plasma_profiler.plasma

    def run_model(self, target_plasma: Any | None = None) -> tuple[Any, Any]:
        """Run forward model and return (brightness, emissivity)."""
        if target_plasma is not None:
            self.model.set_plasma(target_plasma)
        else:
            self.model.set_plasma(self.plasma_profiler.plasma)

        self.model.set_transform(self.transform)
        bckc, emissivity = self.model(return_emissivity=True)
        measurements = bckc["brightness"]
        return measurements, emissivity


def sample_plasma(
    model: Any,
    transform: Any,
    config_name: str = "ion_temperature_phantom_run_all_params",
    overrides: Sequence[str] | None = None,
) -> Any:
    """Create a generator and return one sampled plasma instance."""
    generator = PlasmaGenerator(
        model=model,
        transform=transform,
        config_name=config_name,
        overrides=overrides,
    )
    return generator.generate()


def generate_plasma_sample(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    config_name: str = "ion_temperature_phantom_run_all_params",
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build model, sample plasma, run forward model, and return sample bundle."""
    _ = machine  # reserved for future machine-specific branching
    transform.set_equilibrium(equilibrium)
    transform.spot_shape = "square"
    transform.focal_length = -1000.0

    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    model = PinholeCamera(instrument, power_loss=power_loss)
    model.set_transform(transform)

    generated_plasma = sample_plasma(
        model=model,
        transform=transform,
        config_name=config_name,
        overrides=overrides,
    )
    model.set_plasma(generated_plasma)
    bckc, emissivity = model(return_emissivity=True)
    measurements = bckc["brightness"]

    return {
        "plasma": generated_plasma,
        "transform": transform,
        "measurements": measurements,
        "emissivity": emissivity,
    }


def generate_and_save_dataset(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    n_generations: int = 4000,
    use_all_timepoints: bool = False,
    output_dir: str = ".",
    b_filename: str = "b_slices.csv",
    eps_filename: str = "eps_slices.csv",
) -> dict[str, Any]:
    """Generate (brightness, emissivity) pairs and write them to CSV files."""
    b_slices: list[np.ndarray] = []
    eps_slices: list[np.ndarray] = []

    for _ in range(n_generations):
        sample = generate_plasma_sample(
            machine=machine,
            instrument=instrument,
            transform=transform,
            equilibrium=equilibrium,
        )
        measurements = sample["measurements"]
        emissivity = sample["emissivity"]

        if use_all_timepoints:
            t_indices = range(measurements.sizes["t"])
        else:
            t_indices = [int(np.random.randint(measurements.sizes["t"]))]

        for t_idx in t_indices:
            channel_vector = measurements.isel(t=t_idx).values.astype(np.float32)
            emissivity_slice = emissivity.isel(t=t_idx).values.astype(np.float32)
            b_slices.append(channel_vector)
            eps_slices.append(emissivity_slice)

    b_arr = np.asarray(b_slices, dtype=np.float32)
    eps_arr = np.asarray(eps_slices, dtype=np.float32)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_path = output_path / eps_filename
    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_path, eps_arr, delimiter=",")

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "num_pairs": int(len(b_arr)),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
    }
