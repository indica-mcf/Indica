"""Plasma data-generation utilities extracted from the surrogate notebook."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from indica import Equilibrium
from indica.examples.example_plasma import example_plasma
from indica.models import PinholeCamera
from indica.operators.atomic_data import default_atomic_data
from indica.readers import ST40Reader
from indica.readers.st40_pulse_filtering import filter_pulses
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
    b_filename: str = "vae_firstpass/b_slices.csv",
    eps_filename: str = "vae_firstpass/eps_slices.csv",
    generate_new_data: bool = True,
) -> dict[str, Any]:
    """Generate (brightness, emissivity) pairs and write them to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_path = output_path / eps_filename

    if not generate_new_data:
        if not b_path.exists() or not eps_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but dataset files do not exist: "
                f"{b_path}, {eps_path}"
            )
        b_arr = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        eps_arr = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
        if b_arr.ndim == 1:
            b_arr = b_arr[None, :]
        if eps_arr.ndim == 1:
            eps_arr = eps_arr[None, :]
        return {
            "b_path": str(b_path),
            "eps_path": str(eps_path),
            "num_pairs": int(len(b_arr)),
            "b_shape": tuple(b_arr.shape),
            "eps_shape": tuple(eps_arr.shape),
            "generated_new_data": False,
        }

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

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_path, eps_arr, delimiter=",")

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "num_pairs": int(len(b_arr)),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "generated_new_data": True,
    }


def filter_and_save_valid_pulses(
    r_start: int = 13500,
    r_end: int = 14000,
    output_dir: str = ".",
    filename: str | None = None,
    recompute: bool = False,
) -> dict[str, Any]:
    """Filter a pulse range and store/load valid pulses as a pickle file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"valids_range_{r_start}_{r_end}.pkl"
    cache_path = output_path / filename

    if recompute or not cache_path.exists():
        valids, invalids = filter_pulses(range(r_start, r_end))
        with cache_path.open("wb") as f:
            pickle.dump(valids, f)
        return {
            "valids": list(valids),
            "invalids": list(invalids),
            "cache_path": str(cache_path),
            "num_valids": int(len(valids)),
            "num_invalids": int(len(invalids)),
            "loaded_from_cache": False,
        }

    with cache_path.open("rb") as f:
        valids = pickle.load(f)

    return {
        "valids": list(valids),
        "invalids": None,
        "cache_path": str(cache_path),
        "num_valids": int(len(valids)),
        "num_invalids": None,
        "loaded_from_cache": True,
    }


def build_real_model_for_pulse(
    pulse: int,
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
) -> PinholeCamera:
    """Build a real-data-seeded bolometry model for one pulse."""
    reader = ST40Reader(pulse, tstart - dt, tend + dt, dt=dt, verbose=False)

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)

    plasma = example_plasma(
        machine=machine,
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion="h",
        impurities=("c", "ar", "he"),
        full_run=False,
        n_rad=41,
        n_R=100,
        n_z=100,
    )
    plasma.set_equilibrium(equilibrium)

    ppts = reader.get("", "ppts", 0)
    plasma.electron_density.loc[dict(t=plasma.t)] = (
        ppts["ne_rhop"].interp(t=plasma.t, rhop=plasma.rhop).transpose("t", "rhop").values
    )
    plasma.electron_temperature.loc[dict(t=plasma.t)] = (
        ppts["te_rhop"].interp(t=plasma.t, rhop=plasma.rhop).transpose("t", "rhop").values
    )

    instrument_data = reader.get("", instrument, 0)
    first_quantity = next(iter(instrument_data))
    transform = instrument_data[first_quantity].attrs["transform"]
    transform.set_equilibrium(equilibrium, force=True)
    transform.spot_shape = "square"
    transform.focal_length = -1000.0

    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])
    model = PinholeCamera(instrument, power_loss=power_loss)
    model.set_transform(transform)
    model.set_plasma(plasma)
    return model


def generate_and_save_multipulse_real_dataset(
    pulses: Sequence[int],
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_all_timepoints: bool = True,
    output_dir: str = ".",
    b_filename: str = "vae_firstpass/b_slices_multipulse.csv",
    eps_filename: str = "vae_firstpass/eps_slices_multipulse.csv",
    meta_filename: str = "vae_firstpass/sample_meta_multipulse.csv",
    generate_new_data: bool = True,
) -> dict[str, Any]:
    """Generate and save a multi-pulse real-data dataset."""
    pulse_list = [int(p) for p in pulses]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_path = output_path / eps_filename
    meta_path = output_path / meta_filename

    if not generate_new_data:
        if not b_path.exists() or not eps_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but multipulse files do not exist: "
                f"{b_path}, {eps_path}, {meta_path}"
            )
        b_arr = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        eps_arr = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
        if b_arr.ndim == 1:
            b_arr = b_arr[None, :]
        if eps_arr.ndim == 1:
            eps_arr = eps_arr[None, :]
        with meta_path.open(newline="") as f:
            meta_rows = list(csv.reader(f))
        num_meta_rows = max(0, len(meta_rows) - 1)
        return {
            "b_path": str(b_path),
            "eps_path": str(eps_path),
            "meta_path": str(meta_path),
            "num_pairs": int(len(b_arr)),
            "num_meta_rows": int(num_meta_rows),
            "num_pulses_input": int(len(pulse_list)),
            "num_pulses_skipped": None,
            "skipped": None,
            "generated_new_data": False,
        }

    b_slices: list[np.ndarray] = []
    eps_slices: list[np.ndarray] = []
    sample_meta: list[tuple[int, float]] = []
    skipped: list[tuple[int, str]] = []

    for pulse in pulse_list:
        try:
            model = build_real_model_for_pulse(
                pulse=int(pulse),
                machine=machine,
                instrument=instrument,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
            bckc, emissivity = model(return_emissivity=True)
            measurements = bckc["brightness"]

            if use_all_timepoints:
                t_indices = range(measurements.sizes["t"])
            else:
                t_indices = [int(np.random.randint(measurements.sizes["t"]))]

            for t_idx in t_indices:
                channel_vector = measurements.isel(t=t_idx).values.astype(np.float32)
                emissivity_slice = emissivity.isel(t=t_idx).values.astype(np.float32)
                b_slices.append(channel_vector)
                eps_slices.append(emissivity_slice)
                sample_meta.append((int(pulse), float(measurements.t.isel(t=t_idx).values)))
        except Exception as exc:  # pragma: no cover
            skipped.append((int(pulse), str(exc)))

    b_arr = np.asarray(b_slices, dtype=np.float32)
    eps_arr = np.asarray(eps_slices, dtype=np.float32)

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_path, eps_arr, delimiter=",")
    with meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse", "time_s"])
        writer.writerows(sample_meta)

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "meta_path": str(meta_path),
        "num_pairs": int(len(b_arr)),
        "num_meta_rows": int(len(sample_meta)),
        "num_pulses_input": int(len(pulse_list)),
        "num_pulses_skipped": int(len(skipped)),
        "skipped": skipped,
        "generated_new_data": True,
    }
