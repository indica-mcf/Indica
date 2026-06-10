"""Real ST40 emissivity-to-brightness dataset generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from xarray import DataArray

from indica.workflows.jussiphd.components.data.read_st40 import (
    read_st40_emission_signal,
    read_st40_instrument_data,
)


def _to_emissivity_profile(
    signal: Any,
    use_all_timepoints: bool,
    tstart: float,
    tend: float,
) -> DataArray:
    """Convert raw/ST40 data into emissivity DataArray with dims ('t', 'rhop')."""
    if hasattr(signal, "dims") and hasattr(signal, "values"):
        emissivity = signal
        if "t" not in emissivity.dims:
            emissivity = emissivity.expand_dims(t=[0.5 * (tstart + tend)])

        if "rhop" in emissivity.dims:
            spatial_dim = "rhop"
        elif "channel" in emissivity.dims:
            spatial_dim = "channel"
        else:
            spatial_dim = next(dim for dim in emissivity.dims if dim != "t")

        if spatial_dim != "rhop":
            emissivity = emissivity.rename({spatial_dim: "rhop"})

        if "rhop" not in emissivity.coords:
            emissivity = emissivity.assign_coords(
                rhop=np.linspace(0.0, 1.0, int(emissivity.sizes["rhop"]), dtype=np.float32)
            )

        emissivity = emissivity.transpose("t", "rhop").astype(np.float32)
    else:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("Signal is scalar; expected 1D or 2D emissivity-like data.")
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        n_t, n_rhop = int(arr.shape[0]), int(arr.shape[1])
        t_coords = (
            np.asarray([0.5 * (tstart + tend)], dtype=np.float32)
            if n_t == 1
            else np.linspace(tstart, tend, n_t, dtype=np.float32)
        )
        rhop_coords = np.linspace(0.0, 1.0, n_rhop, dtype=np.float32)
        emissivity = DataArray(arr, coords=[("t", t_coords), ("rhop", rhop_coords)])

    if not use_all_timepoints:
        emissivity = emissivity.isel(t=[int(emissivity.sizes["t"] // 2)])
    return emissivity


def load_real_transform_from_pulse(
    instrument: str,
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    equilibrium: Any,
    revision: int = 0,
    verbose: bool = False,
) -> Any:
    """Read LOS transform for one instrument and attach equilibrium."""
    instrument_data = read_st40_instrument_data(
        pulse=pulse,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        revision=revision,
        verbose=verbose,
    )
    quantity = next(iter(instrument_data))
    transform = instrument_data[quantity].attrs["transform"]
    transform.set_equilibrium(equilibrium, force=True)
    return transform


def generate_and_save_real_dataset(
    machine: str,
    instrument: str,
    emissivity_instrument: str,
    transform: Any,
    equilibrium: Any,
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    output_dir: str = ".",
    b_filename: str = "b_slices_single_real.csv",
    eps_filename: str = "eps_slices_single_real.csv",
    use_all_timepoints: bool = True,
    node: str | None = None,
    revision: int = 0,
    generate_new_data: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build (brightness, emissivity) pairs from real emissivity + forward projection."""
    _ = machine, instrument, equilibrium
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
            "source_pulse": int(pulse),
            "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
        }

    signal = read_st40_emission_signal(
        instrument=emissivity_instrument,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        revision=revision,
        verbose=verbose,
        node=node,
    )
    emissivity = _to_emissivity_profile(
        signal=signal,
        use_all_timepoints=use_all_timepoints,
        tstart=tstart,
        tend=tend,
    )
    brightness = transform.integrate_on_los(emissivity, t=emissivity.t)
    if "t" in brightness.dims:
        emissivity = emissivity.interp(t=brightness.t)

    b_arr = np.asarray(brightness.values, dtype=np.float32)
    eps_arr = np.asarray(emissivity.values, dtype=np.float32)
    if b_arr.ndim == 1:
        b_arr = b_arr[None, :]
    if eps_arr.ndim == 1:
        eps_arr = eps_arr[None, :]

    np.savetxt(b_path, b_arr, delimiter=",")
    np.savetxt(eps_path, eps_arr, delimiter=",")

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "num_pairs": int(len(b_arr)),
        "b_shape": tuple(b_arr.shape),
        "eps_shape": tuple(eps_arr.shape),
        "generated_new_data": True,
        "source_pulse": int(pulse),
        "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
    }
