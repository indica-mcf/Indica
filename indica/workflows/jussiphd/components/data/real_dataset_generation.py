"""Real ST40 emissivity-to-brightness dataset generation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from xarray import DataArray

from indica.workflows.jussiphd.components.data.read_st40 import (
    read_st40_emission_signal,
    read_st40_instrument_data,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
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


def generate_and_save_real_multipulse_dataset(
    pulses: Sequence[int],
    machine: str,
    instrument: str,
    emissivity_instrument: str,
    tstart: float,
    tend: float,
    dt: float,
    output_dir: str = ".",
    b_filename: str = "b_slices_multipulse_real.csv",
    eps_filename: str = "eps_slices_multipulse_real.csv",
    meta_filename: str = "sample_meta_multipulse_real.csv",
    use_all_timepoints: bool = True,
    node: str | None = None,
    revision: int = 0,
    generate_new_data: bool = True,
    verbose: bool = False,
    static_transform: Any | None = None,
) -> dict[str, Any]:
    """Build multi-pulse (brightness, emissivity) pairs from real ST40 data."""
    _ = machine
    pulse_list = [int(p) for p in pulses]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    eps_path = output_path / eps_filename
    meta_path = output_path / meta_filename

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
        if meta_path.exists():
            with meta_path.open(newline="") as f:
                meta_rows = list(csv.reader(f))
            num_meta_rows = max(0, len(meta_rows) - 1)
        else:
            num_meta_rows = None
        return {
            "b_path": str(b_path),
            "eps_path": str(eps_path),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "num_pairs": int(len(b_arr)),
            "b_shape": tuple(b_arr.shape),
            "eps_shape": tuple(eps_arr.shape),
            "num_meta_rows": int(num_meta_rows) if num_meta_rows is not None else None,
            "num_pulses_input": int(len(pulse_list)),
            "num_pulses_skipped": None,
            "skipped": None,
            "generated_new_data": False,
            "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
        }

    b_slices: list[np.ndarray] = []
    eps_slices: list[np.ndarray] = []
    sample_meta: list[tuple[int, float]] = []
    skipped: list[tuple[int, str]] = []

    for pulse in pulse_list:
        try:
            if static_transform is not None:
                transform = static_transform
            else:
                equilibrium = load_real_equilibrium_from_pulse(
                    pulse=pulse,
                    tstart=tstart,
                    tend=tend,
                    dt=dt,
                    verbose=verbose,
                )
                transform = load_real_transform_from_pulse(
                    instrument=instrument,
                    pulse=pulse,
                    tstart=tstart,
                    tend=tend,
                    dt=dt,
                    equilibrium=equilibrium,
                    revision=revision,
                    verbose=verbose,
                )
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
                t_values = np.asarray(brightness.t.values, dtype=float)
            else:
                t_values = np.asarray([0.5 * (tstart + tend)], dtype=float)

            b_arr = np.asarray(brightness.values, dtype=np.float32)
            eps_arr = np.asarray(emissivity.values, dtype=np.float32)
            if b_arr.ndim == 1:
                b_arr = b_arr[None, :]
            if eps_arr.ndim == 1:
                eps_arr = eps_arr[None, :]
            if b_arr.shape[0] != eps_arr.shape[0]:
                raise ValueError(
                    f"Mismatched time samples for pulse {pulse}: "
                    f"b={b_arr.shape}, eps={eps_arr.shape}"
                )
            if b_arr.shape[1] == 0 or eps_arr.shape[1] == 0:
                raise ValueError(f"Empty channel/rhop dimension for pulse {pulse}.")

            if b_slices and b_arr.shape[1] != b_slices[0].shape[0]:
                raise ValueError(
                    f"Brightness channel count mismatch for pulse {pulse}: "
                    f"{b_arr.shape[1]} vs {b_slices[0].shape[0]}"
                )
            if eps_slices and eps_arr.shape[1] != eps_slices[0].shape[0]:
                raise ValueError(
                    f"Emissivity rhop count mismatch for pulse {pulse}: "
                    f"{eps_arr.shape[1]} vs {eps_slices[0].shape[0]}"
                )

            for t_idx in range(b_arr.shape[0]):
                b_slices.append(b_arr[t_idx].astype(np.float32))
                eps_slices.append(eps_arr[t_idx].astype(np.float32))
                t_val = float(t_values[min(t_idx, len(t_values) - 1)])
                sample_meta.append((int(pulse), t_val))
        except Exception as exc:  # pragma: no cover
            skipped.append((int(pulse), str(exc)))

    b_out = np.asarray(b_slices, dtype=np.float32)
    eps_out = np.asarray(eps_slices, dtype=np.float32)
    if b_out.size == 0 or eps_out.size == 0:
        raise RuntimeError(
            "No valid multipulse samples were created. "
            f"All pulses failed: {skipped}"
        )

    np.savetxt(b_path, b_out, delimiter=",")
    np.savetxt(eps_path, eps_out, delimiter=",")
    with meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse", "time_s"])
        writer.writerows(sample_meta)

    return {
        "b_path": str(b_path),
        "eps_path": str(eps_path),
        "meta_path": str(meta_path),
        "num_pairs": int(len(b_out)),
        "b_shape": tuple(b_out.shape),
        "eps_shape": tuple(eps_out.shape),
        "num_meta_rows": int(len(sample_meta)),
        "num_pulses_input": int(len(pulse_list)),
        "num_pulses_skipped": int(len(skipped)),
        "skipped": skipped,
        "generated_new_data": True,
        "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
    }
