"""Real ST40 brightness-only dataset generation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np

from indica.workflows.jussiphd.components.data.read_st40 import (
    read_st40_instrument_data,
    read_st40_node,
)


def _to_brightness_rows(
    signal: Any,
    use_all_timepoints: bool,
    tstart: float,
    tend: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert reader output into 2D [n_t, n_channel] brightness rows and t-values."""
    if hasattr(signal, "dims") and hasattr(signal, "values"):
        brightness = signal
        if "t" not in brightness.dims:
            brightness = brightness.expand_dims(t=[0.5 * (tstart + tend)])

        if "channel" in brightness.dims:
            spatial_dim = "channel"
        else:
            spatial_dim = next(dim for dim in brightness.dims if dim != "t")

        if spatial_dim != "channel":
            brightness = brightness.rename({spatial_dim: "channel"})
        brightness = brightness.transpose("t", "channel").astype(np.float32)

        if not use_all_timepoints:
            brightness = brightness.isel(t=[int(brightness.sizes["t"] // 2)])

        rows = np.asarray(brightness.values, dtype=np.float32)
        t_values = np.asarray(brightness.t.values, dtype=np.float64)
    else:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("Signal is scalar; expected 1D/2D brightness data.")
        if arr.ndim == 1:
            rows = arr[None, :]
            t_values = np.asarray([0.5 * (tstart + tend)], dtype=np.float64)
        else:
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            rows = arr
            t_values = np.linspace(tstart, tend, int(arr.shape[0]), dtype=np.float64)
            if not use_all_timepoints:
                mid = int(rows.shape[0] // 2)
                rows = rows[[mid], :]
                t_values = t_values[[mid]]

    if rows.ndim == 1:
        rows = rows[None, :]
    if t_values.ndim == 0:
        t_values = t_values[None]
    return rows, t_values


def _brightness_slice_passes_basic_quality(
    b_slice: np.ndarray,
    min_finite_fraction: float,
    min_nonzero_fraction: float,
    nonzero_threshold: float,
) -> tuple[bool, str]:
    b = np.asarray(b_slice, dtype=np.float32).reshape(-1)
    finite_fraction = float(np.isfinite(b).mean())
    if finite_fraction < float(min_finite_fraction):
        return (
            False,
            f"brightness finite_fraction={finite_fraction:.3f} < {min_finite_fraction:.3f}",
        )

    nonzero_fraction = float(
        np.mean(np.abs(np.nan_to_num(b, nan=0.0)) > float(nonzero_threshold))
    )
    if nonzero_fraction < float(min_nonzero_fraction):
        return (
            False,
            f"brightness nonzero_fraction={nonzero_fraction:.3f} < {min_nonzero_fraction:.3f}",
        )
    return True, "ok"


def generate_and_save_real_multipulse_brightness_dataset(
    pulses: Sequence[int],
    instrument: str = "blom_rz1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    output_dir: str = ".",
    b_filename: str = "b_slices_multipulse_real_rz1_channels.csv",
    meta_filename: str = "sample_meta_multipulse_real_rz1_channels.csv",
    use_all_timepoints: bool = True,
    node: str | None = None,
    revision: int = 0,
    generate_new_data: bool = True,
    verbose: bool = False,
    apply_basic_quality_filter: bool = True,
    min_finite_fraction: float = 0.95,
    min_nonzero_fraction: float = 0.01,
    nonzero_threshold: float = 0.0,
) -> dict[str, Any]:
    """Build and save brightness-only dataset from ST40 RZ1 channel data over pulses."""
    pulse_list = [int(p) for p in pulses]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    b_path = output_path / b_filename
    meta_path = output_path / meta_filename

    if not generate_new_data:
        if not b_path.exists():
            raise FileNotFoundError(
                "generate_new_data=False but brightness dataset file does not exist: "
                f"{b_path}"
            )
        b_arr = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        if b_arr.ndim == 1:
            b_arr = b_arr[None, :]
        if meta_path.exists():
            with meta_path.open(newline="") as f:
                meta_rows = list(csv.reader(f))
            num_meta_rows = max(0, len(meta_rows) - 1)
        else:
            num_meta_rows = None
        return {
            "b_path": str(b_path),
            "meta_path": str(meta_path) if meta_path.exists() else None,
            "num_pairs": int(len(b_arr)),
            "b_shape": tuple(b_arr.shape),
            "num_meta_rows": int(num_meta_rows) if num_meta_rows is not None else None,
            "num_pulses_input": int(len(pulse_list)),
            "num_pulses_skipped": None,
            "skipped": None,
            "num_slices_filtered": None,
            "generated_new_data": False,
            "source_signal": node if node is not None else f"instrument={instrument}:brightness",
        }

    b_slices: list[np.ndarray] = []
    sample_meta: list[tuple[int, float]] = []
    skipped: list[tuple[int, str]] = []
    num_slices_filtered = 0

    for pulse in pulse_list:
        try:
            if node is not None:
                signal = read_st40_node(
                    node=node,
                    pulse=pulse,
                    tstart=tstart,
                    tend=tend,
                    dt=dt,
                    verbose=verbose,
                )
            else:
                instrument_data = read_st40_instrument_data(
                    pulse=pulse,
                    instrument=instrument,
                    tstart=tstart,
                    tend=tend,
                    dt=dt,
                    revision=revision,
                    verbose=verbose,
                )
                if "brightness" in instrument_data:
                    signal = instrument_data["brightness"]
                else:
                    available = ", ".join(sorted(instrument_data.keys()))
                    raise KeyError(
                        f"Expected brightness in ST40Reader output for instrument '{instrument}', "
                        f"available keys: {available}"
                    )

            b_arr, t_values = _to_brightness_rows(
                signal=signal,
                use_all_timepoints=use_all_timepoints,
                tstart=tstart,
                tend=tend,
            )
            if b_arr.shape[1] == 0:
                raise ValueError(f"Empty channel dimension for pulse {pulse}.")
            if b_slices and b_arr.shape[1] != b_slices[0].shape[0]:
                raise ValueError(
                    f"Brightness channel count mismatch for pulse {pulse}: "
                    f"{b_arr.shape[1]} vs {b_slices[0].shape[0]}"
                )

            pulse_kept = 0
            pulse_filtered = 0
            last_filter_reason = "unknown"
            for t_idx in range(b_arr.shape[0]):
                b_slice = b_arr[t_idx].astype(np.float32)
                if apply_basic_quality_filter:
                    ok, reason = _brightness_slice_passes_basic_quality(
                        b_slice=b_slice,
                        min_finite_fraction=min_finite_fraction,
                        min_nonzero_fraction=min_nonzero_fraction,
                        nonzero_threshold=nonzero_threshold,
                    )
                    if not ok:
                        pulse_filtered += 1
                        last_filter_reason = reason
                        continue

                b_slices.append(b_slice)
                t_val = float(t_values[min(t_idx, len(t_values) - 1)])
                sample_meta.append((int(pulse), t_val))
                pulse_kept += 1

            num_slices_filtered += pulse_filtered
            if pulse_kept == 0:
                if pulse_filtered > 0:
                    skipped.append(
                        (
                            int(pulse),
                            "all slices filtered by basic quality checks"
                            f" (last reason: {last_filter_reason})",
                        )
                    )
                else:
                    skipped.append((int(pulse), "no usable slices found"))

        except Exception as exc:
            skipped.append((int(pulse), str(exc)))

    if not b_slices:
        raise RuntimeError(
            "No valid multipulse brightness samples were created. "
            f"All pulses failed: {skipped}"
        )

    b_matrix = np.stack(b_slices).astype(np.float32)
    np.savetxt(b_path, b_matrix, delimiter=",")

    with meta_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pulse", "t"])
        for pulse, t_val in sample_meta:
            writer.writerow([int(pulse), float(t_val)])

    return {
        "b_path": str(b_path),
        "meta_path": str(meta_path),
        "num_pairs": int(len(b_matrix)),
        "b_shape": tuple(b_matrix.shape),
        "num_meta_rows": int(len(sample_meta)),
        "num_pulses_input": int(len(pulse_list)),
        "num_pulses_skipped": int(len(skipped)),
        "skipped": skipped,
        "num_slices_filtered": int(num_slices_filtered),
        "generated_new_data": True,
        "source_signal": node if node is not None else f"instrument={instrument}:brightness",
    }
