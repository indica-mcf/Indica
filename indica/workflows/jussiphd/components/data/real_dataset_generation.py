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
    allow_channel_based_emissivity: bool = False,
    min_rhop_points: int = 20,
) -> DataArray:
    """Convert raw/ST40 data into emissivity DataArray with dims ('t', 'rhop')."""
    if hasattr(signal, "dims") and hasattr(signal, "values"):
        emissivity = signal
        if "t" not in emissivity.dims:
            emissivity = emissivity.expand_dims(t=[0.5 * (tstart + tend)])

        if "rhop" in emissivity.dims:
            spatial_dim = "rhop"
        elif "channel" in emissivity.dims:
            if not allow_channel_based_emissivity:
                raise ValueError(
                    "Signal appears channel-based (dim='channel') without explicit rhop. "
                    "Refusing to treat it as emissivity profile. "
                    "Set allow_channel_based_emissivity=True to override."
                )
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

    if int(emissivity.sizes["rhop"]) < int(min_rhop_points):
        raise ValueError(
            "Emissivity profile has too few spatial points: "
            f"n_rhop={int(emissivity.sizes['rhop'])} < min_rhop_points={int(min_rhop_points)}"
        )

    if not use_all_timepoints:
        emissivity = emissivity.isel(t=[int(emissivity.sizes["t"] // 2)])
    return emissivity


def _align_emissivity_to_equilibrium_timebase(
    emissivity: DataArray,
    transform: Any,
    use_all_timepoints: bool,
    allow_nearest_fallback: bool = False,
    max_nearest_fallback_gap_s: float | None = None,
) -> DataArray:
    """Align emissivity time coordinates to the transform equilibrium time base."""
    if "t" not in emissivity.dims:
        return emissivity

    equilibrium = getattr(transform, "equilibrium", None)
    if equilibrium is None:
        return emissivity

    eq_t_raw = getattr(equilibrium, "t", None)
    if eq_t_raw is None and hasattr(equilibrium, "rhop") and hasattr(equilibrium.rhop, "t"):
        eq_t_raw = equilibrium.rhop.t
    if eq_t_raw is None:
        return emissivity

    eq_t = eq_t_raw.values if hasattr(eq_t_raw, "values") else eq_t_raw
    eq_t = np.asarray(eq_t, dtype=float).reshape(-1)
    eq_t = eq_t[np.isfinite(eq_t)]
    if eq_t.size == 0:
        return emissivity

    sig_t = np.asarray(emissivity.t.values, dtype=float).reshape(-1)
    sig_t = sig_t[np.isfinite(sig_t)]
    if sig_t.size == 0:
        raise ValueError("Emissivity has no finite time coordinates.")

    # Keep only equilibrium times covered by signal data.
    target_t = eq_t[(eq_t >= sig_t.min()) & (eq_t <= sig_t.max())]

    # If there is no overlap, only allow explicit nearest-time fallback.
    if target_t.size == 0:
        if not allow_nearest_fallback:
            raise ValueError(
                "No overlap between emissivity time grid and equilibrium time grid: "
                f"signal=[{sig_t.min():.6f}, {sig_t.max():.6f}] "
                f"equilibrium=[{eq_t.min():.6f}, {eq_t.max():.6f}]"
            )
        midpoint = float(0.5 * (sig_t.min() + sig_t.max()))
        nearest = eq_t[int(np.argmin(np.abs(eq_t - midpoint)))]
        gap = float(np.abs(nearest - midpoint))
        if max_nearest_fallback_gap_s is not None and gap > float(max_nearest_fallback_gap_s):
            raise ValueError(
                "Nearest-time fallback exceeded max allowed gap: "
                f"gap={gap:.6f}s > max_nearest_fallback_gap_s={float(max_nearest_fallback_gap_s):.6f}s"
            )
        target_t = np.asarray([nearest], dtype=float)

    if not use_all_timepoints and target_t.size > 1:
        target_t = np.asarray([target_t[target_t.size // 2]], dtype=float)

    return emissivity.interp(t=target_t)


def _slice_passes_basic_quality(
    b_slice: np.ndarray,
    eps_slice: np.ndarray,
    min_finite_fraction: float,
    min_nonzero_fraction: float,
    nonzero_threshold: float,
) -> tuple[bool, str]:
    """Check finite/nonzero quality for one (brightness, emissivity) slice pair."""
    b = np.asarray(b_slice, dtype=np.float32).reshape(-1)
    eps = np.asarray(eps_slice, dtype=np.float32).reshape(-1)

    b_finite = float(np.isfinite(b).mean())
    eps_finite = float(np.isfinite(eps).mean())
    if b_finite < float(min_finite_fraction):
        return False, f"brightness finite_fraction={b_finite:.3f} < {min_finite_fraction:.3f}"
    if eps_finite < float(min_finite_fraction):
        return False, f"emissivity finite_fraction={eps_finite:.3f} < {min_finite_fraction:.3f}"

    b_nonzero = float(np.mean(np.abs(np.nan_to_num(b, nan=0.0)) > float(nonzero_threshold)))
    eps_nonzero = float(np.mean(np.abs(np.nan_to_num(eps, nan=0.0)) > float(nonzero_threshold)))
    if b_nonzero < float(min_nonzero_fraction):
        return False, f"brightness nonzero_fraction={b_nonzero:.3f} < {min_nonzero_fraction:.3f}"
    if eps_nonzero < float(min_nonzero_fraction):
        return False, f"emissivity nonzero_fraction={eps_nonzero:.3f} < {min_nonzero_fraction:.3f}"

    return True, "ok"


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
    allow_nearest_time_fallback: bool = False,
    max_nearest_fallback_gap_s: float | None = None,
    allow_channel_based_emissivity: bool = False,
    min_rhop_points: int = 20,
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
        allow_channel_based_emissivity=allow_channel_based_emissivity,
        min_rhop_points=min_rhop_points,
    )
    emissivity = _align_emissivity_to_equilibrium_timebase(
        emissivity=emissivity,
        transform=transform,
        use_all_timepoints=use_all_timepoints,
        allow_nearest_fallback=allow_nearest_time_fallback,
        max_nearest_fallback_gap_s=max_nearest_fallback_gap_s,
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
    apply_basic_quality_filter: bool = True,
    min_finite_fraction: float = 0.95,
    min_nonzero_fraction: float = 0.01,
    nonzero_threshold: float = 0.0,
    allow_nearest_time_fallback: bool = False,
    max_nearest_fallback_gap_s: float | None = None,
    allow_channel_based_emissivity: bool = False,
    min_rhop_points: int = 20,
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
            "num_slices_filtered": None,
            "generated_new_data": False,
            "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
        }

    b_slices: list[np.ndarray] = []
    eps_slices: list[np.ndarray] = []
    sample_meta: list[tuple[int, float]] = []
    skipped: list[tuple[int, str]] = []
    num_slices_filtered = 0

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
                allow_channel_based_emissivity=allow_channel_based_emissivity,
                min_rhop_points=min_rhop_points,
            )
            emissivity = _align_emissivity_to_equilibrium_timebase(
                emissivity=emissivity,
                transform=transform,
                use_all_timepoints=use_all_timepoints,
                allow_nearest_fallback=allow_nearest_time_fallback,
                max_nearest_fallback_gap_s=max_nearest_fallback_gap_s,
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

            pulse_kept = 0
            pulse_filtered = 0
            last_filter_reason = "unknown"
            for t_idx in range(b_arr.shape[0]):
                b_slice = b_arr[t_idx].astype(np.float32)
                eps_slice = eps_arr[t_idx].astype(np.float32)
                if apply_basic_quality_filter:
                    ok, reason = _slice_passes_basic_quality(
                        b_slice=b_slice,
                        eps_slice=eps_slice,
                        min_finite_fraction=min_finite_fraction,
                        min_nonzero_fraction=min_nonzero_fraction,
                        nonzero_threshold=nonzero_threshold,
                    )
                    if not ok:
                        pulse_filtered += 1
                        last_filter_reason = reason
                        continue
                b_slices.append(b_slice)
                eps_slices.append(eps_slice)
                t_val = float(t_values[min(t_idx, len(t_values) - 1)])
                sample_meta.append((int(pulse), t_val))
                pulse_kept += 1
            num_slices_filtered += pulse_filtered
            if pulse_kept == 0:
                skipped.append(
                    (
                        int(pulse),
                        "All slices failed quality filter "
                        f"(filtered={pulse_filtered}, last_reason={last_filter_reason})",
                    )
                )
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
        "num_slices_filtered": int(num_slices_filtered),
        "generated_new_data": True,
        "source_signal": node if node is not None else f"instrument={emissivity_instrument}",
    }
