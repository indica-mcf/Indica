"""Real ST40 brightness-only dataset generation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np

from indica.workflows.jussiphd.components.data.read_st40 import (
    pulse_has_st40_plasma,
    read_st40_instrument_data,
    read_st40_node,
    read_st40_ppts_signal,
)


def _to_brightness_rows(
    signal: Any,
    use_all_timepoints: bool,
    tstart: float,
    tend: float,
    canonicalize_profile_coordinate: bool = False,
    canonicalize_profile_coordinate_mode: str = "auto",
    use_source_coordinate_grid: bool = False,
    source_coordinate_min: float = 0.0,
    source_coordinate_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert reader output into 2D [n_t, n_channel] brightness rows and t-values."""
    def _canonicalize_rows(
        rows_in: np.ndarray,
        coord_in: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        """Map profile-like coordinates to a consistent core->edge [0, 1] axis."""
        rows_in = np.asarray(rows_in, dtype=np.float32)
        coord = np.asarray(coord_in, dtype=np.float64).reshape(-1)
        if rows_in.ndim != 2 or coord.size != rows_in.shape[1]:
            return rows_in

        finite_coord = np.isfinite(coord)
        if int(finite_coord.sum()) < 2:
            return rows_in

        c = coord.copy()
        mode_l = str(mode).strip().lower()
        if mode_l not in {"auto", "signed_abs", "core_peak_fold"}:
            raise ValueError(
                "canonicalize_profile_coordinate_mode must be one of "
                "{'auto','signed_abs','core_peak_fold'}"
            )

        has_neg = bool(np.nanmin(c[finite_coord]) < 0.0)
        has_pos = bool(np.nanmax(c[finite_coord]) > 0.0)
        do_signed_abs = mode_l == "signed_abs" or (mode_l == "auto" and has_neg and has_pos)
        if do_signed_abs:
            # Typical signed radial coordinate: fold to distance from core.
            c = np.abs(c)
        else:
            # Positive-only coordinate (e.g. R-like): estimate core location from peak.
            # Use robust mean profile across time rows and fold distance from peak coord.
            prof_mean = np.nanmean(rows_in, axis=0)
            valid_core = np.isfinite(prof_mean) & np.isfinite(c)
            if int(valid_core.sum()) >= 2:
                core_rel = int(np.nanargmax(prof_mean[valid_core]))
                core_idx = np.flatnonzero(valid_core)[core_rel]
                c_core = float(c[core_idx])
                c = np.abs(c - c_core)
            else:
                # Last fallback: fold around midpoint of coordinate range.
                c_mid = float(0.5 * (np.nanmin(c[finite_coord]) + np.nanmax(c[finite_coord])))
                c = np.abs(c - c_mid)

        c_valid = c[finite_coord]
        cmin = float(np.nanmin(c_valid))
        cmax = float(np.nanmax(c_valid))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            return rows_in

        c_norm = (c - cmin) / (cmax - cmin)
        # Target stays fixed-length to keep row shape stable across pulses.
        x_target = np.linspace(0.0, 1.0, rows_in.shape[1], dtype=np.float64)
        out = np.full_like(rows_in, np.nan, dtype=np.float32)

        for i in range(rows_in.shape[0]):
            y = np.asarray(rows_in[i], dtype=np.float64)
            valid = np.isfinite(y) & np.isfinite(c_norm)
            if int(valid.sum()) < 2:
                continue

            xv = c_norm[valid]
            yv = y[valid]
            order = np.argsort(xv)
            xv = xv[order]
            yv = yv[order]

            # Collapse duplicate x (common after abs-folding) by averaging.
            x_unique, inv = np.unique(xv, return_inverse=True)
            y_unique = np.zeros_like(x_unique, dtype=np.float64)
            cnt = np.zeros_like(x_unique, dtype=np.int32)
            for j, g in enumerate(inv):
                y_unique[g] += yv[j]
                cnt[g] += 1
            y_unique = y_unique / np.maximum(cnt, 1)

            if x_unique.size < 2:
                continue
            out[i] = np.interp(
                x_target,
                x_unique,
                y_unique,
                left=np.nan,
                right=np.nan,
            ).astype(np.float32)
        return out

    def _resample_rows_on_source_coordinate(
        rows_in: np.ndarray,
        coord_in: np.ndarray,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        """Resample rows onto a shared physical coordinate grid without per-row stretching."""
        rows_in = np.asarray(rows_in, dtype=np.float32)
        coord = np.asarray(coord_in, dtype=np.float64).reshape(-1)
        if rows_in.ndim != 2 or coord.size != rows_in.shape[1]:
            return rows_in

        finite_coord = np.isfinite(coord)
        if int(finite_coord.sum()) < 2:
            return rows_in

        c = coord.copy()
        mode_l = str(canonicalize_profile_coordinate_mode).strip().lower()
        has_neg = bool(np.nanmin(c[finite_coord]) < 0.0)
        has_pos = bool(np.nanmax(c[finite_coord]) > 0.0)
        do_signed_abs = mode_l == "signed_abs" or (mode_l == "auto" and has_neg and has_pos)
        if do_signed_abs:
            c = np.abs(c)

        x_lo = float(x_min)
        x_hi = float(x_max)
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
            return rows_in

        x_target = np.linspace(x_lo, x_hi, rows_in.shape[1], dtype=np.float64)
        out = np.full_like(rows_in, np.nan, dtype=np.float32)
        for i in range(rows_in.shape[0]):
            y = np.asarray(rows_in[i], dtype=np.float64)
            valid = np.isfinite(y) & np.isfinite(c)
            if int(valid.sum()) < 2:
                continue

            xv = c[valid]
            yv = y[valid]
            order = np.argsort(xv)
            xv = xv[order]
            yv = yv[order]

            x_unique, inv = np.unique(xv, return_inverse=True)
            y_unique = np.zeros_like(x_unique, dtype=np.float64)
            cnt = np.zeros_like(x_unique, dtype=np.int32)
            for j, g in enumerate(inv):
                y_unique[g] += yv[j]
                cnt[g] += 1
            y_unique = y_unique / np.maximum(cnt, 1)
            if x_unique.size < 2:
                continue

            out[i] = np.interp(x_target, x_unique, y_unique, left=np.nan, right=np.nan).astype(np.float32)
        return out

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
        if canonicalize_profile_coordinate:
            coord_values = np.asarray(brightness.coords["channel"].values, dtype=np.float64)
            rows = _canonicalize_rows(
                rows,
                coord_values,
                canonicalize_profile_coordinate_mode,
            )
        elif use_source_coordinate_grid and "channel" in brightness.coords:
            coord_values = np.asarray(brightness.coords["channel"].values, dtype=np.float64)
            if source_coordinate_max is None:
                finite = coord_values[np.isfinite(coord_values)]
                if finite.size >= 2:
                    x_max = float(np.nanmax(np.abs(finite)))
                else:
                    x_max = float(rows.shape[1] - 1)
            else:
                x_max = float(source_coordinate_max)
            rows = _resample_rows_on_source_coordinate(
                rows_in=rows,
                coord_in=coord_values,
                x_min=float(source_coordinate_min),
                x_max=x_max,
            )
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
    ppts_profile_key: str | None = None,
    revision: int = 0,
    generate_new_data: bool = True,
    verbose: bool = False,
    apply_basic_quality_filter: bool = True,
    min_finite_fraction: float = 0.95,
    min_nonzero_fraction: float = 0.01,
    nonzero_threshold: float = 0.0,
    require_plasma_summary: bool = False,
    canonicalize_profile_coordinate: bool = False,
    canonicalize_profile_coordinate_mode: str = "auto",
    use_source_coordinate_grid: bool = False,
    source_coordinate_min: float = 0.0,
    source_coordinate_max: float | None = None,
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
            "num_no_plasma_skipped": None,
            "skipped": None,
            "num_slices_filtered": None,
            "generated_new_data": False,
            "source_signal": (
                f"ppts:{ppts_profile_key}"
                if ppts_profile_key is not None
                else (node if node is not None else f"instrument={instrument}:brightness")
            ),
        }

    b_slices: list[np.ndarray] = []
    sample_meta: list[tuple[int, float]] = []
    skipped: list[tuple[int, str]] = []
    num_slices_filtered = 0
    num_no_plasma_skipped = 0

    for pulse in pulse_list:
        if require_plasma_summary and not pulse_has_st40_plasma(
            pulse=pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=verbose,
        ):
            skipped.append((int(pulse), "No plasma according to \\ST40::TOP.SUMMARY:PLASMA"))
            num_no_plasma_skipped += 1
            continue

        try:
            if node is not None and ppts_profile_key is not None:
                raise ValueError("Pass only one of `node` or `ppts_profile_key`.")
            if ppts_profile_key is not None:
                signal = read_st40_ppts_signal(
                    signal_key=ppts_profile_key,
                    pulse=pulse,
                    tstart=tstart,
                    tend=tend,
                    dt=dt,
                    revision=revision,
                    verbose=verbose,
                )
            elif node is not None:
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
                canonicalize_profile_coordinate=canonicalize_profile_coordinate,
                canonicalize_profile_coordinate_mode=canonicalize_profile_coordinate_mode,
                use_source_coordinate_grid=use_source_coordinate_grid,
                source_coordinate_min=source_coordinate_min,
                source_coordinate_max=source_coordinate_max,
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
        "num_no_plasma_skipped": int(num_no_plasma_skipped),
        "generated_new_data": True,
        "source_signal": (
            f"ppts:{ppts_profile_key}"
            if ppts_profile_key is not None
            else (node if node is not None else f"instrument={instrument}:brightness")
        ),
    }
