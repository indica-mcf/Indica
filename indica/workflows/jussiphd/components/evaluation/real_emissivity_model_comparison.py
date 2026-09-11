"""Compare saved VAE emissivity inference against real-node emissivity products."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from indica.workflows.jussiphd.components.data.read_st40 import read_st40_node_with_dims
from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset


def _load_vae(model_path: str) -> CVAENetwork:
    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_scaling=int(ckpt.get("hidden_scaling", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _to_2d_rows(arr: Any) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("Expected array-like node data, got scalar.")
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return np.asarray(x, dtype=np.float64)


def _extract_time_axis(
    rows: np.ndarray,
    dims: Sequence[np.ndarray] | None,
    tstart: float,
    tend: float,
) -> np.ndarray:
    """Resolve time axis for row matrix using dim_of(node,0) when available."""
    n_t = int(rows.shape[0])
    if dims is not None and len(dims) > 0:
        t = np.asarray(dims[0], dtype=np.float64).reshape(-1)
        if t.size == n_t and np.all(np.isfinite(t)):
            return t
    if n_t <= 1:
        return np.asarray([0.5 * (float(tstart) + float(tend))], dtype=np.float64)
    return np.linspace(float(tstart), float(tend), n_t, dtype=np.float64)


def _interp_rows_in_time(
    rows: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """Interpolate profile rows in time onto target timestamps."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D rows for time interpolation, got shape {x.shape}.")
    ts = np.asarray(t_src, dtype=np.float64).reshape(-1)
    td = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if ts.size != x.shape[0]:
        raise ValueError("Source time axis length does not match row count.")

    valid_t = np.isfinite(ts)
    if int(np.sum(valid_t)) < 1:
        return np.full((td.size, x.shape[1]), np.nan, dtype=np.float64)

    ts = ts[valid_t]
    x = x[valid_t]
    order = np.argsort(ts)
    ts = ts[order]
    x = x[order]

    if ts.size == 1:
        return np.repeat(x[:1], td.size, axis=0)

    out = np.full((td.size, x.shape[1]), np.nan, dtype=np.float64)
    for j in range(x.shape[1]):
        y = x[:, j]
        valid = np.isfinite(y) & np.isfinite(ts)
        if int(np.sum(valid)) < 2:
            continue
        out[:, j] = np.interp(td, ts[valid], y[valid], left=np.nan, right=np.nan)
    return out


def _resample_row_to_dim(row: np.ndarray, target_dim: int) -> np.ndarray:
    y = np.asarray(row, dtype=np.float64).reshape(-1)
    if y.size == int(target_dim):
        return y
    if y.size < 2:
        return np.full(int(target_dim), np.nan, dtype=np.float64)
    x_src = np.linspace(0.0, 1.0, y.size, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, int(target_dim), dtype=np.float64)
    return np.interp(x_dst, x_src, y).astype(np.float64)


def _choose_time_indices(n_t: int, max_traces: int) -> np.ndarray:
    n = int(max(1, n_t))
    m = int(max(1, max_traces))
    if n <= m:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, m)
    return np.unique(np.round(idx).astype(int))


def compare_saved_model_vs_real_emissivity_nodes(
    pulses: Sequence[int],
    model_path: str,
    reference_b_path: str,
    reference_eps_path: str,
    output_dir: str,
    dv1_emission_node: str,
    xy1_emis_loc_node: str,
    xy1_emis_loc_err_node: str,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    k_samples: int = 40,
    seed: int | None = 0,
    enforce_nonnegative_output: bool = True,
    max_plot_times_per_pulse: int = 6,
) -> dict[str, Any]:
    """Run VAE emissivity inference on real DV1 bolometry and compare to XY1 T1D emissivity."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(b_path=reference_b_path, eps_path=reference_eps_path)
    model = _load_vae(model_path)
    rng = np.random.default_rng(seed)

    metrics_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    pulse_plots: list[str] = []

    for pulse in [int(p) for p in pulses]:
        try:
            b_bundle = read_st40_node_with_dims(
                node=dv1_emission_node,
                pulse=pulse,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
            e_loc_bundle = read_st40_node_with_dims(
                node=xy1_emis_loc_node,
                pulse=pulse,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
            e_err_bundle = read_st40_node_with_dims(
                node=xy1_emis_loc_err_node,
                pulse=pulse,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "pulse": int(pulse),
                    "n_times": 0,
                    "mean_rmse": np.nan,
                    "mean_coverage": np.nan,
                    "mean_abs_resid_over_err": np.nan,
                    "status": f"read_failed: {exc}",
                }
            )
            continue

        b_rows = _to_2d_rows(b_bundle["values"])
        e_loc_rows = _to_2d_rows(e_loc_bundle["values"])
        e_err_rows = _to_2d_rows(e_err_bundle["values"])

        b_t = _extract_time_axis(
            b_rows,
            dims=b_bundle.get("dims"),
            tstart=tstart,
            tend=tend,
        )
        e_loc_t = _extract_time_axis(
            e_loc_rows,
            dims=e_loc_bundle.get("dims"),
            tstart=tstart,
            tend=tend,
        )
        e_err_t = _extract_time_axis(
            e_err_rows,
            dims=e_err_bundle.get("dims"),
            tstart=tstart,
            tend=tend,
        )

        if not (np.any(np.isfinite(b_t)) and np.any(np.isfinite(e_loc_t)) and np.any(np.isfinite(e_err_t))):
            summary_rows.append(
                {
                    "pulse": int(pulse),
                    "n_times": 0,
                    "mean_rmse": np.nan,
                    "mean_coverage": np.nan,
                    "mean_abs_resid_over_err": np.nan,
                    "status": "invalid_time_axes",
                }
            )
            continue

        tmin = max(float(np.nanmin(b_t)), float(np.nanmin(e_loc_t)), float(np.nanmin(e_err_t)))
        tmax = min(float(np.nanmax(b_t)), float(np.nanmax(e_loc_t)), float(np.nanmax(e_err_t)))
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax < tmin:
            summary_rows.append(
                {
                    "pulse": int(pulse),
                    "n_times": 0,
                    "mean_rmse": np.nan,
                    "mean_coverage": np.nan,
                    "mean_abs_resid_over_err": np.nan,
                    "status": "no_time_overlap",
                }
            )
            continue
        keep = np.isfinite(b_t) & (b_t >= tmin) & (b_t <= tmax)
        b_rows = b_rows[keep]
        b_t = b_t[keep]

        e_loc_rows = _interp_rows_in_time(e_loc_rows, e_loc_t, b_t)
        e_err_rows = _interp_rows_in_time(e_err_rows, e_err_t, b_t)

        n_t = int(min(b_rows.shape[0], e_loc_rows.shape[0], e_err_rows.shape[0]))
        if n_t <= 0:
            summary_rows.append(
                {
                    "pulse": int(pulse),
                    "n_times": 0,
                    "mean_rmse": np.nan,
                    "mean_coverage": np.nan,
                    "mean_abs_resid_over_err": np.nan,
                    "status": "empty_time_axis",
                }
            )
            continue

        b_rows = b_rows[:n_t]
        e_loc_rows = e_loc_rows[:n_t]
        e_err_rows = e_err_rows[:n_t]
        b_t = b_t[:n_t]

        pulse_rmses: list[float] = []
        pulse_covs: list[float] = []
        pulse_resid_over_err: list[float] = []

        plot_idx = _choose_time_indices(n_t=n_t, max_traces=max_plot_times_per_pulse)
        n_show = int(plot_idx.size)
        n_cols = 3
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n_show:]:
            ax.axis("off")

        x_e = np.linspace(0.0, 1.1, int(model.e_dim), dtype=np.float64)
        z_fixed = torch.from_numpy(rng.normal(size=(int(k_samples), int(model.latent_dim))).astype(np.float32))

        for t_idx in range(n_t):
            b_row = _resample_row_to_dim(b_rows[t_idx], int(model.b_dim))
            e_loc = _resample_row_to_dim(e_loc_rows[t_idx], int(model.e_dim))
            e_err = np.abs(_resample_row_to_dim(e_err_rows[t_idx], int(model.e_dim)))

            b_norm = (b_row - float(dataset.mu_b)) / max(float(dataset.sigma_b), 1e-12)
            b_t = torch.from_numpy(np.asarray(b_norm, dtype=np.float32)).unsqueeze(0)

            with torch.no_grad():
                b_rep = b_t.expand(int(k_samples), -1)
                e_samps = model.decode(b_rep, z_fixed)
                e_samps_un = (e_samps * float(dataset.sigma_eps) + float(dataset.mu_eps)).cpu().numpy()

            if enforce_nonnegative_output:
                e_samps_un = np.maximum(e_samps_un, 0.0)

            e_mean = np.mean(e_samps_un, axis=0)
            e_lo = np.percentile(e_samps_un, 2.5, axis=0)
            e_hi = np.percentile(e_samps_un, 97.5, axis=0)

            valid = np.isfinite(e_loc) & np.isfinite(e_mean)
            if int(np.sum(valid)) > 0:
                rmse = float(np.sqrt(np.mean((e_mean[valid] - e_loc[valid]) ** 2)))
                cov = float(np.mean((e_loc[valid] >= e_lo[valid]) & (e_loc[valid] <= e_hi[valid])))
                valid_err = valid & np.isfinite(e_err) & (e_err > 0)
                if int(np.sum(valid_err)) > 0:
                    resid_over_err = float(np.mean(np.abs(e_mean[valid_err] - e_loc[valid_err]) / e_err[valid_err]))
                else:
                    resid_over_err = np.nan
            else:
                rmse = np.nan
                cov = np.nan
                resid_over_err = np.nan

            pulse_rmses.append(rmse)
            pulse_covs.append(cov)
            pulse_resid_over_err.append(resid_over_err)
            metrics_rows.append(
                {
                    "pulse": int(pulse),
                    "t_idx": int(t_idx),
                    "t_s": float(b_t[t_idx]) if t_idx < b_t.size else np.nan,
                    "rmse_model_vs_t1d": rmse,
                    "coverage_t1d_in_model_95": cov,
                    "mean_abs_resid_over_t1d_err": resid_over_err,
                }
            )

            if t_idx in set(plot_idx.tolist()):
                ax = axes[list(plot_idx).index(t_idx)]
                ax.plot(x_e, e_mean, color="tab:blue", linewidth=2.0, label="Model mean")
                ax.fill_between(x_e, e_lo, e_hi, color="tab:blue", alpha=0.22, label="Model 95%")
                ax.plot(x_e, e_loc, color="black", linewidth=1.6, label="T1D EMIS_LOC")
                ax.fill_between(x_e, e_loc - e_err, e_loc + e_err, color="gray", alpha=0.20, label="T1D err")
                ax.set_title(f"t={float(b_t[t_idx]):.4f}s")
                ax.grid(alpha=0.25)

        axes[0].set_ylabel("emissivity")
        for ax in axes[max(0, n_show - n_cols):n_show]:
            ax.set_xlabel("rhop")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8)
        fig.suptitle(f"Pulse {pulse}: DV1->VAE vs T1D XY1 emissivity", y=1.02)
        fig.tight_layout()
        p_plot = out / f"pulse_{pulse}_model_vs_t1d_emissivity.png"
        fig.savefig(p_plot, dpi=170, bbox_inches="tight")
        plt.close(fig)
        pulse_plots.append(str(p_plot))

        summary_rows.append(
            {
                "pulse": int(pulse),
                "n_times": int(n_t),
                "mean_rmse": float(np.nanmean(np.asarray(pulse_rmses, dtype=np.float64))),
                "mean_coverage": float(np.nanmean(np.asarray(pulse_covs, dtype=np.float64))),
                "mean_abs_resid_over_err": float(
                    np.nanmean(np.asarray(pulse_resid_over_err, dtype=np.float64))
                ),
                "status": "ok",
            }
        )

    metrics_csv = out / "model_vs_t1d_metrics_by_time.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pulse",
                "t_idx",
                "t_s",
                "rmse_model_vs_t1d",
                "coverage_t1d_in_model_95",
                "mean_abs_resid_over_t1d_err",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary_csv = out / "model_vs_t1d_metrics_by_pulse.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pulse",
                "n_times",
                "mean_rmse",
                "mean_coverage",
                "mean_abs_resid_over_err",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "metrics_by_time_csv": str(metrics_csv),
        "metrics_by_pulse_csv": str(summary_csv),
        "pulse_plots": pulse_plots,
        "num_pulses_requested": int(len(pulses)),
        "num_pulses_compared": int(sum(1 for r in summary_rows if r["status"] == "ok")),
    }
