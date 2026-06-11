"""Prefect flow for lightweight synthetic inference-time benchmarking."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from prefect import flow, task
from matplotlib.patches import Patch
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset
from indica.workflows.jussiphd.los_bolometry_radiation import calculate_tomo_inversion


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parents[2] / "components" / "data" / "flow_data" / "multipulse_synthetic"
)
DEFAULT_MODEL_DIR = str(
    Path(__file__).resolve().parents[2] / "components" / "ml" / "flow_data" / "multipulse_synthetic"
)


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


def _pick_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_total <= n_pick:
        return np.arange(n_total, dtype=int)
    pos = np.linspace(0, n_total - 1, n_pick)
    return np.unique(np.round(pos).astype(int))


@task(name="run_timing_benchmark")
def run_timing_benchmark_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    transform: Any,
    n_samples: int,
    vae_k_samples: int,
    warmup_samples: int,
    seed: int,
) -> dict[str, Any]:
    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=None)
    model = _load_vae(model_path)

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("Dataset is empty; cannot benchmark.")
    indices = _pick_indices(n_total, int(n_samples))

    # Shared rhop and representative time coordinate for single-slice inversion call.
    rhop = np.linspace(0.0, 1.0, int(dataset.eps_slices.shape[1]), dtype=np.float32)
    eq_t_mid = 0.0
    try:
        if hasattr(transform, "equilibrium") and hasattr(transform.equilibrium, "t"):
            t_arr = np.asarray(transform.equilibrium.t, dtype=float).reshape(-1)
            t_arr = t_arr[np.isfinite(t_arr)]
            if t_arr.size > 0:
                eq_t_mid = float(0.5 * (t_arr.min() + t_arr.max()))
    except Exception:
        eq_t_mid = 0.0

    # Warm-up (not timed) to reduce one-off overhead bias.
    for _ in range(max(0, int(warmup_samples))):
        idx = int(indices[int(rng.integers(0, len(indices)))])
        e_norm, b_norm = dataset[idx]
        b_true = b_norm * dataset.sigma_b + dataset.mu_b
        brightness_single = DataArray(
            np.asarray(b_true, dtype=np.float32)[None, :],
            coords=[("t", np.asarray([eq_t_mid], dtype=np.float32)), ("channel", np.arange(b_true.shape[0]))],
        )
        try:
            _ = calculate_tomo_inversion(brightness_single, transform, rhop)
        except Exception:
            pass
        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(int(vae_k_samples), model.latent_dim)
            b_rep = b_t_norm.expand(int(vae_k_samples), -1)
            _ = model.decode(b_rep, z)

    rows: list[dict[str, Any]] = []
    for idx in indices:
        e_norm, b_norm = dataset[int(idx)]
        b_true = b_norm * dataset.sigma_b + dataset.mu_b
        brightness_single = DataArray(
            np.asarray(b_true, dtype=np.float32)[None, :],
            coords=[("t", np.asarray([eq_t_mid], dtype=np.float32)), ("channel", np.arange(b_true.shape[0]))],
        )

        naive_time = np.nan
        vae_time = np.nan
        naive_ok = True
        vae_ok = True
        naive_error = ""
        vae_error = ""

        t0 = time.perf_counter()
        try:
            _ = calculate_tomo_inversion(brightness_single, transform, rhop)
        except Exception as exc:  # pragma: no cover
            naive_ok = False
            naive_error = str(exc)
        naive_time = float(time.perf_counter() - t0)

        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                z = torch.randn(int(vae_k_samples), model.latent_dim)
                b_rep = b_t_norm.expand(int(vae_k_samples), -1)
                _ = model.decode(b_rep, z).mean(dim=0)
        except Exception as exc:  # pragma: no cover
            vae_ok = False
            vae_error = str(exc)
        vae_time = float(time.perf_counter() - t0)

        speedup = np.nan
        if naive_ok and vae_ok and vae_time > 0:
            speedup = float(naive_time / vae_time)

        rows.append(
            {
                "sample_idx": int(idx),
                "naive_time_s": naive_time,
                "vae_time_s": vae_time,
                "speedup_naive_over_vae": speedup,
                "naive_ok": bool(naive_ok),
                "vae_ok": bool(vae_ok),
                "naive_error": naive_error,
                "vae_error": vae_error,
            }
        )

    naive_vals = np.asarray([r["naive_time_s"] for r in rows if r["naive_ok"]], dtype=float)
    vae_vals = np.asarray([r["vae_time_s"] for r in rows if r["vae_ok"]], dtype=float)
    speed_vals = np.asarray(
        [r["speedup_naive_over_vae"] for r in rows if np.isfinite(r["speedup_naive_over_vae"])],
        dtype=float,
    )

    return {
        "num_samples_requested": int(n_samples),
        "num_samples_benchmarked": int(len(rows)),
        "num_naive_ok": int(np.sum([bool(r["naive_ok"]) for r in rows])),
        "num_vae_ok": int(np.sum([bool(r["vae_ok"]) for r in rows])),
        "naive_time_mean_s": float(np.mean(naive_vals)) if naive_vals.size else np.nan,
        "vae_time_mean_s": float(np.mean(vae_vals)) if vae_vals.size else np.nan,
        "naive_time_median_s": float(np.median(naive_vals)) if naive_vals.size else np.nan,
        "vae_time_median_s": float(np.median(vae_vals)) if vae_vals.size else np.nan,
        "speedup_mean": float(np.mean(speed_vals)) if speed_vals.size else np.nan,
        "speedup_median": float(np.median(speed_vals)) if speed_vals.size else np.nan,
        "rows": rows,
        "model_path": model_path,
        "b_path": b_path,
        "eps_path": eps_path,
    }


@task(name="save_timing_results")
def save_timing_results_task(
    benchmark: dict[str, Any],
    output_dir: str,
    csv_filename: str,
    plot_filename: str,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / csv_filename
    columns = [
        "sample_idx",
        "naive_time_s",
        "vae_time_s",
        "speedup_naive_over_vae",
        "naive_ok",
        "vae_ok",
        "naive_error",
        "vae_error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in benchmark["rows"]:
            writer.writerow({k: row.get(k, "") for k in columns})

    plot_path = out_dir / plot_filename
    rows = benchmark["rows"]
    safe_floor = 1e-8
    naive_t = np.asarray([r["naive_time_s"] for r in rows if r["naive_ok"]], dtype=float)
    vae_t = np.asarray([r["vae_time_s"] for r in rows if r["vae_ok"]], dtype=float)
    naive_t = np.clip(naive_t, safe_floor, None)
    vae_t = np.clip(vae_t, safe_floor, None)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    data = [naive_t, vae_t]
    labels = ["Naive inversion", "VAE inference"]
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 0.0},
    )
    box_colors = ["#4C78A8", "#F58518"]  # naive, vae
    for box, color in zip(bp["boxes"], box_colors):
        box.set_facecolor(color)
        box.set_alpha(0.55)

    # Keep summary consistent with boxplot: report median and IQR only.
    stats_lines: list[str] = []
    for x_pos, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        stats_lines.append(
            f"{labels[x_pos-1]}\n"
            f"  median={median:.3e}s\n"
            f"  IQR=[{q1:.3e}, {q3:.3e}]s"
        )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime [s] (log scale)")
    ax.set_title("Inference Time Comparison (Left=Naive, Right=VAE)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(
        handles=[
            Patch(facecolor=box_colors[0], alpha=0.55, label="Naive inversion (left)"),
            Patch(facecolor=box_colors[1], alpha=0.55, label="VAE inference (right)"),
        ],
        loc="upper right",
    )
    if len(naive_t) > 0 and len(vae_t) > 0:
        y_min = max(safe_floor, float(min(np.min(naive_t), np.min(vae_t))) * 0.6)
        y_max = float(max(np.max(naive_t), np.max(vae_t))) * 1.4
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)
    if stats_lines:
        # Put numeric summary inside axes to avoid expanding figure width.
        ax.text(
            0.02,
            0.02,
            "\n\n".join(stats_lines),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "timing_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"num_samples_benchmarked: {benchmark['num_samples_benchmarked']}\n")
        f.write(f"num_naive_ok: {benchmark['num_naive_ok']}\n")
        f.write(f"num_vae_ok: {benchmark['num_vae_ok']}\n")
        f.write(f"naive_time_mean_s: {benchmark['naive_time_mean_s']}\n")
        f.write(f"vae_time_mean_s: {benchmark['vae_time_mean_s']}\n")
        f.write(f"naive_time_median_s: {benchmark['naive_time_median_s']}\n")
        f.write(f"vae_time_median_s: {benchmark['vae_time_median_s']}\n")
        f.write(f"speedup_mean: {benchmark['speedup_mean']}\n")
        f.write(f"speedup_median: {benchmark['speedup_median']}\n")

    return {
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
        "summary_path": str(summary_path),
    }


@flow(name="benchmark_synthetic_inference_time")
def benchmark_synthetic_inference_time(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    b_path: str = str(Path(DEFAULT_DATA_DIR) / "b_slices_multipulse_synthetic.csv"),
    eps_path: str = str(Path(DEFAULT_DATA_DIR) / "eps_slices_multipulse_synthetic.csv"),
    model_path: str = str(Path(DEFAULT_MODEL_DIR) / "vae_multipulse_synthetic.pt"),
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_samples: int = 100,
    vae_k_samples: int = 20,
    warmup_samples: int = 20,
    seed: int = 0,
    csv_filename: str = "timing_naive_vs_vae.csv",
    plot_filename: str = "timing_naive_vs_vae_log.png",
) -> dict[str, Any]:
    """Benchmark naive inversion vs VAE inference time on synthetic samples."""
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)

    benchmark = run_timing_benchmark_task(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        transform=transform,
        n_samples=int(n_samples),
        vae_k_samples=int(vae_k_samples),
        warmup_samples=int(warmup_samples),
        seed=int(seed),
    )
    outputs = save_timing_results_task(
        benchmark=benchmark,
        output_dir=output_dir,
        csv_filename=csv_filename,
        plot_filename=plot_filename,
    )

    return {
        "benchmark": benchmark,
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = benchmark_synthetic_inference_time()
    print("Timing benchmark complete")
    print(f"Plot: {result['outputs']['plot_path']}")
