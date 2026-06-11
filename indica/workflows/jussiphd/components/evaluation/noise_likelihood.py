"""Noise calibration utilities for matching synthetic and real datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _load_csv_array(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float32)


def add_poisson_noise_with_counts(
    values: np.ndarray,
    count_level: float,
    scale_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply Poisson noise using an effective count level on scaled non-negative values."""
    if count_level <= 0:
        raise ValueError("count_level must be > 0.")
    if scale_value <= 0:
        raise ValueError("scale_value must be > 0.")

    lam = np.clip(values, a_min=0.0, a_max=None).astype(np.float64) / float(scale_value)
    lam = lam * float(count_level)
    noisy_counts = rng.poisson(lam=lam)
    return (noisy_counts.astype(np.float32) / float(count_level)) * float(scale_value)


def _histogram_log_likelihood(
    train_values: np.ndarray,
    observed_values: np.ndarray,
    bins: int = 256,
    floor_prob: float = 1e-12,
) -> float:
    """Estimate average log-likelihood of observed values under a histogram density."""
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed_values, dtype=np.float64).reshape(-1)

    train = train[np.isfinite(train)]
    obs = obs[np.isfinite(obs)]
    if train.size == 0 or obs.size == 0:
        raise ValueError("Cannot compute likelihood with empty/non-finite arrays.")

    lo = float(min(train.min(), obs.min()))
    hi = float(max(train.max(), obs.max()))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite value range for histogram likelihood.")
    if hi <= lo:
        return 0.0

    pad = 0.01 * (hi - lo)
    lo -= pad
    hi += pad

    hist, edges = np.histogram(train, bins=int(bins), range=(lo, hi), density=True)
    widths = np.diff(edges)
    bin_prob = np.clip(hist * widths, floor_prob, None)

    idx = np.searchsorted(edges, obs, side="right") - 1
    idx = np.clip(idx, 0, len(bin_prob) - 1)
    probs = np.clip(bin_prob[idx], floor_prob, None)

    return float(np.mean(np.log(probs)))


def evaluate_noise_levels_against_real(
    synthetic_b_path: str,
    synthetic_eps_path: str,
    real_b_path: str,
    real_eps_path: str,
    count_levels: Sequence[float],
    seed: int | None = 0,
    bins: int = 256,
    b_scale_percentile: float = 99.0,
    eps_scale_percentile: float = 99.0,
) -> dict[str, Any]:
    """Sweep Poisson noise levels and score synthetic-vs-real likelihoods (emissivity only)."""
    syn_b = _load_csv_array(synthetic_b_path)
    syn_eps = _load_csv_array(synthetic_eps_path)
    real_b = _load_csv_array(real_b_path)
    real_eps = _load_csv_array(real_eps_path)

    rng = np.random.default_rng(seed)

    b_scale = float(np.percentile(np.clip(syn_b, a_min=0.0, a_max=None), b_scale_percentile))
    eps_scale = float(np.percentile(np.clip(syn_eps, a_min=0.0, a_max=None), eps_scale_percentile))
    b_scale = b_scale if b_scale > 0 else 1.0
    eps_scale = eps_scale if eps_scale > 0 else 1.0

    rows: list[dict[str, float]] = []
    for count in count_levels:
        c = float(count)
        noisy_eps = add_poisson_noise_with_counts(syn_eps, c, eps_scale, rng)

        # We calibrate against emissivity only; brightness is intentionally not noised/scored here.
        ll_b = float("nan")
        ll_eps = _histogram_log_likelihood(noisy_eps, real_eps, bins=bins)
        ll_total = ll_eps

        rows.append(
            {
                "count_level": c,
                "loglik_brightness": ll_b,
                "loglik_emissivity": ll_eps,
                "loglik_total": ll_total,
            }
        )

    if not rows:
        raise ValueError("No count levels provided.")

    best = max(rows, key=lambda r: r["loglik_total"])

    return {
        "synthetic_shapes": {"b": tuple(syn_b.shape), "eps": tuple(syn_eps.shape)},
        "real_shapes": {"b": tuple(real_b.shape), "eps": tuple(real_eps.shape)},
        "count_levels": [float(c) for c in count_levels],
        "scale_values": {"b_scale": b_scale, "eps_scale": eps_scale},
        "rows": rows,
        "best": best,
    }


def save_noise_likelihood_outputs(
    output_dir: str,
    run_id: str,
    result: dict[str, Any],
) -> dict[str, str]:
    """Write CSV/JSON summaries and a likelihood-vs-noise plot."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = result.get("rows", [])
    if not rows:
        raise ValueError("Result has no rows to write.")

    csv_path = out_dir / f"noise_likelihood_{run_id}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["count_level", "loglik_brightness", "loglik_emissivity", "loglik_total"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = out_dir / f"noise_likelihood_{run_id}.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    counts = np.asarray([row["count_level"] for row in rows], dtype=float)
    ll_b = np.asarray([row["loglik_brightness"] for row in rows], dtype=float)
    ll_eps = np.asarray([row["loglik_emissivity"] for row in rows], dtype=float)
    ll_tot = np.asarray([row["loglik_total"] for row in rows], dtype=float)
    best_idx = int(np.argmax(ll_tot))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(counts, ll_eps, marker="o", linewidth=2.0, label="emissivity log-likelihood")
    ax.scatter([counts[best_idx]], [ll_eps[best_idx]], color="black", zorder=5, label="best")
    ax.set_xscale("log")
    ax.set_xlabel("Poisson count level (effective)")
    ax.set_ylabel("Average log-likelihood")
    ax.set_title("Emissivity-only likelihood across Poisson noise levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / f"noise_likelihood_{run_id}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_path": str(plot_path),
    }
