"""Visualisation helpers for sampled anchor curves from cluster Gaussians."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _shade_for_member(
    base_color: tuple[float, float, float, float],
    member_idx: int,
    n_members: int,
) -> tuple[float, float, float, float]:
    """Create visible per-member shades within a cluster color."""
    if n_members <= 1:
        return base_color
    rgb = np.asarray(base_color[:3], dtype=np.float64)
    white = np.ones(3, dtype=np.float64)
    frac = 0.15 + 0.40 * (member_idx / max(1, n_members - 1))
    mixed = (1.0 - frac) * rgb + frac * white
    return (float(mixed[0]), float(mixed[1]), float(mixed[2]), float(base_color[3]))


def plot_anchor_cluster_gaussian_samples(
    gaussian_params_npz: str,
    output_dir: str,
    plot_filename: str,
    x_values: list[float],
    title: str,
    y_label: str,
    samples_per_cluster: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    params = np.load(gaussian_params_npz)
    clusters = np.asarray(params["clusters"], dtype=int).reshape(-1)
    means = np.asarray(params["means"], dtype=np.float64)
    covs = np.asarray(params["covariances"], dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    if means.ndim != 2 or covs.ndim != 3:
        raise ValueError("Gaussian params must provide means[K,D] and covariances[K,D,D].")
    if means.shape[1] != x.size:
        raise ValueError(f"x_values length {x.size} does not match anchor dim {means.shape[1]}.")

    k = int(clusters.size)
    n_draw = int(max(1, samples_per_cluster))
    rng = np.random.default_rng(int(seed))
    cmap = plt.cm.get_cmap("tab10", max(2, k))

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    total_draws = 0
    for i, cluster_id in enumerate(clusters):
        mu = means[i]
        cov = covs[i]
        try:
            draws = rng.multivariate_normal(mu, cov, size=n_draw, check_valid="ignore")
        except Exception:
            evals, evecs = np.linalg.eigh(cov)
            evals = np.clip(evals, 1e-12, None)
            cov_psd = (evecs * evals) @ evecs.T
            draws = rng.multivariate_normal(mu, cov_psd, size=n_draw, check_valid="ignore")

        base = cmap(i)
        for j in range(n_draw):
            color = _shade_for_member(base, j, n_draw)
            label = f"C{int(cluster_id)} samples" if j == 0 else None
            ax.plot(x, draws[j], color=color, linewidth=1.6, alpha=0.98, label=label)
            total_draws += 1

    ax.set_title(title)
    ax.set_xlabel("rhop knot")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_path = out / plot_filename
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_path": str(plot_path),
        "n_clusters": k,
        "samples_per_cluster": n_draw,
        "n_total_draws": int(total_draws),
    }

