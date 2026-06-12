"""Clustering utilities for emissivity-profile datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_csv_2d(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float32)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    mu = np.mean(arr, axis=1, keepdims=True)
    sigma = np.std(arr, axis=1, keepdims=True)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    return (arr - mu) / sigma


def kmeans_cluster_eps_profiles(
    eps_path: str,
    n_clusters: int = 6,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    normalize_per_profile: bool = True,
) -> dict[str, Any]:
    """Cluster emissivity profiles with a lightweight NumPy k-means."""
    eps = _load_csv_2d(eps_path)
    n_samples = int(eps.shape[0])
    if n_samples == 0:
        raise ValueError("Empty emissivity dataset.")
    k = int(np.clip(int(n_clusters), 1, n_samples))

    x = _normalize_rows(eps) if normalize_per_profile else eps.copy()
    rng = np.random.default_rng(int(seed))

    init_idx = rng.choice(n_samples, size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(int(max_iter)):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = new_labels == c
            if np.any(mask):
                new_centers[c] = np.mean(x[mask], axis=0)
            else:
                new_centers[c] = x[int(rng.integers(0, n_samples))]
        shift = float(np.sqrt(np.mean((new_centers - centers) ** 2)))
        labels = new_labels
        centers = new_centers
        if shift < float(tol):
            break

    counts = np.asarray([(labels == c).sum() for c in range(k)], dtype=int)
    centers_orig = np.zeros((k, eps.shape[1]), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            centers_orig[c] = np.mean(eps[mask], axis=0)
        else:
            centers_orig[c] = np.zeros(eps.shape[1], dtype=np.float32)

    return {
        "eps_path": eps_path,
        "n_samples": n_samples,
        "n_clusters": int(k),
        "labels": labels,
        "counts": counts,
        "centers_normalized": centers,
        "centers_original": centers_orig,
        "normalize_per_profile": bool(normalize_per_profile),
    }


def save_eps_clustering_outputs(
    clustering: dict[str, Any],
    output_dir: str,
    assignments_filename: str = "eps_cluster_assignments.csv",
    gallery_filename: str = "eps_clusters_gallery.png",
    centers_filename: str = "eps_cluster_centers_overlay.png",
    max_profiles_per_cluster: int = 60,
    seed: int = 0,
) -> dict[str, str]:
    """Save cluster assignment table and two profile-cluster visualisations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(clustering["labels"], dtype=int)
    counts = np.asarray(clustering["counts"], dtype=int)
    centers = np.asarray(clustering["centers_original"], dtype=np.float32)
    eps = _load_csv_2d(clustering["eps_path"])
    k = int(clustering["n_clusters"])

    assignments_path = out_dir / assignments_filename
    with assignments_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "cluster"])
        writer.writeheader()
        for i, lab in enumerate(labels):
            writer.writerow({"sample_idx": int(i), "cluster": int(lab)})

    # Gallery: per-cluster sample curves + center curve + count label.
    n_cols = 2
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.linspace(0.0, 1.0, eps.shape[1], dtype=np.float32)
    rng = np.random.default_rng(int(seed))

    for c in range(k):
        ax = axes[c]
        members = np.where(labels == c)[0]
        if len(members) > 0:
            if len(members) > int(max_profiles_per_cluster):
                members = rng.choice(members, size=int(max_profiles_per_cluster), replace=False)
            for idx in members:
                ax.plot(x, eps[int(idx)], color="tab:blue", alpha=0.15, linewidth=1.0)
        ax.plot(x, centers[c], color="black", linewidth=2.2, label="cluster center")
        ax.set_title(f"Cluster {c} (n={int(counts[c])})")
        ax.set_xlabel("normalized position (0..1)")
        ax.set_ylabel("emissivity")
        ax.grid(alpha=0.25)

    for ax in axes[k:]:
        ax.axis("off")

    fig.suptitle("Emissivity profile clusters", y=1.02)
    fig.tight_layout()
    gallery_path = out_dir / gallery_filename
    fig.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Overlay: only cluster centers with count legend.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(2, k)))
    for c in range(k):
        ax.plot(
            x,
            centers[c],
            linewidth=2.0,
            color=colors[c % len(colors)],
            label=f"Cluster {c} (n={int(counts[c])})",
        )
    ax.set_title("Cluster center profiles")
    ax.set_xlabel("normalized position (0..1)")
    ax.set_ylabel("emissivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    centers_path = out_dir / centers_filename
    fig.savefig(centers_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "assignments_csv": str(assignments_path),
        "clusters_gallery_plot": str(gallery_path),
        "cluster_centers_plot": str(centers_path),
    }
