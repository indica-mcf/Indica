"""Equilibrium-boundary clustering output helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_csv_2d(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float64)


def _split_boundary_row(row: np.ndarray, n_boundary_points: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(n_boundary_points)
    flat = np.asarray(row, dtype=np.float64).reshape(-1)
    if flat.size != 2 * n:
        raise ValueError(f"Boundary row length {flat.size} does not match 2*n_boundary_points={2*n}.")
    return flat[:n], flat[n:]


def save_equilibrium_clustering_outputs(
    clustering: dict[str, Any],
    meta_path: str,
    output_dir: str,
    assignments_filename: str = "equilibrium_cluster_assignments.csv",
    gallery_filename: str = "equilibrium_clusters_gallery.png",
    centers_filename: str = "equilibrium_cluster_centers_overlay.png",
    summary_filename: str = "equilibrium_cluster_summary.csv",
    max_boundaries_per_cluster: int = 100,
    seed: int = 0,
    n_boundary_points: int | None = None,
) -> dict[str, str]:
    """Save assignments and R-Z boundary plots for clustered equilibrium snapshots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(clustering["labels"], dtype=int).reshape(-1)
    counts = np.asarray(clustering["counts"], dtype=int).reshape(-1)
    centers = np.asarray(clustering["centers_original"], dtype=np.float64)
    profiles = _load_csv_2d(clustering["data_path"])
    k = int(clustering["n_clusters"])

    if profiles.shape[0] != labels.size:
        raise ValueError("Mismatch between clustered labels and profile rows.")

    if n_boundary_points is None:
        if profiles.shape[1] % 2 != 0:
            raise ValueError("Boundary feature dimension must be even (R and Z concatenation).")
        n_boundary_points = int(profiles.shape[1] // 2)
    n_boundary_points = int(n_boundary_points)

    with Path(meta_path).open(newline="") as f:
        meta_rows = list(csv.DictReader(f))
    if len(meta_rows) != int(profiles.shape[0]):
        raise ValueError(
            "Meta rows do not match number of clustered snapshots: "
            f"meta={len(meta_rows)} profiles={int(profiles.shape[0])}"
        )

    assignments_path = out / assignments_filename
    meta_fields = [kname for kname in (meta_rows[0].keys() if meta_rows else []) if kname != "sample_idx"]
    with assignments_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "cluster", *meta_fields])
        writer.writeheader()
        for i, lab in enumerate(labels):
            rec: dict[str, Any] = {"sample_idx": int(i), "cluster": int(lab)}
            for key in meta_fields:
                rec[key] = meta_rows[i].get(key)
            writer.writerow(rec)

    summary_path = out / summary_filename
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster", "count"])
        writer.writeheader()
        for c in range(k):
            writer.writerow({"cluster": int(c), "count": int(counts[c])})

    rng = np.random.default_rng(int(seed))
    colors = plt.cm.tab20(np.linspace(0, 1, max(2, k)))

    n_cols = int(min(4, max(1, k)))
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.1 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for c in range(k):
        ax = axes[c]
        members = np.where(labels == c)[0]
        if members.size > int(max_boundaries_per_cluster):
            members = rng.choice(members, size=int(max_boundaries_per_cluster), replace=False)
        for idx in members:
            rr, zz = _split_boundary_row(profiles[int(idx)], n_boundary_points=n_boundary_points)
            ax.plot(rr, zz, color=colors[c % len(colors)], alpha=0.14, linewidth=0.9)
        cr, cz = _split_boundary_row(centers[c], n_boundary_points=n_boundary_points)
        ax.plot(cr, cz, color="black", linewidth=2.0)
        ax.set_title(f"Cluster {c} (n={int(counts[c])})")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle("Equilibrium boundary clusters", y=1.02)
    fig.tight_layout()
    gallery_path = out / gallery_filename
    fig.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    for c in range(k):
        cr, cz = _split_boundary_row(centers[c], n_boundary_points=n_boundary_points)
        ax.plot(
            cr,
            cz,
            linewidth=2.1,
            color=colors[c % len(colors)],
            label=f"Cluster {c} (n={int(counts[c])})",
        )
    ax.set_title("Equilibrium boundary cluster centers")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    centers_path = out / centers_filename
    fig.savefig(centers_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "assignments_csv": str(assignments_path),
        "summary_csv": str(summary_path),
        "clusters_gallery_plot": str(gallery_path),
        "cluster_centers_plot": str(centers_path),
    }

