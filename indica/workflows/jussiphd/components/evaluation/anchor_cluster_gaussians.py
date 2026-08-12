"""Gaussian estimation utilities for clustered anchor vectors."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def estimate_anchor_cluster_gaussians(
    anchor_path: str,
    assignments_csv: str,
    output_dir: str,
    prefix: str,
    cov_regularization: float = 1e-8,
) -> dict[str, Any]:
    anchors = np.loadtxt(anchor_path, delimiter=",", dtype=np.float64)
    if anchors.ndim == 1:
        anchors = anchors[None, :]
    n_samples, n_dim = int(anchors.shape[0]), int(anchors.shape[1])

    assignment_rows: list[tuple[int, int]] = []
    with Path(assignments_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignment_rows.append((int(row["sample_idx"]), int(row["cluster"])))

    if len(assignment_rows) != n_samples:
        raise ValueError(
            f"Assignment/sample mismatch: {len(assignment_rows)} assignments for {n_samples} anchors."
        )

    idx = np.asarray([r[0] for r in assignment_rows], dtype=int)
    lab = np.asarray([r[1] for r in assignment_rows], dtype=int)
    if np.any(idx < 0) or np.any(idx >= n_samples):
        raise ValueError("Cluster assignment sample_idx out of range.")

    order = np.argsort(idx)
    idx_sorted = idx[order]
    lab_sorted = lab[order]
    if not np.array_equal(idx_sorted, np.arange(n_samples, dtype=int)):
        raise ValueError("Cluster assignments must contain each sample_idx exactly once.")

    clusters = np.unique(lab_sorted)
    means: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    counts: list[int] = []

    for c in clusters:
        members = anchors[lab_sorted == c]
        n = int(members.shape[0])
        counts.append(n)
        mu = np.mean(members, axis=0, dtype=np.float64)
        if n >= 2:
            cov = np.cov(members, rowvar=False, bias=False).astype(np.float64)
        else:
            cov = np.zeros((n_dim, n_dim), dtype=np.float64)
        cov = cov + np.eye(n_dim, dtype=np.float64) * float(cov_regularization)
        means.append(mu)
        covs.append(cov)

    means_arr = np.asarray(means, dtype=np.float64)
    covs_arr = np.asarray(covs, dtype=np.float64)
    counts_arr = np.asarray(counts, dtype=int)
    clusters_arr = np.asarray(clusters, dtype=int)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{prefix}_cluster_gaussian_params.npz"
    summary_path = out / f"{prefix}_cluster_gaussian_summary.csv"
    np.savez_compressed(
        npz_path,
        clusters=clusters_arr,
        counts=counts_arr,
        means=means_arr,
        covariances=covs_arr,
        anchor_dim=np.asarray([n_dim], dtype=int),
        cov_regularization=np.asarray([float(cov_regularization)], dtype=np.float64),
    )

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "count", "mean_norm", "trace_cov"])
        for i, c in enumerate(clusters_arr):
            writer.writerow(
                [
                    int(c),
                    int(counts_arr[i]),
                    float(np.linalg.norm(means_arr[i])),
                    float(np.trace(covs_arr[i])),
                ]
            )

    return {
        "anchor_path": anchor_path,
        "assignments_csv": assignments_csv,
        "n_samples": n_samples,
        "anchor_dim": n_dim,
        "n_clusters": int(clusters_arr.size),
        "clusters": [int(c) for c in clusters_arr.tolist()],
        "counts": [int(c) for c in counts_arr.tolist()],
        "cov_regularization": float(cov_regularization),
        "params_npz": str(npz_path),
        "summary_csv": str(summary_path),
    }

