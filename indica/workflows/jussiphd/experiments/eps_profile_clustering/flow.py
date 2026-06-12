"""Prefect flow for clustering synthetic emissivity profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.components.evaluation.eps_clustering import (
    kmeans_cluster_eps_profiles,
    save_eps_clustering_outputs,
)


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_EPS_PATH = str(
    Path(__file__).resolve().parents[2]
    / "components"
    / "data"
    / "flow_data"
    / "multipulse_synthetic"
    / "eps_slices_multipulse_synthetic.csv"
)


@task(name="cluster_eps_profiles")
def cluster_eps_profiles_task(
    eps_path: str,
    n_clusters: int,
    max_iter: int,
    tol: float,
    seed: int,
    normalize_per_profile: bool,
) -> dict[str, Any]:
    return kmeans_cluster_eps_profiles(
        eps_path=eps_path,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        normalize_per_profile=normalize_per_profile,
    )


@task(name="save_eps_cluster_outputs")
def save_eps_cluster_outputs_task(
    clustering: dict[str, Any],
    output_dir: str,
    assignments_filename: str,
    gallery_filename: str,
    centers_filename: str,
    max_profiles_per_cluster: int,
    seed: int,
) -> dict[str, str]:
    return save_eps_clustering_outputs(
        clustering=clustering,
        output_dir=output_dir,
        assignments_filename=assignments_filename,
        gallery_filename=gallery_filename,
        centers_filename=centers_filename,
        max_profiles_per_cluster=max_profiles_per_cluster,
        seed=seed,
    )


@flow(name="cluster_synthetic_eps_profiles")
def cluster_synthetic_eps_profiles(
    eps_path: str = DEFAULT_EPS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_clusters: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    normalize_per_profile: bool = True,
    assignments_filename: str = "eps_cluster_assignments.csv",
    gallery_filename: str = "eps_clusters_gallery.png",
    centers_filename: str = "eps_cluster_centers_overlay.png",
    max_profiles_per_cluster: int = 60,
) -> dict[str, Any]:
    """Cluster synthetic eps profiles and save cluster-size annotated plots."""
    clustering = cluster_eps_profiles_task(
        eps_path=eps_path,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        normalize_per_profile=normalize_per_profile,
    )
    outputs = save_eps_cluster_outputs_task(
        clustering=clustering,
        output_dir=output_dir,
        assignments_filename=assignments_filename,
        gallery_filename=gallery_filename,
        centers_filename=centers_filename,
        max_profiles_per_cluster=max_profiles_per_cluster,
        seed=seed,
    )
    return {
        "eps_path": eps_path,
        "n_clusters": int(clustering["n_clusters"]),
        "counts": [int(x) for x in clustering["counts"]],
        "normalize_per_profile": bool(normalize_per_profile),
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = cluster_synthetic_eps_profiles()
    print("EPS profile clustering complete")
    print(f"Outputs: {result['outputs']}")
