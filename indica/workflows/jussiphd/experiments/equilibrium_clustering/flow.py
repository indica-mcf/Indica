"""Prefect flow: cluster real equilibrium snapshots from filtered Te/Ne pulse set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from prefect import flow, task

from indica.workflows.jussiphd.components.data.equilibrium_snapshot_dataset import (
    build_and_save_equilibrium_boundary_dataset,
    load_non_outlier_pulses_from_report,
)
from indica.workflows.jussiphd.components.evaluation.eps_clustering import (
    kmeans_cluster_eps_profiles,
)
from indica.workflows.jussiphd.components.evaluation.equilibrium_clustering import (
    save_equilibrium_clustering_outputs,
)


DEFAULT_OUTLIER_REPORT_PATH = str(
    Path(__file__).resolve().parents[1]
    / "real_tene_clustering"
    / "outputs"
    / "original_data"
    / "te_ne_outlier_report.csv"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


@task(name="load_non_outlier_pulses")
def load_non_outlier_pulses_task(
    outlier_report_path: str,
    deduplicate_pulses: bool,
) -> list[int]:
    return load_non_outlier_pulses_from_report(
        outlier_report_path=outlier_report_path,
        deduplicate=deduplicate_pulses,
    )


@task(name="build_equilibrium_boundary_snapshot_dataset")
def build_equilibrium_boundary_snapshot_dataset_task(
    pulses: Sequence[int],
    output_dir: str,
    features_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    tstart: float,
    tend: float,
    dt: float,
    n_timepoints_per_equilibrium: int,
    n_boundary_points: int,
    verbose: bool,
    skip_failed_pulses: bool,
) -> dict[str, Any]:
    return build_and_save_equilibrium_boundary_dataset(
        pulses=pulses,
        output_dir=output_dir,
        features_filename=features_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        tstart=tstart,
        tend=tend,
        dt=dt,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        n_boundary_points=n_boundary_points,
        verbose=verbose,
        skip_failed_pulses=skip_failed_pulses,
    )


@task(name="cluster_equilibrium_boundary_features")
def cluster_equilibrium_boundary_features_task(
    features_path: str,
    n_clusters: int,
    max_iter: int,
    tol: float,
    seed: int,
    normalize_per_profile: bool,
) -> dict[str, Any]:
    return kmeans_cluster_eps_profiles(
        eps_path=features_path,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        normalize_per_profile=normalize_per_profile,
    )


@task(name="save_equilibrium_cluster_outputs")
def save_equilibrium_cluster_outputs_task(
    clustering: dict[str, Any],
    meta_path: str,
    output_dir: str,
    assignments_filename: str,
    gallery_filename: str,
    centers_filename: str,
    summary_filename: str,
    max_boundaries_per_cluster: int,
    seed: int,
    n_boundary_points: int,
) -> dict[str, str]:
    return save_equilibrium_clustering_outputs(
        clustering=clustering,
        meta_path=meta_path,
        output_dir=output_dir,
        assignments_filename=assignments_filename,
        gallery_filename=gallery_filename,
        centers_filename=centers_filename,
        summary_filename=summary_filename,
        max_boundaries_per_cluster=max_boundaries_per_cluster,
        seed=seed,
        n_boundary_points=n_boundary_points,
    )


@flow(name="equilibrium_clustering")
def cluster_real_equilibrium_boundaries(
    outlier_report_path: str = DEFAULT_OUTLIER_REPORT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pulses: Sequence[int] | None = None,
    deduplicate_pulses: bool = True,
    generate_new_data: bool = True,
    features_filename: str = "equilibrium_boundary_features.csv",
    meta_filename: str = "equilibrium_boundary_meta.csv",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    n_timepoints_per_equilibrium: int = 8,
    n_boundary_points: int = 128,
    verbose: bool = False,
    skip_failed_pulses: bool = True,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    normalize_per_profile: bool = False,
    assignments_filename: str = "equilibrium_cluster_assignments.csv",
    gallery_filename: str = "equilibrium_clusters_gallery.png",
    centers_filename: str = "equilibrium_cluster_centers_overlay.png",
    summary_filename: str = "equilibrium_cluster_summary.csv",
    max_boundaries_per_cluster: int = 100,
) -> dict[str, Any]:
    """Cluster boundary snapshots from non-outlier real Te/Ne pulses."""
    pulse_list = [int(p) for p in (pulses or [])]
    if not pulse_list:
        pulse_list = load_non_outlier_pulses_task(
            outlier_report_path=outlier_report_path,
            deduplicate_pulses=deduplicate_pulses,
        )
    if not pulse_list:
        raise RuntimeError("No pulses available for equilibrium clustering.")

    dataset = build_equilibrium_boundary_snapshot_dataset_task(
        pulses=pulse_list,
        output_dir=output_dir,
        features_filename=features_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        tstart=tstart,
        tend=tend,
        dt=dt,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        n_boundary_points=n_boundary_points,
        verbose=verbose,
        skip_failed_pulses=skip_failed_pulses,
    )

    clustering = cluster_equilibrium_boundary_features_task(
        features_path=dataset["features_path"],
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        normalize_per_profile=normalize_per_profile,
    )

    outputs = save_equilibrium_cluster_outputs_task(
        clustering=clustering,
        meta_path=dataset["meta_path"],
        output_dir=output_dir,
        assignments_filename=assignments_filename,
        gallery_filename=gallery_filename,
        centers_filename=centers_filename,
        summary_filename=summary_filename,
        max_boundaries_per_cluster=max_boundaries_per_cluster,
        seed=seed,
        n_boundary_points=n_boundary_points,
    )

    return {
        "outlier_report_path": outlier_report_path,
        "num_pulses_requested": int(len(pulse_list)),
        "dataset": dataset,
        "n_clusters": int(clustering["n_clusters"]),
        "cluster_counts": [int(x) for x in clustering["counts"]],
        "normalize_per_profile": bool(normalize_per_profile),
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = cluster_real_equilibrium_boundaries()
    print("Equilibrium clustering complete")
    print(
        f"Pulses requested: {result['num_pulses_requested']} | "
        f"Snapshots: {result['dataset']['num_snapshots']}"
    )
    print(f"Outputs: {result['outputs']}")

