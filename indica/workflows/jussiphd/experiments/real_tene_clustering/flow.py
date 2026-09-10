"""Prefect flow to read and visualize real ST40 Te/Ne profiles by pulse."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from prefect import flow, task

from indica.workflows.jussiphd.components.data.real_brightness_dataset_generation import (
    generate_and_save_real_multipulse_brightness_dataset,
)
from indica.workflows.jussiphd.components.data.equilibrium_snapshot_dataset import (
    load_non_outlier_pulses_from_report,
)
from indica.workflows.jussiphd.components.evaluation import (
    estimate_anchor_cluster_gaussians,
    kmeans_cluster_eps_profiles,
    save_profile_clustering_outputs,
)
from indica.workflows.jussiphd.components.preprocessing import (
    align_filter_plot_and_save_te_ne_profiles,
    fit_save_and_plot_te_ne_anchor_space,
    load_monospline_anchor_spec,
)
from indica.workflows.jussiphd.components.visualisations import (
    plot_anchor_cluster_gaussian_samples,
)

TS_NE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:NE"
TS_TE_NODE = r"\ST40::TOP.TS.BEST.PROFILES:TE"
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_TSTART = 0.04
DEFAULT_TEND = 0.15
DEFAULT_FILTERED_PULSE_REPORT = str(
    Path(__file__).resolve().parent / "outputs" / "original_data" / "te_ne_outlier_report.csv"
)

@task(name="build_real_node_profile_dataset")
def build_real_node_profile_dataset_task(
    pulses: list[int],
    node: str | None,
    ppts_profile_key: str | None,
    output_dir: str,
    profile_filename: str,
    meta_filename: str,
    tstart: float,
    tend: float,
    dt: float,
    read_verbose: bool,
    min_finite_fraction: float,
    min_nonzero_fraction: float,
    canonicalize_profile_coordinate: bool,
    canonicalize_profile_coordinate_mode: str,
    use_source_coordinate_grid: bool,
    source_coordinate_min: float,
    source_coordinate_max: float | None,
) -> dict[str, Any]:
    return generate_and_save_real_multipulse_brightness_dataset(
        pulses=pulses,
        instrument="blom_rz1",
        tstart=tstart,
        tend=tend,
        dt=dt,
        output_dir=output_dir,
        b_filename=profile_filename,
        meta_filename=meta_filename,
        use_all_timepoints=False,
        node=node,
        ppts_profile_key=ppts_profile_key,
        generate_new_data=True,
        verbose=read_verbose,
        apply_basic_quality_filter=True,
        min_finite_fraction=min_finite_fraction,
        min_nonzero_fraction=min_nonzero_fraction,
        nonzero_threshold=0.0,
        require_plasma_summary=True,
        canonicalize_profile_coordinate=canonicalize_profile_coordinate,
        canonicalize_profile_coordinate_mode=canonicalize_profile_coordinate_mode,
        use_source_coordinate_grid=use_source_coordinate_grid,
        source_coordinate_min=source_coordinate_min,
        source_coordinate_max=source_coordinate_max,
    )


@task(name="align_and_plot_te_ne_profiles")
def align_and_plot_te_ne_profiles_task(
    ne_dataset: dict[str, Any],
    te_dataset: dict[str, Any],
    output_dir: str,
    plot_filename: str,
    ne_aligned_filename: str,
    te_aligned_filename: str,
    matched_meta_filename: str,
    outlier_report_filename: str,
    apply_outlier_filter: bool,
    outlier_point_z_threshold: float,
    outlier_extreme_point_z_threshold: float,
    outlier_min_bad_points: int,
    outlier_bad_fraction_threshold: float,
) -> dict[str, Any]:
    return align_filter_plot_and_save_te_ne_profiles(
        ne_dataset=ne_dataset,
        te_dataset=te_dataset,
        output_dir=output_dir,
        plot_filename=plot_filename,
        ne_aligned_filename=ne_aligned_filename,
        te_aligned_filename=te_aligned_filename,
        matched_meta_filename=matched_meta_filename,
        outlier_report_filename=outlier_report_filename,
        apply_outlier_filter=apply_outlier_filter,
        outlier_point_z_threshold=outlier_point_z_threshold,
        outlier_extreme_point_z_threshold=outlier_extreme_point_z_threshold,
        outlier_min_bad_points=outlier_min_bad_points,
        outlier_bad_fraction_threshold=outlier_bad_fraction_threshold,
    )


@task(name="fit_te_ne_to_spline_anchor_space")
def fit_te_ne_to_spline_anchor_space_task(
    ne_aligned_path: str,
    te_aligned_path: str,
    matched_meta_path: str,
    output_dir: str,
    config_name: str,
    ne_profile_name: str,
    te_profile_name: str,
    ne_anchor_filename: str,
    te_anchor_filename: str,
    fit_summary_filename: str,
    fit_spec_filename: str,
    ne_anchor_plot_filename: str,
    te_anchor_plot_filename: str,
    ne_middle_knot_plot_filename: str,
    te_middle_knot_plot_filename: str,
    ne_fixed_start: float | None,
    ne_fixed_end: float | None,
    te_fixed_start: float | None,
    te_fixed_end: float | None,
    ne_start_min: float | None,
    ne_start_max: float | None,
    te_start_min: float | None,
    te_start_max: float | None,
) -> dict[str, Any]:
    return fit_save_and_plot_te_ne_anchor_space(
        ne_aligned_path=ne_aligned_path,
        te_aligned_path=te_aligned_path,
        matched_meta_path=matched_meta_path,
        output_dir=output_dir,
        config_name=config_name,
        ne_profile_name=ne_profile_name,
        te_profile_name=te_profile_name,
        ne_anchor_filename=ne_anchor_filename,
        te_anchor_filename=te_anchor_filename,
        fit_summary_filename=fit_summary_filename,
        fit_spec_filename=fit_spec_filename,
        ne_anchor_plot_filename=ne_anchor_plot_filename,
        te_anchor_plot_filename=te_anchor_plot_filename,
        ne_middle_knot_plot_filename=ne_middle_knot_plot_filename,
        te_middle_knot_plot_filename=te_middle_knot_plot_filename,
        ne_fixed_start=ne_fixed_start,
        ne_fixed_end=ne_fixed_end,
        te_fixed_start=te_fixed_start,
        te_fixed_end=te_fixed_end,
        ne_start_min=ne_start_min,
        ne_start_max=ne_start_max,
        te_start_min=te_start_min,
        te_start_max=te_start_max,
    )


@task(name="cluster_anchor_vectors")
def cluster_anchor_vectors_task(
    anchor_path: str,
    output_dir: str,
    prefix: str,
    x_values: list[float],
    y_label: str,
    title_prefix: str,
    x_label: str,
    n_clusters: int,
    max_iter: int,
    tol: float,
    seed: int,
    normalize_per_profile: bool,
    max_profiles_per_cluster: int,
) -> dict[str, Any]:
    clustering = kmeans_cluster_eps_profiles(
        eps_path=anchor_path,
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        normalize_per_profile=normalize_per_profile,
    )
    outputs = save_profile_clustering_outputs(
        clustering=clustering,
        output_dir=output_dir,
        assignments_filename=f"{prefix}_cluster_assignments.csv",
        gallery_filename=f"{prefix}_clusters_gallery.png",
        centers_filename=f"{prefix}_cluster_centers_overlay.png",
        y_label=y_label,
        title_prefix=title_prefix,
        max_profiles_per_cluster=max_profiles_per_cluster,
        seed=seed,
        x_values=np.asarray(x_values, dtype=np.float64),
        x_label=x_label,
    )
    return {
        "anchor_path": anchor_path,
        "n_clusters": int(clustering["n_clusters"]),
        "counts": [int(x) for x in np.asarray(clustering["counts"], dtype=int)],
        "normalize_per_profile": bool(normalize_per_profile),
        "outputs": outputs,
    }


@task(name="estimate_anchor_cluster_gaussians")
def estimate_anchor_cluster_gaussians_task(
    anchor_path: str,
    assignments_csv: str,
    output_dir: str,
    prefix: str,
    cov_regularization: float = 1e-8,
) -> dict[str, Any]:
    return estimate_anchor_cluster_gaussians(
        anchor_path=anchor_path,
        assignments_csv=assignments_csv,
        output_dir=output_dir,
        prefix=prefix,
        cov_regularization=cov_regularization,
    )


@task(name="plot_anchor_cluster_gaussian_samples")
def plot_anchor_cluster_gaussian_samples_task(
    gaussian_params_npz: str,
    output_dir: str,
    plot_filename: str,
    x_values: list[float],
    title: str,
    y_label: str,
    samples_per_cluster: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    return plot_anchor_cluster_gaussian_samples(
        gaussian_params_npz=gaussian_params_npz,
        output_dir=output_dir,
        plot_filename=plot_filename,
        x_values=x_values,
        title=title,
        y_label=y_label,
        samples_per_cluster=samples_per_cluster,
        seed=seed,
    )


@flow(name="real_tene_clustering")
def real_tene_clustering(
    pulses: Sequence[int] | None = None,
    use_filtered_pulses_from_report: bool = True,
    filtered_pulse_report_path: str = DEFAULT_FILTERED_PULSE_REPORT,
    tstart: float = DEFAULT_TSTART,
    tend: float = DEFAULT_TEND,
    dt: float = 0.01,
    read_verbose: bool = False,
    ne_node: str | None = None,
    te_node: str | None = None,
    ne_ppts_profile_key: str = "ne_rhop",
    te_ppts_profile_key: str = "te_rhop",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ne_filename: str = "ne_middle_profiles_real_raw.csv",
    te_filename: str = "te_middle_profiles_real_raw.csv",
    ne_meta_filename: str = "ne_middle_profiles_meta.csv",
    te_meta_filename: str = "te_middle_profiles_meta.csv",
    plot_filename: str = "te_ne_middle_profiles_overlay.png",
    ne_aligned_filename: str = "ne_middle_profiles_real_aligned.csv",
    te_aligned_filename: str = "te_middle_profiles_real_aligned.csv",
    matched_meta_filename: str = "te_ne_matched_meta.csv",
    outlier_report_filename: str = "te_ne_outlier_report.csv",
    min_finite_fraction: float = 0.90,
    min_nonzero_fraction: float = 0.01,
    apply_outlier_filter: bool = True,
    outlier_point_z_threshold: float = 8.0,
    outlier_extreme_point_z_threshold: float = 15.0,
    outlier_min_bad_points: int = 2,
    outlier_bad_fraction_threshold: float = 0.08,
    canonicalize_profile_coordinate: bool = False,
    canonicalize_profile_coordinate_mode: str = "auto",
    use_source_coordinate_grid: bool = True,
    source_coordinate_min: float = 0.0,
    source_coordinate_max: float | None = 1.1,
    spline_config_name: str = "baseline_spline_tene",
    spline_ne_profile_name: str = "electron_density",
    spline_te_profile_name: str = "electron_temperature",
    ne_anchor_filename: str = "ne_spline_anchor_vectors.csv",
    te_anchor_filename: str = "te_spline_anchor_vectors.csv",
    spline_fit_summary_filename: str = "te_ne_spline_fit_summary.csv",
    spline_fit_spec_filename: str = "te_ne_spline_fit_spec.json",
    ne_anchor_plot_filename: str = "ne_spline_anchor_space_overlay.png",
    te_anchor_plot_filename: str = "te_spline_anchor_space_overlay.png",
    ne_middle_knot_plot_filename: str = "ne_middle_profiles_spline_knot_domain_overlay.png",
    te_middle_knot_plot_filename: str = "te_middle_profiles_spline_knot_domain_overlay.png",
    spline_ne_fixed_start: float | None = None,
    spline_ne_fixed_end: float | None = 0.0,
    spline_te_fixed_start: float | None = None,
    spline_te_fixed_end: float | None = 0.0,
    spline_ne_start_min: float | None = 0.3e20,
    spline_ne_start_max: float | None = 1.3e20,
    spline_te_start_min: float | None = 500.0,
    spline_te_start_max: float | None = 2000.0,
    original_data_subdir: str = "original_data",
    cluster_info_subdir: str = "cluster_info",
    reuse_existing_profiles: bool = True,
    run_anchor_clustering: bool = True,
    anchor_cluster_output_subdir: str = "anchor_clusters",
    anchor_cluster_n_clusters: int = 10,
    anchor_cluster_max_iter: int = 150,
    anchor_cluster_tol: float = 1e-4,
    anchor_cluster_seed: int = 0,
    anchor_cluster_normalize_per_profile: bool = False,
    anchor_cluster_max_profiles_per_cluster: int = 80,
    run_anchor_cluster_gaussian_fit: bool = True,
    anchor_cluster_cov_regularization: float = 1e-8,
    run_anchor_cluster_gaussian_samples_plot: bool = True,
    anchor_cluster_samples_per_cluster: int = 4,
    anchor_cluster_samples_seed: int = 0,
    ne_anchor_cluster_samples_plot_filename: str = "ne_anchor_cluster_gaussian_samples_overlay.png",
    te_anchor_cluster_samples_plot_filename: str = "te_anchor_cluster_gaussian_samples_overlay.png",
) -> dict[str, Any]:
    """Read real TS Te/Ne profiles, plasma-gated, and plot middle-time overlays."""
    if tstart < DEFAULT_TSTART or tend > DEFAULT_TEND:
        raise ValueError(
            f"Requested time window [{tstart}, {tend}] is outside allowed "
            f"[{DEFAULT_TSTART}, {DEFAULT_TEND}] s."
        )
    if tstart >= tend:
        raise ValueError(
            f"Invalid time window: tstart={tstart} must be smaller than tend={tend}."
        )
    if ne_ppts_profile_key is None and ne_node is None:
        raise ValueError("For NE, provide either `ne_ppts_profile_key` or `ne_node`.")
    if te_ppts_profile_key is None and te_node is None:
        raise ValueError("For TE, provide either `te_ppts_profile_key` or `te_node`.")

    pulse_list = [int(p) for p in (pulses or [])]
    if use_filtered_pulses_from_report:
        pulse_list = load_non_outlier_pulses_from_report(
            outlier_report_path=filtered_pulse_report_path,
            deduplicate=True,
        )
    if not pulse_list:
        raise RuntimeError(
            "No pulses available: pass `pulses=[...]` or provide a valid filtered pulse report."
        )

    output_root = Path(output_dir)
    original_data_dir = output_root / original_data_subdir
    cluster_info_dir = output_root / cluster_info_subdir
    anchor_cluster_dir = output_root / anchor_cluster_output_subdir
    original_data_dir.mkdir(parents=True, exist_ok=True)
    cluster_info_dir.mkdir(parents=True, exist_ok=True)
    anchor_cluster_dir.mkdir(parents=True, exist_ok=True)

    if reuse_existing_profiles:
        ne_aligned_path = original_data_dir / ne_aligned_filename
        te_aligned_path = original_data_dir / te_aligned_filename
        matched_meta_path = original_data_dir / matched_meta_filename
        missing = [
            str(p)
            for p in (ne_aligned_path, te_aligned_path, matched_meta_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "reuse_existing_profiles=True but required files are missing: "
                + ", ".join(missing)
            )
        outputs = {
            "plot_path": None,
            "ne_aligned_path": str(ne_aligned_path),
            "te_aligned_path": str(te_aligned_path),
            "matched_meta_path": str(matched_meta_path),
            "outlier_report_path": None,
            "num_matched": int(np.loadtxt(ne_aligned_path, delimiter=",", ndmin=2).shape[0]),
            "num_dropped_alignment": None,
            "dropped_alignment_preview": [],
            "num_outliers_removed": None,
            "outlier_filter_applied": False,
            "num_after_outlier_filter": int(
                np.loadtxt(ne_aligned_path, delimiter=",", ndmin=2).shape[0]
            ),
            "num_matched_pre_outlier": int(
                np.loadtxt(ne_aligned_path, delimiter=",", ndmin=2).shape[0]
            ),
            "used_existing_profiles": True,
        }
        ne_dataset: dict[str, Any] | None = None
        te_dataset: dict[str, Any] | None = None
    else:
        ne_dataset = build_real_node_profile_dataset_task(
            pulses=pulse_list,
            node=ne_node if ne_ppts_profile_key is None else None,
            ppts_profile_key=ne_ppts_profile_key,
            output_dir=str(original_data_dir),
            profile_filename=ne_filename,
            meta_filename=ne_meta_filename,
            tstart=tstart,
            tend=tend,
            dt=dt,
            read_verbose=read_verbose,
            min_finite_fraction=min_finite_fraction,
            min_nonzero_fraction=min_nonzero_fraction,
            canonicalize_profile_coordinate=canonicalize_profile_coordinate,
            canonicalize_profile_coordinate_mode=canonicalize_profile_coordinate_mode,
            use_source_coordinate_grid=use_source_coordinate_grid,
            source_coordinate_min=source_coordinate_min,
            source_coordinate_max=source_coordinate_max,
        )
        te_dataset = build_real_node_profile_dataset_task(
            pulses=pulse_list,
            node=te_node if te_ppts_profile_key is None else None,
            ppts_profile_key=te_ppts_profile_key,
            output_dir=str(original_data_dir),
            profile_filename=te_filename,
            meta_filename=te_meta_filename,
            tstart=tstart,
            tend=tend,
            dt=dt,
            read_verbose=read_verbose,
            min_finite_fraction=min_finite_fraction,
            min_nonzero_fraction=min_nonzero_fraction,
            canonicalize_profile_coordinate=canonicalize_profile_coordinate,
            canonicalize_profile_coordinate_mode=canonicalize_profile_coordinate_mode,
            use_source_coordinate_grid=use_source_coordinate_grid,
            source_coordinate_min=source_coordinate_min,
            source_coordinate_max=source_coordinate_max,
        )

        outputs = align_and_plot_te_ne_profiles_task(
            ne_dataset=ne_dataset,
            te_dataset=te_dataset,
            output_dir=str(original_data_dir),
            plot_filename=plot_filename,
            ne_aligned_filename=ne_aligned_filename,
            te_aligned_filename=te_aligned_filename,
            matched_meta_filename=matched_meta_filename,
            outlier_report_filename=outlier_report_filename,
            apply_outlier_filter=apply_outlier_filter,
            outlier_point_z_threshold=outlier_point_z_threshold,
            outlier_extreme_point_z_threshold=outlier_extreme_point_z_threshold,
            outlier_min_bad_points=outlier_min_bad_points,
            outlier_bad_fraction_threshold=outlier_bad_fraction_threshold,
        )
    spline_fit = fit_te_ne_to_spline_anchor_space_task(
        ne_aligned_path=outputs["ne_aligned_path"],
        te_aligned_path=outputs["te_aligned_path"],
        matched_meta_path=outputs["matched_meta_path"],
        output_dir=str(cluster_info_dir),
        config_name=spline_config_name,
        ne_profile_name=spline_ne_profile_name,
        te_profile_name=spline_te_profile_name,
        ne_anchor_filename=ne_anchor_filename,
        te_anchor_filename=te_anchor_filename,
        fit_summary_filename=spline_fit_summary_filename,
        fit_spec_filename=spline_fit_spec_filename,
        ne_anchor_plot_filename=ne_anchor_plot_filename,
        te_anchor_plot_filename=te_anchor_plot_filename,
        ne_middle_knot_plot_filename=ne_middle_knot_plot_filename,
        te_middle_knot_plot_filename=te_middle_knot_plot_filename,
        ne_fixed_start=spline_ne_fixed_start,
        ne_fixed_end=spline_ne_fixed_end,
        te_fixed_start=spline_te_fixed_start,
        te_fixed_end=spline_te_fixed_end,
        ne_start_min=spline_ne_start_min,
        ne_start_max=spline_ne_start_max,
        te_start_min=spline_te_start_min,
        te_start_max=spline_te_start_max,
    )
    outputs["spline_fit"] = spline_fit

    anchor_clustering = None
    if run_anchor_clustering:
        clustering_dir = str(anchor_cluster_dir)
        ne_cluster_spec = load_monospline_anchor_spec(
            config_name=spline_config_name,
            profile_name=spline_ne_profile_name,
        )
        te_cluster_spec = load_monospline_anchor_spec(
            config_name=spline_config_name,
            profile_name=spline_te_profile_name,
        )
        ne_anchor_clustering = cluster_anchor_vectors_task(
            anchor_path=spline_fit["ne_anchor_path"],
            output_dir=clustering_dir,
            prefix="ne_anchors",
            x_values=[float(v) for v in np.asarray(ne_cluster_spec["xknots"]).reshape(-1)],
            y_label="n_e anchor value",
            title_prefix="NE anchor",
            x_label="rhop knot",
            n_clusters=anchor_cluster_n_clusters,
            max_iter=anchor_cluster_max_iter,
            tol=anchor_cluster_tol,
            seed=anchor_cluster_seed,
            normalize_per_profile=anchor_cluster_normalize_per_profile,
            max_profiles_per_cluster=anchor_cluster_max_profiles_per_cluster,
        )
        te_anchor_clustering = cluster_anchor_vectors_task(
            anchor_path=spline_fit["te_anchor_path"],
            output_dir=clustering_dir,
            prefix="te_anchors",
            x_values=[float(v) for v in np.asarray(te_cluster_spec["xknots"]).reshape(-1)],
            y_label="T_e anchor value",
            title_prefix="TE anchor",
            x_label="rhop knot",
            n_clusters=anchor_cluster_n_clusters,
            max_iter=anchor_cluster_max_iter,
            tol=anchor_cluster_tol,
            seed=anchor_cluster_seed,
            normalize_per_profile=anchor_cluster_normalize_per_profile,
            max_profiles_per_cluster=anchor_cluster_max_profiles_per_cluster,
        )
        anchor_clustering = {
            "output_dir": clustering_dir,
            "ne": ne_anchor_clustering,
            "te": te_anchor_clustering,
        }
    outputs["anchor_clustering"] = anchor_clustering

    anchor_cluster_gaussians = None
    if run_anchor_clustering and run_anchor_cluster_gaussian_fit and anchor_clustering is not None:
        ne_gauss = estimate_anchor_cluster_gaussians_task(
            anchor_path=spline_fit["ne_anchor_path"],
            assignments_csv=anchor_clustering["ne"]["outputs"]["assignments_csv"],
            output_dir=str(cluster_info_dir),
            prefix="ne_anchors",
            cov_regularization=anchor_cluster_cov_regularization,
        )
        te_gauss = estimate_anchor_cluster_gaussians_task(
            anchor_path=spline_fit["te_anchor_path"],
            assignments_csv=anchor_clustering["te"]["outputs"]["assignments_csv"],
            output_dir=str(cluster_info_dir),
            prefix="te_anchors",
            cov_regularization=anchor_cluster_cov_regularization,
        )
        anchor_cluster_gaussians = {
            "output_dir": str(cluster_info_dir),
            "ne": ne_gauss,
            "te": te_gauss,
        }
    outputs["anchor_cluster_gaussians"] = anchor_cluster_gaussians

    anchor_cluster_sample_plots = None
    if (
        run_anchor_clustering
        and run_anchor_cluster_gaussian_fit
        and run_anchor_cluster_gaussian_samples_plot
        and anchor_cluster_gaussians is not None
    ):
        ne_cluster_spec = load_monospline_anchor_spec(
            config_name=spline_config_name,
            profile_name=spline_ne_profile_name,
        )
        te_cluster_spec = load_monospline_anchor_spec(
            config_name=spline_config_name,
            profile_name=spline_te_profile_name,
        )
        ne_plot = plot_anchor_cluster_gaussian_samples_task(
            gaussian_params_npz=anchor_cluster_gaussians["ne"]["params_npz"],
            output_dir=str(cluster_info_dir),
            plot_filename=ne_anchor_cluster_samples_plot_filename,
            x_values=[float(v) for v in np.asarray(ne_cluster_spec["xknots"]).reshape(-1)],
            title="NE sampled anchors from cluster Gaussians",
            y_label="n_e anchor value",
            samples_per_cluster=anchor_cluster_samples_per_cluster,
            seed=anchor_cluster_samples_seed,
        )
        te_plot = plot_anchor_cluster_gaussian_samples_task(
            gaussian_params_npz=anchor_cluster_gaussians["te"]["params_npz"],
            output_dir=str(cluster_info_dir),
            plot_filename=te_anchor_cluster_samples_plot_filename,
            x_values=[float(v) for v in np.asarray(te_cluster_spec["xknots"]).reshape(-1)],
            title="TE sampled anchors from cluster Gaussians",
            y_label="T_e anchor value",
            samples_per_cluster=anchor_cluster_samples_per_cluster,
            seed=anchor_cluster_samples_seed + 1,
        )
        anchor_cluster_sample_plots = {"ne": ne_plot, "te": te_plot}
    outputs["anchor_cluster_sample_plots"] = anchor_cluster_sample_plots

    outputs["output_dirs"] = {
        "root": str(output_root),
        "original_data": str(original_data_dir),
        "cluster_info": str(cluster_info_dir),
        "anchor_clusters": str(anchor_cluster_dir),
    }

    return {
        "pulses_requested": pulse_list,
        "n_requested": int(len(pulse_list)),
        "ne_dataset": ne_dataset,
        "te_dataset": te_dataset,
        "outputs": outputs,
    }


if __name__ == "__main__":
    result = real_tene_clustering(
        pulses=None,
        use_filtered_pulses_from_report=True,
        filtered_pulse_report_path=DEFAULT_FILTERED_PULSE_REPORT,
        reuse_existing_profiles=False,
    )
    print("Real Te/Ne clustering read pass complete")
    print(f"Matched pulses: {result['outputs']['num_matched']}/{result['n_requested']}")
    print(f"Outputs: {result['outputs']}")
