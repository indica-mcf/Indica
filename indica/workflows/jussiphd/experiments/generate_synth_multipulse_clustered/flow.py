"""Prefect flow for synthetic dataset generation from real TE/NE anchor clusters."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.cluster_anchor_generation import (
    generate_and_save_dataset_from_anchor_cluster_gaussians,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR_STR,
)

DEFAULT_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR_STR
DEFAULT_REAL_TENE_OUTPUTS = (
    Path(__file__).resolve().parents[1] / "real_tene_clustering" / "outputs"
)
DEFAULT_NE_GAUSS_NPZ = str(
    DEFAULT_REAL_TENE_OUTPUTS / "cluster_info" / "ne_anchors_cluster_gaussian_params.npz"
)
DEFAULT_TE_GAUSS_NPZ = str(
    DEFAULT_REAL_TENE_OUTPUTS / "cluster_info" / "te_anchors_cluster_gaussian_params.npz"
)
DEFAULT_NE_ASSIGN_CSV = str(
    DEFAULT_REAL_TENE_OUTPUTS / "anchor_clusters" / "ne_anchors_cluster_assignments.csv"
)
DEFAULT_TE_ASSIGN_CSV = str(
    DEFAULT_REAL_TENE_OUTPUTS / "anchor_clusters" / "te_anchors_cluster_assignments.csv"
)
DEFAULT_NE_GAUSS_SUMMARY_CSV = str(
    DEFAULT_REAL_TENE_OUTPUTS / "cluster_info" / "ne_anchors_cluster_gaussian_summary.csv"
)
DEFAULT_TE_GAUSS_SUMMARY_CSV = str(
    DEFAULT_REAL_TENE_OUTPUTS / "cluster_info" / "te_anchors_cluster_gaussian_summary.csv"
)


@task(name="load_real_equilibrium")
def load_real_equilibrium_task(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    verbose: bool,
) -> Any:
    return load_real_equilibrium_from_pulse(
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        verbose=verbose,
    )


@task(name="copy_cluster_inputs")
def copy_cluster_inputs_task(
    output_dir: str,
    files_to_copy: list[str],
    subdir: str = "cluster_inputs",
) -> dict[str, Any]:
    dst_dir = Path(output_dir) / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []
    for src in files_to_copy:
        src_path = Path(src)
        if not src_path.exists():
            missing.append(str(src_path))
            continue
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        copied.append(str(dst_path))
    if missing:
        raise FileNotFoundError("Missing cluster input files: " + ", ".join(missing))
    return {
        "output_dir": str(dst_dir),
        "copied_files": copied,
        "num_copied": int(len(copied)),
    }


@task(name="generate_multipulse_synthetic_dataset_from_anchor_clusters")
def generate_multipulse_synthetic_dataset_from_anchor_clusters_task(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    ne_gaussian_params_path: str,
    te_gaussian_params_path: str,
    ne_xknots: list[float],
    te_xknots: list[float],
    n_generations: int,
    use_all_timepoints: bool,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    config_name: str,
    config_overrides: list[str] | None,
    single_timepoint_mode: str,
    seed: int,
    sample_weight_by_cluster_counts: bool,
    enforce_nonnegative_profiles: bool,
    enforce_strictly_positive_profiles: bool,
    positive_profile_floor: float,
    c_concentration: float,
    ar_concentration: float,
    impurity_flat_zeff: bool,
) -> dict[str, Any]:
    return generate_and_save_dataset_from_anchor_cluster_gaussians(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        ne_gaussian_params_path=ne_gaussian_params_path,
        te_gaussian_params_path=te_gaussian_params_path,
        ne_xknots=ne_xknots,
        te_xknots=te_xknots,
        n_generations=n_generations,
        use_all_timepoints=use_all_timepoints,
        single_timepoint_mode=single_timepoint_mode,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        config_name=config_name,
        config_overrides=config_overrides,
        seed=seed,
        sample_weight_by_cluster_counts=sample_weight_by_cluster_counts,
        enforce_nonnegative_profiles=enforce_nonnegative_profiles,
        enforce_strictly_positive_profiles=enforce_strictly_positive_profiles,
        positive_profile_floor=positive_profile_floor,
        impurity_concentrations={"c": float(c_concentration), "ar": float(ar_concentration)},
        impurity_flat_zeff=impurity_flat_zeff,
    )


@flow(name="bolometry_inversion_multipulse_synthetic_clustered")
def bolometry_inversion_multipulse_synthetic_clustered(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    b_filename: str = "b_slices_multipulse_synthetic_clustered.csv",
    eps_filename: str = "eps_slices_multipulse_synthetic_clustered.csv",
    meta_filename: str = "sample_meta_multipulse_synthetic_clustered.csv",
    n_generations: int = 100,
    generate_new_data: bool = True,
    use_all_timepoints: bool = True,
    single_timepoint_mode: str = "middle",
    config_name: str = "baseline_spline_tene",
    config_overrides: list[str] | None = None,
    seed: int = 0,
    sample_weight_by_cluster_counts: bool = True,
    enforce_nonnegative_profiles: bool = True,
    enforce_strictly_positive_profiles: bool = False,
    positive_profile_floor: float = 1e-12,
    c_concentration: float = 0.05,
    ar_concentration: float = 0.01,
    impurity_flat_zeff: bool = True,
    ne_gaussian_params_path: str = DEFAULT_NE_GAUSS_NPZ,
    te_gaussian_params_path: str = DEFAULT_TE_GAUSS_NPZ,
    ne_xknots: list[float] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    te_xknots: list[float] = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    copy_cluster_inputs: bool = True,
    cluster_input_subdir: str = "cluster_inputs",
    ne_assignment_csv: str = DEFAULT_NE_ASSIGN_CSV,
    te_assignment_csv: str = DEFAULT_TE_ASSIGN_CSV,
    ne_gaussian_summary_csv: str = DEFAULT_NE_GAUSS_SUMMARY_CSV,
    te_gaussian_summary_csv: str = DEFAULT_TE_GAUSS_SUMMARY_CSV,
) -> dict[str, Any]:
    transforms = load_default_objects(machine, "geometry")
    if use_real_equilibrium:
        equilibrium = load_real_equilibrium_task(
            pulse=real_equilibrium_pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=real_equilibrium_verbose,
        )
    else:
        equilibrium = load_default_objects(machine, "equilibrium")
    transform = transforms[instrument]

    copied_cluster_inputs = None
    if copy_cluster_inputs:
        copied_cluster_inputs = copy_cluster_inputs_task(
            output_dir=output_dir,
            subdir=cluster_input_subdir,
            files_to_copy=[
                ne_gaussian_params_path,
                te_gaussian_params_path,
                ne_assignment_csv,
                te_assignment_csv,
                ne_gaussian_summary_csv,
                te_gaussian_summary_csv,
            ],
        )

    synthetic_dataset = generate_multipulse_synthetic_dataset_from_anchor_clusters_task(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        ne_gaussian_params_path=ne_gaussian_params_path,
        te_gaussian_params_path=te_gaussian_params_path,
        ne_xknots=ne_xknots,
        te_xknots=te_xknots,
        n_generations=n_generations,
        use_all_timepoints=use_all_timepoints,
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        config_name=config_name,
        config_overrides=config_overrides,
        single_timepoint_mode=single_timepoint_mode,
        seed=seed,
        sample_weight_by_cluster_counts=sample_weight_by_cluster_counts,
        enforce_nonnegative_profiles=enforce_nonnegative_profiles,
        enforce_strictly_positive_profiles=enforce_strictly_positive_profiles,
        positive_profile_floor=positive_profile_floor,
        c_concentration=c_concentration,
        ar_concentration=ar_concentration,
        impurity_flat_zeff=impurity_flat_zeff,
    )

    return {
        "synthetic_dataset": synthetic_dataset,
        "copied_cluster_inputs": copied_cluster_inputs,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    result = bolometry_inversion_multipulse_synthetic_clustered()
    print("Clustered-anchor synthetic dataset generation complete")
    print(result)
