"""Prefect flow: clustered-anchor synthetic generation then expanded equilibria projection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence

from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.cluster_anchor_generation import (
    generate_and_save_dataset_from_anchor_cluster_gaussians,
)
from indica.workflows.jussiphd.components.data.combined_transform import (
    build_combined_los_transform,
    save_combined_channel_map,
)
from indica.workflows.jussiphd.components.data.expanded_equilibria_generation import (
    expand_brightness_with_equilibria,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR_STR,
    MULTIPULSE_SYNTHETIC_CLUSTERED_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR,
)

DEFAULT_CLUSTERED_OUTPUT_DIR = MULTIPULSE_SYNTHETIC_CLUSTERED_DATA_DIR_STR
DEFAULT_EXPANDED_OUTPUT_DIR = (
    MULTIPULSE_SYNTHETIC_CLUSTERED_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR
)
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
DEFAULT_CLUSTER_XKNOTS: tuple[float, ...] = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.8,
    0.875,
    0.925,
    0.95,
    0.975,
    0.99,
    1.0,
    1.025,
    1.035,
    1.05,
    1.075,
    1.085,
    1.1,
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


@task(name="generate_clustered_constant_imp_dataset")
def generate_clustered_constant_imp_dataset_task(
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
    single_timepoint_mode: str,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    config_name: str,
    config_overrides: list[str] | None,
    seed: int,
    sample_weight_by_cluster_counts: bool,
    enforce_nonnegative_profiles: bool,
    enforce_strictly_positive_profiles: bool,
    positive_profile_floor: float,
    c_concentration: float,
    ar_concentration: float,
    impurity_flat_zeff: bool,
    show_progress: bool,
    progress_every: int | None,
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
        show_progress=show_progress,
        progress_every=progress_every,
    )


@task(name="expand_clustered_eps_with_equilibria")
def expand_clustered_eps_with_equilibria_task(
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    b_filename: str,
    eps_filename: str,
    meta_filename: str,
    generate_new_data: bool,
    n_timepoints_per_equilibrium: int,
    base_transform: Any | None,
) -> dict[str, Any]:
    return expand_brightness_with_equilibria(
        eps_path=eps_path,
        output_dir=output_dir,
        machine=machine,
        instrument=instrument,
        b_filename=b_filename,
        eps_filename=eps_filename,
        meta_filename=meta_filename,
        generate_new_data=generate_new_data,
        n_timepoints_per_equilibrium=n_timepoints_per_equilibrium,
        base_transform=base_transform,
    )


@task(name="save_combined_los_channel_map")
def save_combined_los_channel_map_task(
    channel_map: list[dict[str, int | str]],
    output_dir: str,
    filename: str,
) -> str:
    return save_combined_channel_map(
        channel_map=channel_map,
        output_dir=output_dir,
        filename=filename,
    )


@flow(name="build_multipulse_synthetic_clustered_expanded_equilibria_constant_imp_dataset")
def build_multipulse_synthetic_clustered_expanded_equilibria_constant_imp_dataset(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    use_combined_instruments: bool = True,
    combined_instruments: tuple[str, ...] = ("blom_xy1", "blom_rz1"),
    combined_instrument_name: str = "blom_xy1_rz1_combined",
    save_combined_map: bool = True,
    combined_map_filename: str = "combined_los_channel_map.csv",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    clustered_output_dir: str = DEFAULT_CLUSTERED_OUTPUT_DIR,
    clustered_b_filename: str = "b_slices_multipulse_synthetic_clustered_xy1rz1.csv",
    clustered_eps_filename: str = "eps_slices_multipulse_synthetic_clustered_xy1rz1.csv",
    clustered_meta_filename: str = "sample_meta_multipulse_synthetic_clustered_xy1rz1.csv",
    generate_new_clustered_data: bool = True,
    n_generations: int = 2500,
    use_all_timepoints: bool = False,
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
    show_progress: bool = True,
    progress_every: int | None = 1,
    ne_gaussian_params_path: str = DEFAULT_NE_GAUSS_NPZ,
    te_gaussian_params_path: str = DEFAULT_TE_GAUSS_NPZ,
    ne_xknots: Sequence[float] = DEFAULT_CLUSTER_XKNOTS,
    te_xknots: Sequence[float] = DEFAULT_CLUSTER_XKNOTS,
    copy_cluster_inputs: bool = True,
    cluster_input_subdir: str = "cluster_inputs",
    ne_assignment_csv: str = DEFAULT_NE_ASSIGN_CSV,
    te_assignment_csv: str = DEFAULT_TE_ASSIGN_CSV,
    ne_gaussian_summary_csv: str = DEFAULT_NE_GAUSS_SUMMARY_CSV,
    te_gaussian_summary_csv: str = DEFAULT_TE_GAUSS_SUMMARY_CSV,
    expanded_output_dir: str = DEFAULT_EXPANDED_OUTPUT_DIR,
    expanded_b_filename: str = "b_slices_multipulse_synthetic_clustered_xy1rz1_expanded_equilibria_constant_imp.csv",
    expanded_eps_filename: str = "eps_slices_multipulse_synthetic_clustered_xy1rz1_expanded_equilibria_constant_imp.csv",
    expanded_meta_filename: str = "sample_meta_multipulse_synthetic_clustered_xy1rz1_expanded_equilibria_constant_imp.csv",
    generate_new_expanded_data: bool = True,
    n_timepoints_per_equilibrium: int = 6,
) -> dict[str, Any]:
    ne_xknots_resolved = [float(x) for x in ne_xknots]
    te_xknots_resolved = [float(x) for x in te_xknots]

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

    if use_combined_instruments:
        transform, channel_map = build_combined_los_transform(
            machine=machine,
            instruments=list(combined_instruments),
            combined_name=combined_instrument_name,
        )
        instrument_for_model = combined_instrument_name
    else:
        transform = transforms[instrument]
        channel_map = []
        instrument_for_model = instrument

    combined_channel_map_path = None
    if use_combined_instruments and save_combined_map:
        combined_channel_map_path = save_combined_los_channel_map_task(
            channel_map=channel_map,
            output_dir=clustered_output_dir,
            filename=combined_map_filename,
        )

    cluster_input_files = [
        ne_gaussian_params_path,
        te_gaussian_params_path,
        ne_assignment_csv,
        te_assignment_csv,
        ne_gaussian_summary_csv,
        te_gaussian_summary_csv,
    ]
    copied_cluster_inputs = None
    if copy_cluster_inputs:
        copied_cluster_inputs = copy_cluster_inputs_task(
            output_dir=clustered_output_dir,
            subdir=cluster_input_subdir,
            files_to_copy=cluster_input_files,
        )

    clustered_generation_kwargs: dict[str, Any] = {
        "machine": machine,
        "instrument": instrument_for_model,
        "transform": transform,
        "equilibrium": equilibrium,
        "ne_gaussian_params_path": ne_gaussian_params_path,
        "te_gaussian_params_path": te_gaussian_params_path,
        "ne_xknots": ne_xknots_resolved,
        "te_xknots": te_xknots_resolved,
        "n_generations": n_generations,
        "use_all_timepoints": use_all_timepoints,
        "single_timepoint_mode": single_timepoint_mode,
        "output_dir": clustered_output_dir,
        "b_filename": clustered_b_filename,
        "eps_filename": clustered_eps_filename,
        "meta_filename": clustered_meta_filename,
        "generate_new_data": generate_new_clustered_data,
        "config_name": config_name,
        "config_overrides": config_overrides,
        "seed": seed,
        "sample_weight_by_cluster_counts": sample_weight_by_cluster_counts,
        "enforce_nonnegative_profiles": enforce_nonnegative_profiles,
        "enforce_strictly_positive_profiles": enforce_strictly_positive_profiles,
        "positive_profile_floor": positive_profile_floor,
        "c_concentration": c_concentration,
        "ar_concentration": ar_concentration,
        "impurity_flat_zeff": impurity_flat_zeff,
        "show_progress": show_progress,
        "progress_every": progress_every,
    }
    clustered_dataset = generate_clustered_constant_imp_dataset_task(
        **clustered_generation_kwargs,
    )

    expanded_generation_kwargs: dict[str, Any] = {
        "eps_path": str(clustered_dataset["eps_path"]),
        "output_dir": expanded_output_dir,
        "machine": machine,
        "instrument": instrument_for_model,
        "b_filename": expanded_b_filename,
        "eps_filename": expanded_eps_filename,
        "meta_filename": expanded_meta_filename,
        "generate_new_data": generate_new_expanded_data,
        "n_timepoints_per_equilibrium": n_timepoints_per_equilibrium,
        "base_transform": transform,
    }
    expanded_dataset = expand_clustered_eps_with_equilibria_task(
        **expanded_generation_kwargs,
    )

    return {
        "clustered_dataset": clustered_dataset,
        "expanded_dataset": expanded_dataset,
        "copied_cluster_inputs": copied_cluster_inputs,
        "clustered_output_dir": clustered_output_dir,
        "expanded_output_dir": expanded_output_dir,
        "instrument_used": instrument_for_model,
        "use_combined_instruments": bool(use_combined_instruments),
        "combined_instruments": list(combined_instruments),
        "combined_channel_map_path": combined_channel_map_path,
    }


if __name__ == "__main__":
    result = build_multipulse_synthetic_clustered_expanded_equilibria_constant_imp_dataset()
    print("Clustered constant-imp + expanded-equilibria synthetic dataset complete")
    print(result)
