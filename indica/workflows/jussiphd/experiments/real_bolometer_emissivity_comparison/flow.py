"""Prefect flow: compare model-inferred emissivity from real DV1 bolometry to T1D XY1 emissivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from prefect import flow, task

from indica.workflows.jussiphd.components.data.equilibrium_snapshot_dataset import (
    load_non_outlier_pulses_from_report,
)
from indica.workflows.jussiphd.components.evaluation import (
    compare_saved_model_vs_real_emissivity_nodes,
)
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_CLUSTERED_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR,
)


DV1_EMISSION_NODE = r"\ST40::TOP.BLOM_DV1.BEST.PROFILES:EMISSION"
XY1_EMIS_LOC_NODE = r"\ST40::TOP.T1D_BLOM_XY1.BEST.PROFILES.PSI_NORM:EMIS_LOC"
XY1_EMIS_LOC_ERR_NODE = r"\ST40::TOP.T1D_BLOM_XY1.BEST.PROFILES.PSI_NORM:EMIS_LOC_ERR"

DEFAULT_OUTLIER_REPORT_PATH = str(
    Path(__file__).resolve().parents[1]
    / "real_tene_clustering"
    / "outputs"
    / "original_data"
    / "te_ne_outlier_report.csv"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[2]
    / "components"
    / "ml"
    / "flow_data"
    / "multipulse_synthetic_clustered_expanded_equilibria_constant_imp_compare"
    / "vae_multipulse_synthetic_clustered_expanded_equilibria_constant_imp.pt"
)
DEFAULT_REFERENCE_B_PATH = str(
    MULTIPULSE_SYNTHETIC_CLUSTERED_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR
    / "b_slices_multipulse_synthetic_clustered_expanded_equilibria_constant_imp.csv"
)
DEFAULT_REFERENCE_EPS_PATH = str(
    MULTIPULSE_SYNTHETIC_CLUSTERED_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR
    / "eps_slices_multipulse_synthetic_clustered_expanded_equilibria_constant_imp.csv"
)


@task(name="sample_verified_non_outlier_pulses")
def sample_verified_non_outlier_pulses_task(
    outlier_report_path: str,
    n_pulses: int,
    seed: int | None,
) -> list[int]:
    pulses = load_non_outlier_pulses_from_report(
        outlier_report_path=outlier_report_path,
        deduplicate=True,
    )
    if not pulses:
        raise RuntimeError("No non-outlier pulses available in outlier report.")
    rng = np.random.default_rng(seed)
    n = min(int(max(1, n_pulses)), len(pulses))
    if len(pulses) <= n:
        return [int(p) for p in pulses]
    sampled = rng.choice(np.asarray(pulses, dtype=int), size=n, replace=False)
    return [int(p) for p in sampled.tolist()]


@task(name="compare_model_vs_t1d_for_real_pulses")
def compare_model_vs_t1d_for_real_pulses_task(
    pulses: list[int],
    model_path: str,
    reference_b_path: str,
    reference_eps_path: str,
    output_dir: str,
    dv1_emission_node: str,
    xy1_emis_loc_node: str,
    xy1_emis_loc_err_node: str,
    tstart: float,
    tend: float,
    dt: float,
    k_samples: int,
    seed: int | None,
    enforce_nonnegative_output: bool,
    max_plot_times_per_pulse: int,
) -> dict[str, Any]:
    return compare_saved_model_vs_real_emissivity_nodes(
        pulses=pulses,
        model_path=model_path,
        reference_b_path=reference_b_path,
        reference_eps_path=reference_eps_path,
        output_dir=output_dir,
        dv1_emission_node=dv1_emission_node,
        xy1_emis_loc_node=xy1_emis_loc_node,
        xy1_emis_loc_err_node=xy1_emis_loc_err_node,
        tstart=tstart,
        tend=tend,
        dt=dt,
        k_samples=k_samples,
        seed=seed,
        enforce_nonnegative_output=enforce_nonnegative_output,
        max_plot_times_per_pulse=max_plot_times_per_pulse,
    )


@flow(name="real_bolometer_emissivity_comparison")
def compare_real_bolometer_vs_t1d_emissivity(
    outlier_report_path: str = DEFAULT_OUTLIER_REPORT_PATH,
    n_random_pulses: int = 30,
    pulse_seed: int | None = 0,
    model_path: str = DEFAULT_MODEL_PATH,
    reference_b_path: str = DEFAULT_REFERENCE_B_PATH,
    reference_eps_path: str = DEFAULT_REFERENCE_EPS_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dv1_emission_node: str = DV1_EMISSION_NODE,
    xy1_emis_loc_node: str = XY1_EMIS_LOC_NODE,
    xy1_emis_loc_err_node: str = XY1_EMIS_LOC_ERR_NODE,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    k_samples: int = 40,
    inference_seed: int | None = 0,
    enforce_nonnegative_output: bool = True,
    max_plot_times_per_pulse: int = 6,
) -> dict[str, Any]:
    pulses = sample_verified_non_outlier_pulses_task(
        outlier_report_path=outlier_report_path,
        n_pulses=n_random_pulses,
        seed=pulse_seed,
    )
    comparison = compare_model_vs_t1d_for_real_pulses_task(
        pulses=pulses,
        model_path=model_path,
        reference_b_path=reference_b_path,
        reference_eps_path=reference_eps_path,
        output_dir=output_dir,
        dv1_emission_node=dv1_emission_node,
        xy1_emis_loc_node=xy1_emis_loc_node,
        xy1_emis_loc_err_node=xy1_emis_loc_err_node,
        tstart=tstart,
        tend=tend,
        dt=dt,
        k_samples=k_samples,
        seed=inference_seed,
        enforce_nonnegative_output=enforce_nonnegative_output,
        max_plot_times_per_pulse=max_plot_times_per_pulse,
    )
    return {
        "selected_pulses": [int(p) for p in pulses],
        "num_selected_pulses": int(len(pulses)),
        "comparison": comparison,
    }


if __name__ == "__main__":
    result = compare_real_bolometer_vs_t1d_emissivity()
    print("Real bolometer vs T1D emissivity comparison complete")
    print(f"Pulses: {result['selected_pulses']}")
    print(f"Outputs: {result['comparison']}")

