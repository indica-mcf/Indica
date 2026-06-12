"""Prefect flow for calibrating synthetic brightness noise against real RZ1 brightness."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Sequence

from prefect import flow, task

from indica.workflows.jussiphd.components.data.real_brightness_dataset_generation import (
    generate_and_save_real_multipulse_brightness_dataset,
)
from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    evaluate_brightness_noise_levels_against_real,
    save_brightness_noise_likelihood_outputs,
)
from indica.workflows.jussiphd.experiments.multipulse_synthetic.flow import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SYNTHETIC_DATA_OUTPUT_DIR,
    bolometry_inversion_multipulse_synthetic,
)


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def _parse_int_csv(values_csv: str) -> list[int]:
    return [int(v.strip()) for v in values_csv.split(",") if v.strip()]


def _parse_float_csv(values_csv: str) -> list[float]:
    return [float(v.strip()) for v in values_csv.split(",") if v.strip()]


def _resolve_existing_file(dir_path: Path, candidates: Sequence[str], kind: str) -> str:
    for name in candidates:
        p = dir_path / name
        if p.exists():
            return name
    raise FileNotFoundError(
        f"Could not find existing {kind} file in {dir_path}. Tried: {list(candidates)}"
    )


@task(name="run_real_brightness_dataset_only")
def run_real_brightness_dataset_only_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return generate_and_save_real_multipulse_brightness_dataset(**kwargs)


@task(name="run_synthetic_dataset_only")
def run_synthetic_dataset_only_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return bolometry_inversion_multipulse_synthetic(**kwargs)


@task(name="evaluate_brightness_noise_likelihood")
def evaluate_brightness_noise_likelihood_task(
    synthetic_b_path: str,
    real_b_path: str,
    count_levels: list[float],
    seed: int,
    bins: int,
    b_scale_percentile: float,
) -> dict[str, Any]:
    return evaluate_brightness_noise_levels_against_real(
        synthetic_b_path=synthetic_b_path,
        real_b_path=real_b_path,
        count_levels=count_levels,
        seed=seed,
        bins=bins,
        b_scale_percentile=b_scale_percentile,
    )


@task(name="save_brightness_noise_likelihood_results")
def save_brightness_noise_likelihood_results_task(
    output_dir: str,
    run_id: str,
    result: dict[str, Any],
) -> dict[str, str]:
    return save_brightness_noise_likelihood_outputs(
        output_dir=output_dir,
        run_id=run_id,
        result=result,
    )


@flow(name="calibrate_synthetic_brightness_noise_against_real")
def calibrate_synthetic_brightness_noise_against_real(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pulses: Sequence[int] | None = None,
    pulses_csv: str = "13622,13623,13624",
    count_levels: Sequence[float] | None = None,
    count_levels_csv: str = "1,2,5,10,20,50,100,200,500,1000",
    machine: str = "st40",
    synthetic_instrument: str = "blom_xy1",
    real_brightness_instrument: str = "blom_rz1",
    real_brightness_node: str | None = None,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    revision: int = 0,
    synthetic_n_generations: int = 3000,
    real_generate_new_data: bool = True,
    synthetic_generate_new_data: bool = False,
    real_existing_output_dir: str | None = None,
    synthetic_existing_output_dir: str | None = None,
    seed: int = 0,
    bins: int = 256,
    b_scale_percentile: float = 99.0,
    apply_basic_quality_filter: bool = True,
    min_finite_fraction: float = 0.95,
    min_nonzero_fraction: float = 0.01,
    nonzero_threshold: float = 0.0,
    read_verbose: bool = False,
) -> dict[str, Any]:
    """Create real RZ1 brightness + synthetic brightness datasets and calibrate Poisson level."""
    pulse_list = [int(p) for p in (pulses or [])]
    if not pulse_list:
        pulse_list = _parse_int_csv(pulses_csv)
    if not pulse_list:
        raise ValueError("`pulses_csv` must contain at least one pulse.")

    count_level_list = [float(c) for c in (count_levels or [])]
    if not count_level_list:
        count_level_list = _parse_float_csv(count_levels_csv)
    if not count_level_list:
        raise ValueError("`count_levels_csv` must contain at least one value.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = out_dir / run_id

    real_output_dir = (
        str(Path(real_existing_output_dir))
        if (not real_generate_new_data and real_existing_output_dir)
        else str(run_root / "real_brightness_data")
    )
    synthetic_output_dir = (
        str(Path(synthetic_existing_output_dir))
        if (not synthetic_generate_new_data and synthetic_existing_output_dir)
        else (
            DEFAULT_SYNTHETIC_DATA_OUTPUT_DIR
            if not synthetic_generate_new_data
            else str(run_root / "synthetic_data")
        )
    )

    synthetic_dir_path = Path(synthetic_output_dir)
    real_dir_path = Path(real_output_dir)
    synthetic_b_filename = "b_slices.csv"
    synthetic_eps_filename = "eps_slices.csv"
    if not synthetic_generate_new_data:
        synthetic_b_filename = _resolve_existing_file(
            synthetic_dir_path,
            ["b_slices.csv", "b_slices_multipulse_synthetic.csv"],
            "synthetic brightness",
        )
        synthetic_eps_filename = _resolve_existing_file(
            synthetic_dir_path,
            ["eps_slices.csv", "eps_slices_multipulse_synthetic.csv"],
            "synthetic emissivity",
        )

    real_b_filename = "b_slices.csv"
    if not real_generate_new_data:
        real_b_filename = _resolve_existing_file(
            real_dir_path,
            ["b_slices.csv", "b_slices_multipulse_real_rz1_channels.csv"],
            "real brightness",
        )

    real_kwargs = {
        "pulses": pulse_list,
        "instrument": real_brightness_instrument,
        "tstart": tstart,
        "tend": tend,
        "dt": dt,
        "output_dir": real_output_dir,
        "b_filename": real_b_filename,
        "meta_filename": "sample_meta.csv",
        "use_all_timepoints": True,
        "node": real_brightness_node,
        "revision": revision,
        "generate_new_data": real_generate_new_data,
        "verbose": read_verbose,
        "apply_basic_quality_filter": apply_basic_quality_filter,
        "min_finite_fraction": min_finite_fraction,
        "min_nonzero_fraction": min_nonzero_fraction,
        "nonzero_threshold": nonzero_threshold,
    }
    synthetic_kwargs = {
        "machine": machine,
        "instrument": synthetic_instrument,
        "tstart": tstart,
        "tend": tend,
        "dt": dt,
        "output_dir": synthetic_output_dir,
        "b_filename": synthetic_b_filename,
        "eps_filename": synthetic_eps_filename,
        "n_generations": synthetic_n_generations,
        "generate_new_data": synthetic_generate_new_data,
        "use_all_timepoints": True,
        "create_training_dataset": False,
        "run_vae_training": False,
        "run_vae_metrics": False,
        "run_visualisations": False,
    }

    real_dataset = run_real_brightness_dataset_only_task(real_kwargs)
    synthetic_result = run_synthetic_dataset_only_task(synthetic_kwargs)
    synthetic_dataset = synthetic_result.get("synthetic_dataset", {})

    noise_result = evaluate_brightness_noise_likelihood_task(
        synthetic_b_path=str(synthetic_dataset["b_path"]),
        real_b_path=str(real_dataset["b_path"]),
        count_levels=count_level_list,
        seed=int(seed),
        bins=int(bins),
        b_scale_percentile=float(b_scale_percentile),
    )
    persisted = save_brightness_noise_likelihood_results_task(
        output_dir=str(run_root),
        run_id=run_id,
        result=noise_result,
    )

    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "real_brightness_dataset": real_dataset,
        "synthetic_result": synthetic_result,
        "noise_likelihood": noise_result,
        "outputs": persisted,
        "best_count_level": noise_result["best"]["count_level"],
    }


if __name__ == "__main__":
    result = calibrate_synthetic_brightness_noise_against_real(pulses=list(range(12800, 13000)))
    print("Brightness-only noise calibration complete")
    print(f"Best count level: {result['best_count_level']}")
    print(f"Plot: {result['outputs']['plot_path']}")
