"""Prefect flow for calibrating synthetic Poisson noise against real data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Sequence

from prefect import flow, task

from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    evaluate_noise_levels_against_real,
    save_noise_likelihood_outputs,
)
from indica.workflows.jussiphd.experiments.multipulse_real.flow import (
    bolometry_inversion_multipulse_real,
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


@task(name="run_real_dataset_only")
def run_real_dataset_only_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return bolometry_inversion_multipulse_real(**kwargs)


@task(name="run_synthetic_dataset_only")
def run_synthetic_dataset_only_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return bolometry_inversion_multipulse_synthetic(**kwargs)


@task(name="evaluate_noise_likelihood")
def evaluate_noise_likelihood_task(
    synthetic_b_path: str,
    synthetic_eps_path: str,
    real_b_path: str,
    real_eps_path: str,
    count_levels: list[float],
    seed: int,
    bins: int,
    b_scale_percentile: float,
    eps_scale_percentile: float,
) -> dict[str, Any]:
    return evaluate_noise_levels_against_real(
        synthetic_b_path=synthetic_b_path,
        synthetic_eps_path=synthetic_eps_path,
        real_b_path=real_b_path,
        real_eps_path=real_eps_path,
        count_levels=count_levels,
        seed=seed,
        bins=bins,
        b_scale_percentile=b_scale_percentile,
        eps_scale_percentile=eps_scale_percentile,
    )


@task(name="save_noise_likelihood_results")
def save_noise_likelihood_results_task(
    output_dir: str,
    run_id: str,
    result: dict[str, Any],
) -> dict[str, str]:
    return save_noise_likelihood_outputs(
        output_dir=output_dir,
        run_id=run_id,
        result=result,
    )


@flow(name="calibrate_synthetic_noise_against_real")
def calibrate_synthetic_noise_against_real(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pulses: Sequence[int] | None = None,
    pulses_csv: str = "13622,13623,13624",
    count_levels: Sequence[float] | None = None,
    count_levels_csv: str = "1,2,5,10,20,50,100,200,500,1000",
    machine: str = "st40",
    instrument: str = "blom_xy1",
    emissivity_instrument: str = "blom_rz1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    revision: int = 0,
    node: str | None = None,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    synthetic_n_generations: int = 3000,
    real_generate_new_data: bool = True,
    synthetic_generate_new_data: bool = False,
    real_existing_output_dir: str | None = None,
    synthetic_existing_output_dir: str | None = None,
    seed: int = 0,
    bins: int = 256,
    b_scale_percentile: float = 99.0,
    eps_scale_percentile: float = 99.0,
) -> dict[str, Any]:
    """Create real+synthetic datasets and calibrate synthetic Poisson noise level."""
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
        else str(run_root / "real_data")
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
    real_eps_filename = "eps_slices.csv"
    if not real_generate_new_data:
        real_b_filename = _resolve_existing_file(
            real_dir_path,
            ["b_slices.csv", "b_slices_multipulse_real.csv"],
            "real brightness",
        )
        real_eps_filename = _resolve_existing_file(
            real_dir_path,
            ["eps_slices.csv", "eps_slices_multipulse_real.csv"],
            "real emissivity",
        )

    real_kwargs = {
        "machine": machine,
        "instrument": instrument,
        "emissivity_instrument": emissivity_instrument,
        "pulses": pulse_list,
        "tstart": tstart,
        "tend": tend,
        "dt": dt,
        "use_real_equilibrium": use_real_equilibrium,
        "real_equilibrium_pulse": real_equilibrium_pulse,
        "real_equilibrium_verbose": real_equilibrium_verbose,
        "revision": revision,
        "node": node,
        "output_dir": real_output_dir,
        "b_filename": real_b_filename,
        "eps_filename": real_eps_filename,
        "generate_new_data": real_generate_new_data,
        "use_all_timepoints": True,
        "create_training_dataset": False,
        "run_vae_training": False,
        "run_vae_metrics": False,
        "run_visualisations": False,
    }
    synthetic_kwargs = {
        "machine": machine,
        "instrument": instrument,
        "tstart": tstart,
        "tend": tend,
        "dt": dt,
        "use_real_equilibrium": use_real_equilibrium,
        "real_equilibrium_pulse": real_equilibrium_pulse,
        "real_equilibrium_verbose": real_equilibrium_verbose,
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

    real_result = run_real_dataset_only_task(real_kwargs)
    synthetic_result = run_synthetic_dataset_only_task(synthetic_kwargs)

    real_dataset = real_result.get("real_dataset", {})
    synthetic_dataset = synthetic_result.get("synthetic_dataset", {})
    noise_result = evaluate_noise_likelihood_task(
        synthetic_b_path=str(synthetic_dataset["b_path"]),
        synthetic_eps_path=str(synthetic_dataset["eps_path"]),
        real_b_path=str(real_dataset["b_path"]),
        real_eps_path=str(real_dataset["eps_path"]),
        count_levels=count_level_list,
        seed=int(seed),
        bins=int(bins),
        b_scale_percentile=float(b_scale_percentile),
        eps_scale_percentile=float(eps_scale_percentile),
    )

    persisted = save_noise_likelihood_results_task(
        output_dir=str(run_root),
        run_id=run_id,
        result=noise_result,
    )

    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "real_result": real_result,
        "synthetic_result": synthetic_result,
        "noise_likelihood": noise_result,
        "outputs": persisted,
        "best_count_level": noise_result["best"]["count_level"],
    }


if __name__ == "__main__":
    result = calibrate_synthetic_noise_against_real(pulses=list(range(12800,13000)))
    print("Noise calibration complete")
    print(f"Best count level: {result['best_count_level']}")
    print(f"Plot: {result['outputs']['plot_path']}")
