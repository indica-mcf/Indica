"""Prefect flow for aligned real-vs-synthetic experiment comparisons."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from prefect import flow, task

from indica.workflows.jussiphd.experiments.comparison.config import ComparisonConfig
from indica.workflows.jussiphd.experiments.comparison.config import DEFAULT_OUTPUT_DIR
from indica.workflows.jussiphd.experiments.comparison.config import build_real_flow_kwargs
from indica.workflows.jussiphd.experiments.comparison.config import build_synthetic_flow_kwargs
from indica.workflows.jussiphd.experiments.comparison.config import parse_pulses_csv
from indica.workflows.jussiphd.experiments.multipulse_real.flow import (
    bolometry_inversion_multipulse_real,
)
from indica.workflows.jussiphd.experiments.multipulse_synthetic.flow import (
    bolometry_inversion_multipulse_synthetic,
)


def _extract_core_metrics(result: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    dataset = result.get(dataset_key, {}) or {}
    metrics = result.get("vae_metrics", {}) or {}
    diversity = metrics.get("diversity", {}) or {}
    forward = metrics.get("forward_consistency", {}) or {}
    train = result.get("vae_training", {}) or {}

    return {
        "num_pairs": dataset.get("num_pairs"),
        "b_shape": str(dataset.get("b_shape")),
        "eps_shape": str(dataset.get("eps_shape")),
        "vae_model_path": train.get("model_path"),
        "vae_num_parameters": train.get("num_parameters"),
        "vae_last_epoch_loss": train.get("last_epoch_loss"),
        "vae_last_recon_loss": train.get("last_recon_loss"),
        "vae_last_kl_loss": train.get("last_kl_loss"),
        "metric_div_l2_mean_to_true_norm": diversity.get("l2_sample_mean_to_true_norm"),
        "metric_div_mean_l2_to_sample_mean_norm": diversity.get("mean_l2_to_sample_mean_norm"),
        "metric_div_mean_per_dim_std_norm": diversity.get("mean_per_dim_std_norm"),
        "metric_fw_rmse_norm": forward.get("rmse_norm"),
        "metric_fw_mean_l2_norm": forward.get("mean_l2_norm"),
        "metric_fw_max_abs_error_norm": forward.get("max_abs_error_norm"),
    }


def _build_summary_rows(
    real_result: dict[str, Any],
    synthetic_result: dict[str, Any],
) -> list[dict[str, Any]]:
    real_core = _extract_core_metrics(real_result, dataset_key="real_dataset")
    synth_core = _extract_core_metrics(synthetic_result, dataset_key="synthetic_dataset")

    rows: list[dict[str, Any]] = []
    real_row = {"run_type": "real"}
    real_row.update(real_core)
    rows.append(real_row)

    synth_row = {"run_type": "synthetic"}
    synth_row.update(synth_core)
    rows.append(synth_row)

    delta_row = {"run_type": "delta_real_minus_synthetic"}
    for key in real_core.keys():
        rv = real_core.get(key)
        sv = synth_core.get(key)
        if isinstance(rv, (int, float)) and isinstance(sv, (int, float)):
            delta_row[key] = rv - sv
        else:
            delta_row[key] = ""
    rows.append(delta_row)

    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["run_type"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def _write_summary_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return str(path)


@task(name="run_real_experiment")
def run_real_experiment_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return bolometry_inversion_multipulse_real(**kwargs)


@task(name="run_synthetic_experiment")
def run_synthetic_experiment_task(kwargs: dict[str, Any]) -> dict[str, Any]:
    return bolometry_inversion_multipulse_synthetic(**kwargs)


@task(name="build_comparison_rows")
def build_comparison_rows_task(
    real_result: dict[str, Any],
    synthetic_result: dict[str, Any],
) -> list[dict[str, Any]]:
    return _build_summary_rows(real_result, synthetic_result)


@task(name="write_comparison_summaries")
def write_comparison_summaries_task(
    output_dir: str,
    run_id: str,
    cfg: ComparisonConfig,
    real_kwargs: dict[str, Any],
    synthetic_kwargs: dict[str, Any],
    real_result: dict[str, Any],
    synthetic_result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    out_dir = Path(output_dir)
    csv_path = _write_summary_csv(out_dir / f"real_vs_synthetic_summary_{run_id}.csv", rows)
    json_payload = {
        "run_id": run_id,
        "config": cfg.as_dict(),
        "real_kwargs": real_kwargs,
        "synthetic_kwargs": synthetic_kwargs,
        "rows": rows,
        "real_result": real_result,
        "synthetic_result": synthetic_result,
    }
    json_path = _write_summary_json(
        out_dir / f"real_vs_synthetic_summary_{run_id}.json",
        json_payload,
    )
    return {"summary_csv": csv_path, "summary_json": json_path}


@flow(name="compare_real_vs_synthetic")
def compare_real_vs_synthetic(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    pulses_csv: str = "13622,13623,13624",
    instrument: str = "blom_xy1",
    emissivity_instrument: str = "blom_rz1",
    n_generations_synthetic: int = 3000,
    run_visualisations: bool = False,
    use_real_equilibrium: bool = False,
    real_equilibrium_pulse: int = 13622,
    vae_hidden_scaling: int = 8,
    vae_latent_dim: int = 4,
    vae_n_epochs: int = 25,
) -> dict[str, Any]:
    """Run aligned real and synthetic experiments and save summary CSV/JSON."""
    cfg = ComparisonConfig(
        instrument=instrument,
        emissivity_instrument=emissivity_instrument,
        pulses=tuple(parse_pulses_csv(pulses_csv)),
        n_generations_synthetic=int(n_generations_synthetic),
        run_visualisations=bool(run_visualisations),
        use_real_equilibrium=bool(use_real_equilibrium),
        real_equilibrium_pulse=int(real_equilibrium_pulse),
        vae_hidden_scaling=int(vae_hidden_scaling),
        vae_latent_dim=int(vae_latent_dim),
        vae_n_epochs=int(vae_n_epochs),
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    real_output_dir = str(out_dir / f"{run_id}_real")
    real_vae_dir = str(out_dir / f"{run_id}_real_vae")
    real_vis_dir = str(out_dir / f"{run_id}_real_vis")
    synth_output_dir = str(out_dir / f"{run_id}_synthetic")
    synth_vae_dir = str(out_dir / f"{run_id}_synthetic_vae")
    synth_vis_dir = str(out_dir / f"{run_id}_synthetic_vis")

    real_kwargs = build_real_flow_kwargs(cfg)
    real_kwargs.update(
        {
            "output_dir": real_output_dir,
            "vae_output_dir": real_vae_dir,
            "visualisations_output_dir": real_vis_dir,
            "b_filename": "b_slices.csv",
            "eps_filename": "eps_slices.csv",
            "vae_model_filename": "vae_real.pt",
        }
    )

    synthetic_kwargs = build_synthetic_flow_kwargs(cfg)
    synthetic_kwargs.update(
        {
            "output_dir": synth_output_dir,
            "vae_output_dir": synth_vae_dir,
            "visualisations_output_dir": synth_vis_dir,
            "b_filename": "b_slices.csv",
            "eps_filename": "eps_slices.csv",
            "vae_model_filename": "vae_synthetic.pt",
        }
    )

    real_result = run_real_experiment_task(real_kwargs)
    synthetic_result = run_synthetic_experiment_task(synthetic_kwargs)
    rows = build_comparison_rows_task(real_result, synthetic_result)
    summaries = write_comparison_summaries_task(
        output_dir=output_dir,
        run_id=run_id,
        cfg=cfg,
        real_kwargs=real_kwargs,
        synthetic_kwargs=synthetic_kwargs,
        real_result=real_result,
        synthetic_result=synthetic_result,
        rows=rows,
    )

    return {
        "run_id": run_id,
        "summary_csv": summaries["summary_csv"],
        "summary_json": summaries["summary_json"],
        "real_result": real_result,
        "synthetic_result": synthetic_result,
    }


if __name__ == "__main__":
    result = compare_real_vs_synthetic()
    print("Comparison complete")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Summary JSON: {result['summary_json']}")
