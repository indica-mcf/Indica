"""Prefect flow for VAE scaling study on synthetic single-slice generated data."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from prefect import flow, task

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.data_generation import (
    generate_and_save_dataset,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.components.evaluation.metrics import (
    compute_vae_diversity_and_forward_metrics,
)
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv


DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[1] / "components" / "data" / "flow_data" / "scaling_study"
)
DEFAULT_VAE_DIR = str(Path(__file__).resolve().parents[1] / "components" / "ml" / "flow_data" / "scaling_study")
DEFAULT_VIS_DIR = str(
    Path(__file__).resolve().parents[1] / "components" / "visualisations" / "outputs" / "scaling_study"
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


@task(name="generate_scaling_dataset")
def generate_scaling_dataset_task(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    n_generations: int,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
) -> dict[str, Any]:
    return generate_and_save_dataset(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=n_generations,
        use_all_timepoints=False,
        single_timepoint_mode="middle",
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        generate_new_data=True,
    )


@task(name="train_scaled_vae")
def train_scaled_vae_task(
    b_path: str,
    eps_path: str,
    hidden_scaling: int,
    latent_dim: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str,
    model_filename: str,
) -> dict[str, Any]:
    return train_vae_from_csv(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=None,
        latent_dim=latent_dim,
        hidden_scaling=hidden_scaling,
        n_epochs=n_epochs,
        lr=lr,
        train_fraction=1.0,
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        output_dir=output_dir,
        model_filename=model_filename,
    )


@task(name="evaluate_scaled_vae")
def evaluate_scaled_vae_task(
    model_path: str,
    eval_b_path: str,
    eval_eps_path: str,
    idx: int,
    k_samples: int,
) -> dict[str, Any]:
    return compute_vae_diversity_and_forward_metrics(
        model_path=model_path,
        b_path=eval_b_path,
        eps_path=eval_eps_path,
        meta_path=None,
        idx=idx,
        k_samples=k_samples,
        seed=None,
    )


@task(name="save_scaling_results")
def save_scaling_results_task(
    rows: list[dict[str, Any]],
    output_dir: str,
    filename: str,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    columns = [
        "hidden_scaling",
        "train_generations",
        "repeat_idx",
        "model_path",
        "metric_l2_sample_mean_to_true_norm",
        "metric_forward_rmse_norm",
        "last_epoch_loss",
        "last_recon_loss",
        "last_kl_loss",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return str(out_path)


@task(name="plot_scaling_results")
def plot_scaling_results_task(
    rows: list[dict[str, Any]],
    output_dir: str,
    filename: str,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        s = int(row["hidden_scaling"])
        n = int(row["train_generations"])
        y = float(row["metric_l2_sample_mean_to_true_norm"])
        if np.isfinite(y):
            grouped[s][n].append(y)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for scaling in sorted(grouped.keys()):
        ns = sorted(grouped[scaling].keys())
        means = [float(np.mean(grouped[scaling][n])) for n in ns]
        stds = [float(np.std(grouped[scaling][n])) for n in ns]
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"hidden_scaling={scaling}",
        )

    ax.set_title("VAE Scaling Study: Accuracy vs Training Data Size")
    ax.set_xlabel("Training samples (n_generations)")
    ax.set_ylabel("L2(sample_mean, true) norm (lower is better)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


@flow(name="bolometry_inversion_vae_scaling_study")
def bolometry_inversion_vae_scaling_study(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    vae_output_dir: str = DEFAULT_VAE_DIR,
    visualisations_output_dir: str = DEFAULT_VIS_DIR,
    hidden_scalings: Sequence[int] = (1, 2, 4),
    train_generations_grid: Sequence[int] = (50, 100, 200, 500, 1000),
    n_repeats: int = 10,
    eval_generations: int = 100,
    eval_b_filename: str = "b_slices_eval_fixed_100.csv",
    eval_eps_filename: str = "eps_slices_eval_fixed_100.csv",
    vae_latent_dim: int = 4,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    batch_size: int = 8,
    metrics_idx: int = 10,
    metrics_k_samples: int = 100,
    results_csv_filename: str = "vae_scaling_results.csv",
    results_plot_filename: str = "vae_scaling_accuracy_vs_data.png",
) -> dict[str, Any]:
    """Run model-width/data-size sweep with one fixed synthetic evaluation set."""
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

    eval_dataset = generate_scaling_dataset_task(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=int(eval_generations),
        output_dir=output_dir,
        b_filename=eval_b_filename,
        eps_filename=eval_eps_filename,
    )

    rows: list[dict[str, Any]] = []
    for hidden_scaling in hidden_scalings:
        for train_generations in train_generations_grid:
            for repeat_idx in range(int(n_repeats)):
                tag = f"s{int(hidden_scaling)}_n{int(train_generations)}_r{int(repeat_idx)}"
                train_dataset = generate_scaling_dataset_task(
                    machine=machine,
                    instrument=instrument,
                    transform=transform,
                    equilibrium=equilibrium,
                    n_generations=int(train_generations),
                    output_dir=output_dir,
                    b_filename=f"train_sets/b_slices_{tag}.csv",
                    eps_filename=f"train_sets/eps_slices_{tag}.csv",
                )

                vae_training = train_scaled_vae_task(
                    b_path=train_dataset["b_path"],
                    eps_path=train_dataset["eps_path"],
                    hidden_scaling=int(hidden_scaling),
                    latent_dim=vae_latent_dim,
                    n_epochs=vae_n_epochs,
                    lr=vae_lr,
                    batch_size=batch_size,
                    output_dir=vae_output_dir,
                    model_filename=f"vae_{tag}.pt",
                )

                metrics = evaluate_scaled_vae_task(
                    model_path=vae_training["model_path"],
                    eval_b_path=eval_dataset["b_path"],
                    eval_eps_path=eval_dataset["eps_path"],
                    idx=metrics_idx,
                    k_samples=metrics_k_samples,
                )

                rows.append(
                    {
                        "hidden_scaling": int(hidden_scaling),
                        "train_generations": int(train_generations),
                        "repeat_idx": int(repeat_idx),
                        "model_path": str(vae_training["model_path"]),
                        "metric_l2_sample_mean_to_true_norm": float(
                            metrics["diversity"]["l2_sample_mean_to_true_norm"]
                        ),
                        "metric_forward_rmse_norm": float(
                            metrics["forward_consistency"]["rmse_norm"]
                        ),
                        "last_epoch_loss": float(vae_training["last_epoch_loss"]),
                        "last_recon_loss": float(vae_training["last_recon_loss"]),
                        "last_kl_loss": float(vae_training["last_kl_loss"]),
                    }
                )

    results_csv_path = save_scaling_results_task(
        rows=rows,
        output_dir=output_dir,
        filename=results_csv_filename,
    )
    results_plot_path = plot_scaling_results_task(
        rows=rows,
        output_dir=visualisations_output_dir,
        filename=results_plot_filename,
    )

    return {
        "eval_dataset": eval_dataset,
        "n_rows": int(len(rows)),
        "hidden_scalings": [int(x) for x in hidden_scalings],
        "train_generations_grid": [int(x) for x in train_generations_grid],
        "n_repeats": int(n_repeats),
        "results_csv_path": results_csv_path,
        "results_plot_path": results_plot_path,
    }


if __name__ == "__main__":
    result = bolometry_inversion_vae_scaling_study()
