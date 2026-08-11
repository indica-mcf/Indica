"""Prefect flow for VAE scaling study on synthetic generated data."""

from __future__ import annotations

import csv
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)
from indica.workflows.jussiphd.components.ml.vae import train_vae_from_csv
from indica.workflows.jussiphd.datasets.paths import (
    MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR,
)


DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parents[2] / "components" / "data" / "flow_data" / "scaling_study"
)
DEFAULT_VAE_DIR = str(Path(__file__).resolve().parents[2] / "components" / "ml" / "flow_data" / "scaling_study")
DEFAULT_VIS_DIR = str(Path(__file__).resolve().parent / "outputs")
DEFAULT_CONSTANT_IMP_EXPANDED_EQ_DIR = MULTIPULSE_SYNTHETIC_EXPANDED_EQUILIBRIA_CONSTANT_IMP_DATA_DIR_STR


def _load_csv_2d(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _save_csv_2d(path: str, arr: np.ndarray) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.asarray(arr, dtype=np.float32), delimiter=",")
    return str(out_path)


def _load_results_rows(path: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _poisson_noise_array(
    values: np.ndarray,
    count_level: float,
    scale_percentile: float,
    seed: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    scale_value = float(np.percentile(np.clip(arr, a_min=0.0, a_max=None), float(scale_percentile)))
    if scale_value <= 0:
        scale_value = 1.0
    rng = np.random.default_rng(int(seed))
    return add_poisson_noise_with_counts(
        values=arr,
        count_level=float(count_level),
        scale_value=scale_value,
        rng=rng,
    ).astype(np.float32)


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
        "num_parameters",
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
    x_log_scale: bool = False,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    params_by_scaling: dict[int, int] = {}
    for row in rows:
        s = int(row["hidden_scaling"])
        if "num_parameters" in row and row["num_parameters"] not in (None, ""):
            params_by_scaling[s] = int(row["num_parameters"])
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
            label=(
                f"{params_by_scaling[scaling] / 1000.0:.0f}k params"
                if scaling in params_by_scaling
                else f"hidden_scaling={scaling}"
            ),
        )

    ax.set_title("VAE Scaling Study: Accuracy vs Training Data Size")
    ax.set_xlabel("Training samples (n_generations)")
    ax.set_ylabel("L2(sample_mean, true) norm (lower is better)")
    if x_log_scale:
        ax.set_xscale("log")
        # Keep labels human-readable for common sample counts.
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        unique_ns = sorted({int(row["train_generations"]) for row in rows})
        if unique_ns:
            ax.set_xticks(unique_ns)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def replot_scaling_results_from_pickle(
    pickle_path: str = "scaling_results.pickle",
    output_plot_filename: str = "vae_scaling_accuracy_vs_data_logx.png",
    x_log_scale: bool = True,
) -> str:
    with Path(pickle_path).open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Expected a dict in scaling_results pickle")
    results_csv_path = result.get("results_csv_path")
    if not results_csv_path:
        raise ValueError("results_csv_path missing from scaling_results pickle")
    rows = _load_results_rows(str(results_csv_path))
    results_plot_path = result.get("results_plot_path")
    output_dir = (
        str(Path(results_plot_path).parent)
        if results_plot_path
        else DEFAULT_VIS_DIR
    )
    return plot_scaling_results_task.fn(
        rows=rows,
        output_dir=output_dir,
        filename=output_plot_filename,
        x_log_scale=bool(x_log_scale),
    )


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
    use_existing_single_generated_pool: bool = True,
    pool_b_path: str = str(
        Path(DEFAULT_CONSTANT_IMP_EXPANDED_EQ_DIR)
        / "b_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv"
    ),
    pool_eps_path: str = str(
        Path(DEFAULT_CONSTANT_IMP_EXPANDED_EQ_DIR)
        / "eps_slices_multipulse_synthetic_expanded_equilibria_constant_imp.csv"
    ),
    apply_b_poisson_noise: bool = True,
    b_poisson_count_level: float = 200.0,
    b_poisson_scale_percentile: float = 99.0,
    b_poisson_seed: int = 1,
    apply_eps_poisson_noise: bool = False,
    eps_poisson_count_level: float = 200.0,
    eps_poisson_scale_percentile: float = 99.0,
    eps_poisson_seed: int = 0,
    hidden_scalings: Sequence[int] = (1, 2, 3, 4),
    train_generations_grid: Sequence[int] = (50, 200, 700, 2500, 10000, 25000 ),

    n_repeats: int = 5,
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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    b_pool: np.ndarray | None = None
    eps_pool: np.ndarray | None = None

    if use_existing_single_generated_pool:
        b_pool = _load_csv_2d(pool_b_path)
        eps_pool = _load_csv_2d(pool_eps_path)
        if apply_b_poisson_noise:
            b_pool = _poisson_noise_array(
                values=b_pool,
                count_level=b_poisson_count_level,
                scale_percentile=b_poisson_scale_percentile,
                seed=b_poisson_seed,
            )
        if apply_eps_poisson_noise:
            eps_pool = _poisson_noise_array(
                values=eps_pool,
                count_level=eps_poisson_count_level,
                scale_percentile=eps_poisson_scale_percentile,
                seed=eps_poisson_seed,
            )
        if len(b_pool) != len(eps_pool):
            raise ValueError("pool_b_path and pool_eps_path must have the same number of rows")
        max_train = int(max(int(n) for n in train_generations_grid))
        n_pool = int(len(b_pool))
        if n_pool < int(eval_generations) + max_train:
            raise ValueError(
                "Existing pool is too small for fixed eval split + largest train size: "
                f"pool={n_pool}, eval={int(eval_generations)}, max_train={max_train}"
            )

        perm = np.random.permutation(n_pool)
        eval_indices = perm[: int(eval_generations)]
        train_pool_indices = perm[int(eval_generations) :]

        eval_b_path = _save_csv_2d(str(out_dir / eval_b_filename), b_pool[eval_indices])
        eval_eps_values = eps_pool[eval_indices]
        eval_eps_path = _save_csv_2d(str(out_dir / eval_eps_filename), eval_eps_values)
        eval_dataset = {
            "b_path": eval_b_path,
            "eps_path": eval_eps_path,
            "num_pairs": int(len(eval_indices)),
            "generated_new_data": False,
            "source": "existing_single_generated_pool",
            "pool_size": n_pool,
        }
    else:
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
        if apply_b_poisson_noise:
            eval_b = _load_csv_2d(eval_dataset["b_path"])
            eval_b_noisy = _poisson_noise_array(
                values=eval_b,
                count_level=b_poisson_count_level,
                scale_percentile=b_poisson_scale_percentile,
                seed=b_poisson_seed,
            )
            eval_dataset["b_path"] = _save_csv_2d(
                str(Path(eval_dataset["b_path"]).with_name(
                    f"{Path(eval_dataset['b_path']).stem}_poisson{int(b_poisson_count_level)}.csv"
                )),
                eval_b_noisy,
            )
        if apply_eps_poisson_noise:
            eval_eps = _load_csv_2d(eval_dataset["eps_path"])
            eval_eps_noisy = _poisson_noise_array(
                values=eval_eps,
                count_level=eps_poisson_count_level,
                scale_percentile=eps_poisson_scale_percentile,
                seed=eps_poisson_seed,
            )
            eval_dataset["eps_path"] = _save_csv_2d(
                str(Path(eval_dataset["eps_path"]).with_name(
                    f"{Path(eval_dataset['eps_path']).stem}_poisson{int(eps_poisson_count_level)}.csv"
                )),
                eval_eps_noisy,
            )
        train_pool_indices = None

    rows: list[dict[str, Any]] = []
    for hidden_scaling in hidden_scalings:
        for train_generations in train_generations_grid:
            for repeat_idx in range(int(n_repeats)):
                tag = f"s{int(hidden_scaling)}_n{int(train_generations)}_r{int(repeat_idx)}"
                if use_existing_single_generated_pool:
                    assert b_pool is not None and eps_pool is not None and train_pool_indices is not None
                    pick = np.random.choice(
                        train_pool_indices,
                        size=int(train_generations),
                        replace=False,
                    )
                    train_b_path = _save_csv_2d(
                        str(out_dir / f"train_sets/b_slices_{tag}.csv"),
                        b_pool[pick],
                    )
                    train_eps_path = _save_csv_2d(
                        str(out_dir / f"train_sets/eps_slices_{tag}.csv"),
                        eps_pool[pick],
                    )
                    train_dataset = {
                        "b_path": train_b_path,
                        "eps_path": train_eps_path,
                        "num_pairs": int(len(pick)),
                        "generated_new_data": False,
                    }
                else:
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
                    if apply_b_poisson_noise:
                        train_b = _load_csv_2d(train_dataset["b_path"])
                        train_b_noisy = _poisson_noise_array(
                            values=train_b,
                            count_level=b_poisson_count_level,
                            scale_percentile=b_poisson_scale_percentile,
                            seed=b_poisson_seed + int(repeat_idx) + int(hidden_scaling) + int(train_generations),
                        )
                        train_dataset["b_path"] = _save_csv_2d(
                            str(Path(train_dataset["b_path"]).with_name(
                                f"{Path(train_dataset['b_path']).stem}_poisson{int(b_poisson_count_level)}.csv"
                            )),
                            train_b_noisy,
                        )
                    if apply_eps_poisson_noise:
                        train_eps = _load_csv_2d(train_dataset["eps_path"])
                        train_eps_noisy = _poisson_noise_array(
                            values=train_eps,
                            count_level=eps_poisson_count_level,
                            scale_percentile=eps_poisson_scale_percentile,
                            seed=eps_poisson_seed + int(repeat_idx) + int(hidden_scaling) + int(train_generations),
                        )
                        train_dataset["eps_path"] = _save_csv_2d(
                            str(Path(train_dataset["eps_path"]).with_name(
                                f"{Path(train_dataset['eps_path']).stem}_poisson{int(eps_poisson_count_level)}.csv"
                            )),
                            train_eps_noisy,
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
                        "num_parameters": int(vae_training.get("num_parameters", 0)),
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
        "use_existing_single_generated_pool": bool(use_existing_single_generated_pool),
        "apply_b_poisson_noise": bool(apply_b_poisson_noise),
        "b_poisson_count_level": float(b_poisson_count_level),
        "apply_eps_poisson_noise": bool(apply_eps_poisson_noise),
        "eps_poisson_count_level": float(eps_poisson_count_level),
        "pool_b_path": pool_b_path if use_existing_single_generated_pool else None,
        "pool_eps_path": pool_eps_path if use_existing_single_generated_pool else None,
        "hidden_scalings": [int(x) for x in hidden_scalings],
        "train_generations_grid": [int(x) for x in train_generations_grid],
        "n_repeats": int(n_repeats),
        "results_csv_path": results_csv_path,
        "results_plot_path": results_plot_path,
    }


if __name__ == "__main__":
    result = bolometry_inversion_vae_scaling_study()
    import pickle


    with open('scaling_results.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
