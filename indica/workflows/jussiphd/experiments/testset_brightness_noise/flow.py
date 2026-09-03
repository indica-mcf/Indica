"""Prefect flow: synthetic training with brightness noise applied only to test set."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from prefect import flow, task
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.workflows.jussiphd.components.data.data_generation import (
    generate_and_save_dataset,
)
from indica.workflows.jussiphd.components.data.noise_injection import (
    add_poisson_noise_to_b_csv,
)
from indica.workflows.jussiphd.components.data.real_equilibrium import (
    load_real_equilibrium_from_pulse,
)
from indica.workflows.jussiphd.components.ml.vae import CVAENetwork, train_vae_from_csv
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    generate_generated_dataset_visualisations,
    generate_vae_training_progress_visualisation,
)


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def _load_csv_2d(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.asarray(arr, dtype=np.float32)


def _save_csv_2d(path: str, arr: np.ndarray) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, np.asarray(arr, dtype=np.float32), delimiter=",")
    return str(out_path)


def _load_vae(model_path: str) -> CVAENetwork:
    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_scaling=int(ckpt.get("hidden_scaling", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


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


@task(name="generate_full_synthetic_dataset")
def generate_full_synthetic_dataset_task(
    machine: str,
    instrument: str,
    transform: Any,
    equilibrium: Any,
    n_generations: int,
    output_dir: str,
    b_filename: str,
    eps_filename: str,
    generate_new_data: bool,
) -> dict[str, Any]:
    return generate_and_save_dataset(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=n_generations,
        use_all_timepoints=True,
        single_timepoint_mode="middle",
        output_dir=output_dir,
        b_filename=b_filename,
        eps_filename=eps_filename,
        generate_new_data=generate_new_data,
    )


@task(name="split_train_test")
def split_train_test_task(
    b_path: str,
    eps_path: str,
    output_dir: str,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    b = _load_csv_2d(b_path)
    eps = _load_csv_2d(eps_path)
    if len(b) != len(eps):
        raise ValueError("b and eps row counts differ.")

    n = len(b)
    n_test = max(1, int(round(float(test_fraction) * n)))
    n_test = min(n_test, n - 1) if n > 1 else 1
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    out = Path(output_dir)
    train_b_path = _save_csv_2d(str(out / "train_b.csv"), b[train_idx])
    train_eps_path = _save_csv_2d(str(out / "train_eps.csv"), eps[train_idx])
    test_b_path = _save_csv_2d(str(out / "test_b.csv"), b[test_idx])
    test_eps_path = _save_csv_2d(str(out / "test_eps.csv"), eps[test_idx])

    return {
        "train_b_path": train_b_path,
        "train_eps_path": train_eps_path,
        "test_b_path": test_b_path,
        "test_eps_path": test_eps_path,
        "num_total": int(n),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
    }


@task(name="noise_test_brightness")
def noise_test_brightness_task(
    test_b_path: str,
    count_level: float,
    scale_percentile: float,
    seed: int,
) -> dict[str, Any]:
    out_path = str(Path(test_b_path).with_name(Path(test_b_path).stem + "_poisson.csv"))
    return add_poisson_noise_to_b_csv(
        b_path=test_b_path,
        count_level=count_level,
        output_path=out_path,
        scale_percentile=scale_percentile,
        seed=seed,
    )


@task(name="train_vae_on_clean_train")
def train_vae_on_clean_train_task(
    train_b_path: str,
    train_eps_path: str,
    latent_dim: int,
    hidden_scaling: int,
    n_epochs: int,
    lr: float,
    kl_scaling: float,
    batch_size: int,
    output_dir: str,
    model_filename: str,
) -> dict[str, Any]:
    return train_vae_from_csv(
        b_path=train_b_path,
        eps_path=train_eps_path,
        meta_path=None,
        latent_dim=latent_dim,
        hidden_scaling=hidden_scaling,
        n_epochs=n_epochs,
        lr=lr,
        kl_scaling=kl_scaling,
        train_fraction=1.0,
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        output_dir=output_dir,
        model_filename=model_filename,
    )


@task(name="evaluate_clean_vs_noisy_test")
def evaluate_clean_vs_noisy_test_task(
    model_path: str,
    train_b_path: str,
    train_eps_path: str,
    test_b_clean_path: str,
    test_b_noisy_path: str,
    test_eps_path: str,
    k_samples: int,
    output_dir: str,
) -> dict[str, Any]:
    model = _load_vae(model_path)

    train_b = _load_csv_2d(train_b_path)
    train_eps = _load_csv_2d(train_eps_path)
    test_b_clean = _load_csv_2d(test_b_clean_path)
    test_b_noisy = _load_csv_2d(test_b_noisy_path)
    test_eps = _load_csv_2d(test_eps_path)

    mu_b = float(np.mean(train_b))
    sigma_b = float(np.std(train_b))
    mu_eps = float(np.mean(train_eps))
    sigma_eps = float(np.std(train_eps))
    sigma_b = sigma_b if sigma_b > 0 else 1.0
    sigma_eps = sigma_eps if sigma_eps > 0 else 1.0

    n = len(test_eps)
    if len(test_b_clean) != n or len(test_b_noisy) != n:
        raise ValueError("Test set row mismatch between b_clean, b_noisy and eps.")

    rows: list[dict[str, Any]] = []
    rmse_clean: list[float] = []
    rmse_noisy: list[float] = []

    for i in range(n):
        e_true = test_eps[i].astype(np.float32)
        b_clean_n = ((test_b_clean[i].astype(np.float32) - mu_b) / sigma_b)[None, :]
        b_noisy_n = ((test_b_noisy[i].astype(np.float32) - mu_b) / sigma_b)[None, :]

        with torch.no_grad():
            z = torch.randn(int(k_samples), model.latent_dim)

            b_rep_clean = torch.from_numpy(b_clean_n).expand(int(k_samples), -1)
            e_clean_n = model.decode(b_rep_clean, z).mean(dim=0).cpu().numpy()
            e_clean = e_clean_n * sigma_eps + mu_eps

            b_rep_noisy = torch.from_numpy(b_noisy_n).expand(int(k_samples), -1)
            e_noisy_n = model.decode(b_rep_noisy, z).mean(dim=0).cpu().numpy()
            e_noisy = e_noisy_n * sigma_eps + mu_eps

        r_clean = float(np.sqrt(np.mean((e_clean - e_true) ** 2)))
        r_noisy = float(np.sqrt(np.mean((e_noisy - e_true) ** 2)))
        rmse_clean.append(r_clean)
        rmse_noisy.append(r_noisy)
        rows.append(
            {
                "sample_idx": int(i),
                "rmse_clean_test_b": r_clean,
                "rmse_noisy_test_b": r_noisy,
                "delta_noisy_minus_clean": float(r_noisy - r_clean),
            }
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "clean_vs_noisy_test_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "rmse_clean_test_b", "rmse_noisy_test_b", "delta_noisy_minus_clean"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.boxplot(
        [np.asarray(rmse_clean, dtype=float), np.asarray(rmse_noisy, dtype=float)],
        labels=["Clean test b", "Noisy test b"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    ax.set_title("Test-set RMSE: clean vs noisy brightness input")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plot_path = out / "clean_vs_noisy_test_rmse_boxplot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "metrics_csv_path": str(csv_path),
        "metrics_plot_path": str(plot_path),
        "rmse_clean_mean": float(np.mean(rmse_clean)) if rmse_clean else np.nan,
        "rmse_noisy_mean": float(np.mean(rmse_noisy)) if rmse_noisy else np.nan,
        "rmse_clean_median": float(np.median(rmse_clean)) if rmse_clean else np.nan,
        "rmse_noisy_median": float(np.median(rmse_noisy)) if rmse_noisy else np.nan,
    }


@task(name="generate_noisy_test_generated_visualisations")
def generate_noisy_test_generated_visualisations_task(
    model_path: str,
    b_path: str,
    eps_path: str,
    transform: Any,
    output_dir: str,
    n_examples: int,
    k_samples: int,
    n_uncertainty_samples: int,
) -> dict[str, Any]:
    return generate_generated_dataset_visualisations(
        model_path=model_path,
        b_path=b_path,
        eps_path=eps_path,
        transform=transform,
        output_dir=output_dir,
        n_examples=n_examples,
        k_samples=k_samples,
        n_uncertainty_samples=n_uncertainty_samples,
    )


@task(name="generate_noisy_test_vae_training_progress_visualisation")
def generate_noisy_test_vae_training_progress_visualisation_task(
    model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    return generate_vae_training_progress_visualisation(
        model_path=model_path,
        output_dir=output_dir,
    )


@task(name="visualise_noisy_b_only")
def visualise_noisy_b_only_task(
    noisy_b_path: str,
    output_dir: str,
    n_examples: int,
) -> dict[str, Any]:
    b_noisy = _load_csv_2d(noisy_b_path)
    n = len(b_noisy)
    if n == 0:
        raise ValueError("No noisy test-b samples to visualise.")
    idx = np.unique(np.round(np.linspace(0, n - 1, min(max(1, n_examples), n))).astype(int))

    n_cols = 2
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    channels = np.arange(b_noisy.shape[1], dtype=int)

    for ax in axes[len(idx):]:
        ax.axis("off")
    for ax, i in zip(axes[: len(idx)], idx):
        ax.plot(channels, b_noisy[int(i)], linewidth=1.8, color="tab:orange")
        ax.set_title(f"Noisy test-b sample idx={int(i)}")
        ax.set_xlabel("channel")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.25)

    fig.suptitle("Noisy test-set brightness samples", y=1.02)
    fig.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "test_b_noisy_gallery.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"plot_path": str(plot_path), "num_examples": int(len(idx))}


@task(name="visualise_noisy_b_vs_true_e")
def visualise_noisy_b_vs_true_e_task(
    noisy_b_path: str,
    true_eps_path: str,
    output_dir: str,
    n_examples: int,
) -> dict[str, Any]:
    b_noisy = _load_csv_2d(noisy_b_path)
    eps_true = _load_csv_2d(true_eps_path)
    if b_noisy.shape[0] != eps_true.shape[0]:
        raise ValueError(
            f"Row mismatch noisy b vs true eps: {b_noisy.shape} vs {eps_true.shape}"
        )
    n = len(b_noisy)
    if n == 0:
        raise ValueError("No samples available for noisy-b vs true-e visualisation.")

    idx = np.unique(np.round(np.linspace(0, n - 1, min(max(1, n_examples), n))).astype(int))
    n_cols = 2
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    x_b = np.linspace(0.0, 1.0, b_noisy.shape[1], dtype=np.float32)
    x_e = np.linspace(0.0, 1.0, eps_true.shape[1], dtype=np.float32)

    def _normalize_curve(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        lo = float(np.min(y))
        hi = float(np.max(y))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(y, dtype=np.float32)
        return (y - lo) / (hi - lo)

    for ax in axes[len(idx):]:
        ax.axis("off")
    for ax, i in zip(axes[: len(idx)], idx):
        b_norm = _normalize_curve(b_noisy[int(i)])
        e_norm = _normalize_curve(eps_true[int(i)])
        ax.plot(x_b, b_norm, linewidth=1.9, color="tab:orange", label="noisy b (normalized)")
        ax.plot(x_e, e_norm, linewidth=1.9, color="black", label="ground truth e (normalized)")
        ax.set_title(f"Noisy b vs true e idx={int(i)}")
        ax.set_xlabel("normalized position (0..1)")
        ax.set_ylabel("normalized amplitude")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Shape comparison: noisy test-b vs ground-truth emissivity", y=1.02)
    fig.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "test_b_noisy_vs_true_e_overlay.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"plot_path": str(plot_path), "num_examples": int(len(idx))}


@task(name="visualise_noisy_b_vs_vae_forward_b_by_channel")
def visualise_noisy_b_vs_vae_forward_b_by_channel_task(
    model_path: str,
    train_b_path: str,
    train_eps_path: str,
    noisy_b_path: str,
    transform: Any,
    output_dir: str,
    n_examples: int,
    k_samples: int,
) -> dict[str, Any]:
    """Compare noisy input b vs VAE->forward b as profile plots per selected sample."""
    model = _load_vae(model_path)
    train_b = _load_csv_2d(train_b_path)
    train_eps = _load_csv_2d(train_eps_path)
    noisy_b = _load_csv_2d(noisy_b_path)

    mu_b = float(np.mean(train_b))
    sigma_b = float(np.std(train_b))
    mu_eps = float(np.mean(train_eps))
    sigma_eps = float(np.std(train_eps))
    sigma_b = sigma_b if sigma_b > 0 else 1.0
    sigma_eps = sigma_eps if sigma_eps > 0 else 1.0

    n = len(noisy_b)
    if n == 0:
        raise ValueError("No noisy-b samples for VAE/forward comparison.")
    idx = np.unique(np.round(np.linspace(0, n - 1, min(max(1, n_examples), n))).astype(int))

    eq_t_mid = 0.0
    try:
        if hasattr(transform, "equilibrium") and hasattr(transform.equilibrium, "t"):
            t_arr = np.asarray(transform.equilibrium.t, dtype=float).reshape(-1)
            t_arr = t_arr[np.isfinite(t_arr)]
            if t_arr.size > 0:
                eq_t_mid = float(0.5 * (t_arr.min() + t_arr.max()))
    except Exception:
        eq_t_mid = 0.0

    b_forward = []
    for i in idx:
        b_noisy_n = ((noisy_b[int(i)].astype(np.float32) - mu_b) / sigma_b)[None, :]
        with torch.no_grad():
            z = torch.randn(int(k_samples), model.latent_dim)
            b_rep = torch.from_numpy(b_noisy_n).expand(int(k_samples), -1)
            e_pred_n = model.decode(b_rep, z).mean(dim=0).cpu().numpy().astype(np.float32)
        e_pred = e_pred_n * sigma_eps + mu_eps

        e_da = DataArray(
            np.asarray(e_pred, dtype=np.float32)[None, :],
            coords=[("t", np.asarray([eq_t_mid], dtype=np.float32)), ("rhop", np.linspace(0.0, 1.0, e_pred.shape[0]))],
        )
        b_fw_da = transform.integrate_on_los(e_da, t=e_da.t)
        b_fw = np.asarray(b_fw_da.values, dtype=np.float32).reshape(-1)
        b_forward.append(b_fw)

    b_forward = np.asarray(b_forward, dtype=np.float32)  # [n_examples, n_channels]
    b_noisy_pick = np.asarray([noisy_b[int(i)] for i in idx], dtype=np.float32)
    if b_noisy_pick.shape != b_forward.shape:
        raise ValueError(
            f"Noisy-vs-forward shape mismatch: noisy={b_noisy_pick.shape}, forward={b_forward.shape}"
        )

    n_channels = int(b_noisy_pick.shape[1])
    n_cols = 2
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    channels = np.arange(n_channels, dtype=int)

    for ax in axes[len(idx):]:
        ax.axis("off")

    for ax, i, b_noisy_sample, b_forward_sample in zip(
        axes[: len(idx)],
        idx,
        b_noisy_pick,
        b_forward,
    ):
        ax.plot(channels, b_noisy_sample, linewidth=1.8, color="tab:orange", label="noisy b")
        ax.plot(channels, b_forward_sample, linewidth=1.8, color="tab:blue", label="VAE->forward b")
        ax.set_title(f"Sample idx={int(i)}")
        ax.set_xlabel("channel")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Profile comparison per sample: noisy input b vs VAE->forward b", y=1.01)
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "test_b_noisy_vs_vae_forward_by_sample.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"plot_path": str(plot_path), "num_examples": int(len(idx)), "n_channels": n_channels}


@flow(name="synthetic_testset_brightness_noise")
def synthetic_testset_brightness_noise(
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    use_real_equilibrium: bool = True,
    real_equilibrium_pulse: int = 13622,
    real_equilibrium_verbose: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dataset_b_filename: str = "full_synthetic_b.csv",
    dataset_eps_filename: str = "full_synthetic_eps.csv",
    n_generations: int = 3000,
    generate_new_data: bool = False,
    test_fraction: float = 0.2,
    split_seed: int = 0,
    noise_count_level: float = 200.0,
    noise_scale_percentile: float = 99.0,
    noise_seed: int = 0,
    vae_output_dir: str = DEFAULT_OUTPUT_DIR,
    vae_model_filename: str = "vae_full_synthetic_clean_train.pt",
    vae_latent_dim: int = 4,
    vae_hidden_scaling: int = 8,
    vae_n_epochs: int = 25,
    vae_lr: float = 1e-3,
    vae_kl_scaling: float = 0.2,
    batch_size: int = 8,
    eval_k_samples: int = 20,
    run_visualisations: bool = True,
    visualisations_output_dir: str = DEFAULT_OUTPUT_DIR,
    visualisations_n_examples: int = 6,
    visualisations_k_samples: int = 20,
    visualisations_n_uncertainty_samples: int = 200,
) -> dict[str, Any]:
    """Generate synthetic data, noise only test-set b, and compare evaluation RMSE."""
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

    dataset = generate_full_synthetic_dataset_task(
        machine=machine,
        instrument=instrument,
        transform=transform,
        equilibrium=equilibrium,
        n_generations=n_generations,
        output_dir=output_dir,
        b_filename=dataset_b_filename,
        eps_filename=dataset_eps_filename,
        generate_new_data=generate_new_data,
    )

    split = split_train_test_task(
        b_path=dataset["b_path"],
        eps_path=dataset["eps_path"],
        output_dir=str(Path(output_dir) / "split"),
        test_fraction=test_fraction,
        seed=split_seed,
    )
    noisy_b = noise_test_brightness_task(
        test_b_path=split["test_b_path"],
        count_level=noise_count_level,
        scale_percentile=noise_scale_percentile,
        seed=noise_seed,
    )
    train = train_vae_on_clean_train_task(
        train_b_path=split["train_b_path"],
        train_eps_path=split["train_eps_path"],
        latent_dim=vae_latent_dim,
        hidden_scaling=vae_hidden_scaling,
        n_epochs=vae_n_epochs,
        lr=vae_lr,
        kl_scaling=vae_kl_scaling,
        batch_size=batch_size,
        output_dir=vae_output_dir,
        model_filename=vae_model_filename,
    )
    eval_result = evaluate_clean_vs_noisy_test_task(
        model_path=train["model_path"],
        train_b_path=split["train_b_path"],
        train_eps_path=split["train_eps_path"],
        test_b_clean_path=split["test_b_path"],
        test_b_noisy_path=noisy_b["output_b_path"],
        test_eps_path=split["test_eps_path"],
        k_samples=eval_k_samples,
        output_dir=str(Path(output_dir) / "evaluation"),
    )

    visualisations = None
    if run_visualisations:
        noisy_b_only_visualisation = visualise_noisy_b_only_task(
            noisy_b_path=noisy_b["output_b_path"],
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
        )
        noisy_b_vs_true_e_visualisation = visualise_noisy_b_vs_true_e_task(
            noisy_b_path=noisy_b["output_b_path"],
            true_eps_path=split["test_eps_path"],
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
        )
        noisy_b_vs_vae_forward_b_visualisation = visualise_noisy_b_vs_vae_forward_b_by_channel_task(
            model_path=train["model_path"],
            train_b_path=split["train_b_path"],
            train_eps_path=split["train_eps_path"],
            noisy_b_path=noisy_b["output_b_path"],
            transform=transform,
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
            k_samples=visualisations_k_samples,
        )
        generated_visualisations = generate_noisy_test_generated_visualisations_task(
            model_path=train["model_path"],
            b_path=noisy_b["output_b_path"],
            eps_path=split["test_eps_path"],
            transform=transform,
            output_dir=visualisations_output_dir,
            n_examples=visualisations_n_examples,
            k_samples=visualisations_k_samples,
            n_uncertainty_samples=visualisations_n_uncertainty_samples,
        )
        training_progress_visualisation = generate_noisy_test_vae_training_progress_visualisation_task(
            model_path=train["model_path"],
            output_dir=visualisations_output_dir,
        )
        visualisations = {
            "noisy_b_only_visualisation": noisy_b_only_visualisation,
            "noisy_b_vs_true_e_visualisation": noisy_b_vs_true_e_visualisation,
            "noisy_b_vs_vae_forward_b_visualisation": noisy_b_vs_vae_forward_b_visualisation,
            "generated_dataset_visualisations": generated_visualisations,
            "training_progress_visualisation": training_progress_visualisation,
        }

    return {
        "dataset": dataset,
        "split": split,
        "noisy_test_b": noisy_b,
        "vae_training": train,
        "evaluation": eval_result,
        "visualisations": visualisations,
    }


if __name__ == "__main__":
    result = synthetic_testset_brightness_noise()
    print("Synthetic test-set brightness-noise experiment complete")
    print(f"Metrics CSV: {result['evaluation']['metrics_csv_path']}")
