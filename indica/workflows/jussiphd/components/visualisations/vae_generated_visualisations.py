"""Visualisations for synthetic generated datasets (no real-pulse metadata required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from xarray import DataArray

from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset
from indica.workflows.jussiphd.los_bolometry_radiation import calculate_tomo_inversion


def _load_vae(model_path: str) -> CVAENetwork:
    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _pick_indices(n_total: int, n_pick: int) -> np.ndarray:
    if n_total <= n_pick:
        return np.arange(n_total, dtype=int)
    pos = np.linspace(0, n_total - 1, n_pick)
    return np.unique(np.round(pos).astype(int))


def _equilibrium_midpoint_time(transform: Any) -> float:
    """Return midpoint time from transform equilibrium; fallback to 0.0."""
    try:
        equilibrium = transform.equilibrium
        if hasattr(equilibrium, "t"):
            t_vals = equilibrium.t.values if hasattr(equilibrium.t, "values") else equilibrium.t
        elif hasattr(equilibrium, "rhop") and hasattr(equilibrium.rhop, "t"):
            t_vals = (
                equilibrium.rhop.t.values
                if hasattr(equilibrium.rhop.t, "values")
                else equilibrium.rhop.t
            )
        else:
            return 0.0
        t_arr = np.asarray(t_vals, dtype=float)
        t_arr = t_arr[np.isfinite(t_arr)]
        if t_arr.size == 0:
            return 0.0
        return float(0.5 * (t_arr.min() + t_arr.max()))
    except Exception:
        return 0.0


def generate_generated_dataset_visualisations(
    model_path: str,
    b_path: str,
    eps_path: str,
    transform: Any,
    output_dir: str,
    n_examples: int = 6,
    k_samples: int = 20,
    n_uncertainty_samples: int = 200,
) -> dict[str, Any]:
    """Save generated-data visualisations for VAE behavior inspection."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=None)
    model = _load_vae(model_path)
    eq_t_mid = _equilibrium_midpoint_time(transform)
    tomo_success_count = 0
    tomo_fail_count = 0
    tomo_rmse_success_count = 0
    tomo_rmse_fail_count = 0
    rmse_invalid_count = 0

    indices = _pick_indices(len(dataset), n_examples)
    rhop = np.arange(dataset.eps_slices.shape[1], dtype=float)

    # 1) Emissivity comparison with VAE samples
    n_show = len(indices)
    n_cols = 2
    n_rows = int(np.ceil(max(1, n_show) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_show:]:
        ax.axis("off")

    for ax, idx in zip(axes[:n_show], indices):
        e_norm, b_norm = dataset[idx]
        e_true = e_norm * dataset.sigma_eps + dataset.mu_eps

        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(k_samples, model.latent_dim)
            b_rep = b_t_norm.expand(k_samples, -1)
            e_samps = model.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()

        e_vae_mean = e_samps_un.mean(axis=0)

        ax.plot(rhop, e_true, color="k", linewidth=2.2, label="Ground truth")
        for i in range(k_samples):
            label = "VAE sample" if i == 0 else None
            ax.plot(rhop, e_samps_un[i], alpha=0.35, linewidth=1.0, label=label)
        ax.plot(rhop, e_vae_mean, color="tab:red", linewidth=2.2, label="VAE mean")
        ax.set_title(f"sample idx = {int(idx)}")
        ax.set_xlabel("rhop-index")
        ax.grid(alpha=0.25)

    if n_show > 0:
        axes[0].set_ylabel("emissivity")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Generated data: emissivity comparison with VAE samples", y=1.02)
    fig.tight_layout()

    emissivity_path = out_dir / "generated_emissivity_sampling.png"
    fig.savefig(emissivity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Spatial uncertainty map
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in indices:
        _, b_norm = dataset[idx]
        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            z = torch.randn(n_uncertainty_samples, model.latent_dim)
            b_rep = b_t_norm.expand(n_uncertainty_samples, -1)
            e_samps = model.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()

        e_var = e_samps_un.var(axis=0)
        ax.plot(rhop, e_var, linewidth=2, label=f"idx={int(idx)}")

    ax.set_xlabel("rhop-index")
    ax.set_ylabel("variance")
    ax.set_title("Generated data: spatial uncertainty (VAE posterior variance)")
    ax.grid(alpha=0.25)
    if len(indices) > 0:
        ax.legend(ncol=2)
    fig.tight_layout()

    uncertainty_path = out_dir / "generated_spatial_uncertainty.png"
    fig.savefig(uncertainty_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) VAE vs naive (tomographic) inversion RMSE (paired) + VAE RMSE distribution
    rhop = np.linspace(0.0, 1.0, int(dataset.eps_slices.shape[1]), dtype=np.float32)

    vae_rmse_vals = []
    naive_rmse_vals = []
    for idx in range(len(dataset)):
        e_norm, b_norm = dataset[idx]
        e_true = e_norm * dataset.sigma_eps + dataset.mu_eps

        b_true = b_norm * dataset.sigma_b + dataset.mu_b
        brightness_single = DataArray(
            np.asarray(b_true, dtype=np.float32)[None, :],
            coords=[("t", np.asarray([eq_t_mid], dtype=np.float32)), ("channel", np.arange(b_true.shape[0]))],
        )
        try:
            e_naive = (
                calculate_tomo_inversion(
                    brightness_single,
                    transform,
                    rhop,
                )
                .isel(t=0)
                .values.astype(np.float32)
            )
            tomo_success_count += 1
            tomo_rmse_success_count += 1
        except Exception:
            tomo_fail_count += 1
            tomo_rmse_fail_count += 1
            continue

        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(k_samples, model.latent_dim)
            b_rep = b_t_norm.expand(k_samples, -1)
            e_samps = model.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()
        e_vae_mean = e_samps_un.mean(axis=0)
        valid = np.isfinite(e_naive) & np.isfinite(e_true) & np.isfinite(e_vae_mean)
        if not np.any(valid):
            rmse_invalid_count += 1
            continue
        vae_rmse_vals.append(float(np.sqrt(np.mean((e_vae_mean[valid] - e_true[valid]) ** 2))))
        naive_rmse_vals.append(float(np.sqrt(np.mean((e_naive[valid] - e_true[valid]) ** 2))))

    vae_rmse_vals = np.asarray(vae_rmse_vals, dtype=float)
    naive_rmse_vals = np.asarray(naive_rmse_vals, dtype=float)

    naive_vs_vae_path = out_dir / "generated_vae_vs_naive_rmse_scatter.png"
    if len(vae_rmse_vals) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(naive_rmse_vals, vae_rmse_vals, alpha=0.7, s=30)
        mn = float(min(naive_rmse_vals.min(), vae_rmse_vals.min()))
        mx = float(max(naive_rmse_vals.max(), vae_rmse_vals.max()))
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y = x")
        ax.set_xlabel("Naive inversion RMSE")
        ax.set_ylabel("VAE RMSE")
        ax.set_title(f"Generated data RMSE comparison over {len(vae_rmse_vals)} samples")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()

        delta = naive_rmse_vals - vae_rmse_vals
        ax.text(
            0.02,
            0.98,
            f"VAE wins: {100.0 * np.mean(delta > 0):.1f}%\nmedian Δ={np.median(delta):.4f}",
            transform=ax.transAxes,
            va="top",
        )

        fig.savefig(naive_vs_vae_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.text(
            0.5,
            0.5,
            "No valid naive inversion pairs\nfor RMSE scatter",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(naive_vs_vae_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4) Ground truth vs naive inversion vs VAE mean for selected slices
    n_show = len(indices)
    n_cols = 2
    n_rows = int(np.ceil(max(1, n_show) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4.5 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_show:]:
        ax.axis("off")

    for ax, idx in zip(axes[:n_show], indices):
        e_norm, b_norm = dataset[idx]
        e_true = e_norm * dataset.sigma_eps + dataset.mu_eps
        b_true = b_norm * dataset.sigma_b + dataset.mu_b
        brightness_single = DataArray(
            np.asarray(b_true, dtype=np.float32)[None, :],
            coords=[("t", np.asarray([eq_t_mid], dtype=np.float32)), ("channel", np.arange(b_true.shape[0]))],
        )
        try:
            e_naive = (
                calculate_tomo_inversion(
                    brightness_single,
                    transform,
                    rhop,
                )
                .isel(t=0)
                .values.astype(np.float32)
            )
            tomo_success_count += 1
        except Exception:
            tomo_fail_count += 1
            continue

        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(k_samples, model.latent_dim)
            b_rep = b_t_norm.expand(k_samples, -1)
            e_samps = model.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()
        e_vae_mean = e_samps_un.mean(axis=0)
        valid = np.isfinite(e_naive) & np.isfinite(e_true) & np.isfinite(e_vae_mean)
        if not np.any(valid):
            continue

        ax.plot(rhop[valid], e_true[valid], color="k", linewidth=2.2, label="Ground truth")
        ax.plot(rhop[valid], e_naive[valid], color="tab:blue", linewidth=2.0, label="Naive inversion")
        ax.plot(rhop[valid], e_vae_mean[valid], color="tab:red", linewidth=2.0, label="VAE mean")
        ax.set_title(f"sample idx = {int(idx)}")
        ax.set_xlabel("rhop-index")
        ax.grid(alpha=0.25)

    if n_show > 0:
        axes[0].set_ylabel("emissivity")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Generated data: ground truth vs naive inversion vs VAE", y=1.02)
    fig.tight_layout()

    comparison_path = out_dir / "generated_truth_naive_vae_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) VAE RMSE distribution
    rmse_vals = vae_rmse_vals
    rmse_path = out_dir / "generated_vae_rmse_distribution.png"
    if len(rmse_vals) > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(rmse_vals, bins=30, alpha=0.8)
        ax.axvline(rmse_vals.mean(), color="tab:red", linestyle="--", label=f"mean={rmse_vals.mean():.4f}")
        ax.set_title("Generated data: VAE emissivity RMSE distribution")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(rmse_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "saved_files": {
            "emissivity_sampling": str(emissivity_path),
            "spatial_uncertainty": str(uncertainty_path),
            "vae_vs_naive_rmse_scatter": str(naive_vs_vae_path),
            "truth_naive_vae_comparison": str(comparison_path),
            "rmse_distribution": str(rmse_path),
        },
        "num_samples": int(len(dataset)),
        "num_example_indices": int(len(indices)),
        "baseline_method": "calculate_tomo_inversion",
        "equilibrium_midpoint_time_used": float(eq_t_mid),
        "num_tomo_success": int(tomo_success_count),
        "num_tomo_fail": int(tomo_fail_count),
        "num_tomo_rmse_success": int(tomo_rmse_success_count),
        "num_tomo_rmse_fail": int(tomo_rmse_fail_count),
        "num_rmse_pairs": int(len(vae_rmse_vals)),
        "num_rmse_invalid": int(rmse_invalid_count),
    }


def generate_vae_training_progress_visualisation(
    model_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Save VAE training-progress visualisation from checkpoint history."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location="cpu")
    history = ckpt.get("training_history", {})
    epochs = np.asarray(history.get("epoch", []), dtype=int)
    total_loss = np.asarray(history.get("total_loss", []), dtype=float)
    recon_loss = np.asarray(history.get("recon_loss", []), dtype=float)
    kl_loss = np.asarray(history.get("kl_loss", []), dtype=float)

    if epochs.size == 0:
        return {
            "saved_files": {},
            "status": "no_training_history_in_checkpoint",
            "model_path": model_path,
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, total_loss, linewidth=2.2, label="Total loss")
    if recon_loss.size == epochs.size:
        ax.plot(epochs, recon_loss, linewidth=1.8, label="Recon loss")
    if kl_loss.size == epochs.size:
        ax.plot(epochs, kl_loss, linewidth=1.8, label="KL loss (beta-weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("VAE training progress")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    progress_path = out_dir / "generated_vae_training_progress.png"
    fig.savefig(progress_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "saved_files": {"training_progress": str(progress_path)},
        "status": "ok",
        "n_epochs_logged": int(epochs.size),
    }
