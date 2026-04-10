"""VAE diversity and forward-consistency metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset


def compute_vae_diversity_and_forward_metrics(
    model_path: str,
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    idx: int = 10,
    k_samples: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute notebook-style conditional diversity and forward consistency metrics."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=meta_path)
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")
    idx = int(np.clip(idx, 0, n - 1))

    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
        hidden_scaling=int(ckpt.get("hidden_scaling", 1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    e_true_np, b_star_np = dataset[idx]
    e_true = torch.from_numpy(e_true_np).unsqueeze(0)
    b_star = torch.from_numpy(b_star_np).unsqueeze(0)

    with torch.no_grad():
        z = torch.randn(k_samples, model.latent_dim)
        b_rep = b_star.expand(k_samples, -1)
        e_samps = model.decode(b_rep, z)

    e_samps_un = e_samps * dataset.sigma_eps + dataset.mu_eps
    e_true_un = e_true * dataset.sigma_eps + dataset.mu_eps

    e_mean_un = e_samps_un.mean(dim=0, keepdim=True)
    l2_to_mean = torch.norm(e_samps_un - e_mean_un, dim=1)
    mean_l2 = float(l2_to_mean.mean().item())

    std_per_dim = e_samps_un.std(dim=0)
    mean_std = float(std_per_dim.mean().item())
    max_std = float(std_per_dim.max().item())

    e_scale = float(torch.norm(e_true_un, dim=1).item()) + 1e-8
    mean_l2_norm = mean_l2 / e_scale
    mean_std_norm = mean_std / (e_scale / (e_true_un.shape[1] ** 0.5))
    max_std_norm = max_std / (e_scale / (e_true_un.shape[1] ** 0.5))

    err_to_true = float(torch.norm(e_mean_un - e_true_un, dim=1).item())
    err_to_true_norm = err_to_true / e_scale

    e_all = torch.as_tensor(dataset.eps_slices)
    b_all = torch.as_tensor(dataset.b_slices)
    A = torch.linalg.lstsq(e_all, b_all).solution

    b_star_un = b_star * dataset.sigma_b + dataset.mu_b
    b_hat = e_samps_un @ A
    diff = b_hat - b_star_un

    l2 = torch.norm(diff, dim=1)
    mean_l2_fw = float(l2.mean().item())
    rmse_fw = float(torch.sqrt((diff**2).mean()).item())
    max_abs_fw = float(diff.abs().max().item())

    b_scale = float(torch.norm(b_star_un, dim=1).item()) + 1e-8
    mean_l2_fw_norm = mean_l2_fw / b_scale
    rmse_fw_norm = rmse_fw / (b_scale / (b_star_un.shape[1] ** 0.5))
    max_abs_fw_norm = max_abs_fw / (b_scale / (b_star_un.shape[1] ** 0.5))

    return {
        "idx": idx,
        "k_samples": k_samples,
        "diversity": {
            "mean_l2_to_sample_mean_norm": mean_l2_norm,
            "mean_per_dim_std_norm": mean_std_norm,
            "max_per_dim_std_norm": max_std_norm,
            "l2_sample_mean_to_true_norm": err_to_true_norm,
        },
        "forward_consistency": {
            "mean_l2_norm": mean_l2_fw_norm,
            "rmse_norm": rmse_fw_norm,
            "max_abs_error_norm": max_abs_fw_norm,
        },
    }
