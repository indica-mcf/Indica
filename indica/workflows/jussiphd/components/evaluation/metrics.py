"""VAE diversity and forward-consistency metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from indica.workflows.jussiphd.components.ml.vae import CVAENetwork
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset


def _summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_vae_diversity_and_forward_metrics(
    model_path: str,
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    idx: int | None = None,
    n_eval_samples: int | None = 200,
    k_samples: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute pointwise aggregated diversity/consistency/accuracy metrics."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=meta_path)
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")

    if n_eval_samples is None or n_eval_samples <= 0 or n_eval_samples >= n:
        eval_indices = np.arange(n, dtype=int)
    elif n_eval_samples == 1 and idx is not None:
        eval_indices = np.asarray([int(np.clip(idx, 0, n - 1))], dtype=int)
    else:
        rng = np.random.default_rng(seed if seed is not None else None)
        eval_indices = np.sort(rng.choice(n, size=int(n_eval_samples), replace=False)).astype(int)
        if idx is not None:
            anchor = int(np.clip(idx, 0, n - 1))
            if anchor not in set(eval_indices.tolist()):
                eval_indices[0] = anchor

    ckpt = torch.load(model_path, map_location="cpu")
    model = CVAENetwork(
        b_dim=int(ckpt["b_dim"]),
        e_dim=int(ckpt["e_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    e_all = torch.as_tensor(dataset.eps_slices, dtype=torch.float32)
    b_all = torch.as_tensor(dataset.b_slices, dtype=torch.float32)
    A = torch.linalg.lstsq(e_all, b_all).solution
    W = np.linalg.lstsq(dataset.b_slices, dataset.eps_slices, rcond=None)[0]

    diversity_mean_l2_norm_vals: list[float] = []
    diversity_mean_std_norm_vals: list[float] = []
    diversity_max_std_norm_vals: list[float] = []
    diversity_err_to_true_norm_vals: list[float] = []

    fw_mean_l2_norm_vals: list[float] = []
    fw_rmse_norm_vals: list[float] = []
    fw_max_abs_norm_vals: list[float] = []

    vae_rmse_vals: list[float] = []
    naive_rmse_vals: list[float] = []

    for eval_idx in eval_indices:
        e_true_norm, b_star_norm = dataset[int(eval_idx)]
        e_true_un_np = e_true_norm * dataset.sigma_eps + dataset.mu_eps
        b_star_un_np = b_star_norm * dataset.sigma_b + dataset.mu_b

        e_true_un = torch.from_numpy(e_true_un_np.astype(np.float32)).unsqueeze(0)
        b_star = torch.from_numpy(b_star_norm.astype(np.float32)).unsqueeze(0)
        b_star_un = torch.from_numpy(b_star_un_np.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            z = torch.randn(k_samples, model.latent_dim)
            b_rep = b_star.expand(k_samples, -1)
            e_samps = model.decode(b_rep, z)

        e_samps_un = e_samps * dataset.sigma_eps + dataset.mu_eps
        e_mean_un = e_samps_un.mean(dim=0, keepdim=True)

        l2_to_mean = torch.norm(e_samps_un - e_mean_un, dim=1)
        mean_l2 = float(l2_to_mean.mean().item())
        std_per_dim = e_samps_un.std(dim=0)
        mean_std = float(std_per_dim.mean().item())
        max_std = float(std_per_dim.max().item())

        e_scale = float(torch.norm(e_true_un, dim=1).item()) + 1e-8
        diversity_mean_l2_norm_vals.append(mean_l2 / e_scale)
        diversity_mean_std_norm_vals.append(mean_std / (e_scale / (e_true_un.shape[1] ** 0.5)))
        diversity_max_std_norm_vals.append(max_std / (e_scale / (e_true_un.shape[1] ** 0.5)))
        err_to_true = float(torch.norm(e_mean_un - e_true_un, dim=1).item())
        diversity_err_to_true_norm_vals.append(err_to_true / e_scale)

        b_hat = e_samps_un @ A
        diff = b_hat - b_star_un
        l2 = torch.norm(diff, dim=1)
        mean_l2_fw = float(l2.mean().item())
        rmse_fw = float(torch.sqrt((diff**2).mean()).item())
        max_abs_fw = float(diff.abs().max().item())
        b_scale = float(torch.norm(b_star_un, dim=1).item()) + 1e-8
        fw_mean_l2_norm_vals.append(mean_l2_fw / b_scale)
        fw_rmse_norm_vals.append(rmse_fw / (b_scale / (b_star_un.shape[1] ** 0.5)))
        fw_max_abs_norm_vals.append(max_abs_fw / (b_scale / (b_star_un.shape[1] ** 0.5)))

        e_vae_mean_np = e_mean_un.squeeze(0).cpu().numpy()
        e_naive_np = b_star_un_np @ W
        vae_rmse_vals.append(float(np.sqrt(np.mean((e_vae_mean_np - e_true_un_np) ** 2))))
        naive_rmse_vals.append(float(np.sqrt(np.mean((e_naive_np - e_true_un_np) ** 2))))

    rmse_delta = (np.asarray(naive_rmse_vals, dtype=float) - np.asarray(vae_rmse_vals, dtype=float)).tolist()
    vae_win_rate = float(np.mean(np.asarray(rmse_delta, dtype=float) > 0.0))
    ref_idx = int(np.clip(idx, 0, n - 1)) if idx is not None else int(eval_indices[0])

    return {
        "idx": ref_idx,
        "n_total_samples": int(n),
        "n_eval_samples": int(len(eval_indices)),
        "eval_indices": [int(i) for i in eval_indices.tolist()],
        "k_samples": k_samples,
        "diversity": {
            "mean_l2_to_sample_mean_norm": float(np.mean(diversity_mean_l2_norm_vals)),
            "mean_per_dim_std_norm": float(np.mean(diversity_mean_std_norm_vals)),
            "max_per_dim_std_norm": float(np.mean(diversity_max_std_norm_vals)),
            "l2_sample_mean_to_true_norm": float(np.mean(diversity_err_to_true_norm_vals)),
            "distribution": {
                "mean_l2_to_sample_mean_norm": _summary_stats(diversity_mean_l2_norm_vals),
                "mean_per_dim_std_norm": _summary_stats(diversity_mean_std_norm_vals),
                "max_per_dim_std_norm": _summary_stats(diversity_max_std_norm_vals),
                "l2_sample_mean_to_true_norm": _summary_stats(diversity_err_to_true_norm_vals),
            },
        },
        "forward_consistency": {
            "mean_l2_norm": float(np.mean(fw_mean_l2_norm_vals)),
            "rmse_norm": float(np.mean(fw_rmse_norm_vals)),
            "max_abs_error_norm": float(np.mean(fw_max_abs_norm_vals)),
            "distribution": {
                "mean_l2_norm": _summary_stats(fw_mean_l2_norm_vals),
                "rmse_norm": _summary_stats(fw_rmse_norm_vals),
                "max_abs_error_norm": _summary_stats(fw_max_abs_norm_vals),
            },
        },
        "pointwise_accuracy": {
            "vae_rmse": _summary_stats(vae_rmse_vals),
            "naive_rmse": _summary_stats(naive_rmse_vals),
            "rmse_delta_naive_minus_vae": _summary_stats(rmse_delta),
            "vae_win_rate": vae_win_rate,
        },
    }
