"""Context-aware generated comparisons: VAE vs naive inversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from xarray import DataArray

from indica.workflows.jussiphd.components.data.data_generation import sample_plasma
from indica.workflows.jussiphd.components.data.expanded_equilibria_generation import (
    DEFAULT_EQUILIBRIUM_SPECS,
    align_plasma_fz_to_times,
    build_equilibrium_contexts,
)
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    load_vae,
    next_available_path,
)
from indica.workflows.jussiphd.los_bolometry_radiation import calculate_tomo_inversion


def generate_contextual_vae_vs_naive_visualisations(
    model_path: str,
    b_path: str,
    eps_path: str,
    output_dir: str,
    machine: str,
    instrument: str,
    config_name: str,
    config_overrides: list[str] | None,
    n_timepoints_per_equilibrium: int,
    n_generated_samples: int,
    c_concentration: float,
    ar_concentration: float,
    k_samples: int,
    n_examples: int,
    seed: int | None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=None)
    vae = load_vae(model_path)

    contexts = build_equilibrium_contexts(
        machine=machine,
        instrument=instrument,
        equilibrium_specs=list(DEFAULT_EQUILIBRIUM_SPECS),
        n_timepoints_per_equilibrium=int(n_timepoints_per_equilibrium),
    )
    points: list[dict[str, Any]] = []
    for ctx in contexts:
        target_t = np.asarray(ctx["target_t"], dtype=float).reshape(-1)
        for tidx, t_value in enumerate(target_t):
            points.append(
                {
                    "eq_idx": int(ctx["eq_idx"]),
                    "spec": ctx["spec"],
                    "model": ctx["model"],
                    "target_t": target_t.copy(),
                    "tidx": int(tidx),
                    "t_s": float(t_value),
                }
            )
    if len(points) == 0:
        raise ValueError("No equilibrium/time points available for comparison generation.")

    reference_model = contexts[0]["model"]
    reference_transform = contexts[0]["transform"]
    overrides = (
        list(config_overrides)
        if config_overrides is not None
        else ["plasma.settings.n_rad=41", "tstart=0.015", "tend=0.160", "dt=0.005"]
    )

    rows: list[dict[str, Any]] = []
    example_store: list[dict[str, np.ndarray]] = []
    for idx in range(int(n_generated_samples)):
        pt = points[int(rng.integers(len(points)))]
        model = pt["model"]
        target_t = np.asarray(pt["target_t"], dtype=float).reshape(-1)
        tidx = int(pt["tidx"])
        t_s = float(pt["t_s"])

        plasma = sample_plasma(
            model=reference_model,
            transform=reference_transform,
            config_name=config_name,
            overrides=overrides,
        )
        plasma.set_impurity_concentration(
            element="c",
            concentration=float(c_concentration),
            flat_zeff=True,
        )
        plasma.set_impurity_concentration(
            element="ar",
            concentration=float(ar_concentration),
            flat_zeff=True,
        )

        base_fz = {elem: fz_da.copy(deep=True) for elem, fz_da in plasma.fz.items()}
        for elem, fz_da in base_fz.items():
            plasma.fz[elem] = fz_da.copy(deep=True)
        aligned_fz = align_plasma_fz_to_times(plasma, target_t)
        for elem, fz_da in aligned_fz.items():
            plasma.fz[elem] = fz_da

        model.set_plasma(plasma)
        bckc, emissivity = model(t=target_t, return_emissivity=True)
        b_true = bckc["brightness"].isel(t=tidx).values.astype(np.float32).reshape(-1)
        e_true = emissivity.isel(t=tidx).values.astype(np.float32).reshape(-1)
        rhop = emissivity.rhop.values.astype(np.float32)

        brightness_single = DataArray(
            b_true[None, :],
            coords=[("t", np.asarray([t_s], dtype=float)), ("channel", np.arange(b_true.shape[0]))],
        )
        e_naive = (
            calculate_tomo_inversion(
                brightness_single,
                model.transform,
                rhop,
            )
            .isel(t=0)
            .values.astype(np.float32)
        )

        b_norm = (b_true - dataset.mu_b) / dataset.sigma_b
        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(int(k_samples), vae.latent_dim)
            b_rep = b_t_norm.expand(int(k_samples), -1)
            e_samps = vae.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()
        e_vae_mean = e_samps_un.mean(axis=0).astype(np.float32)

        valid = np.isfinite(e_true) & np.isfinite(e_naive) & np.isfinite(e_vae_mean)
        if not np.any(valid):
            continue

        naive_rmse = float(np.sqrt(np.mean((e_naive[valid] - e_true[valid]) ** 2)))
        vae_rmse = float(np.sqrt(np.mean((e_vae_mean[valid] - e_true[valid]) ** 2)))
        rows.append(
            {
                "sample_index": len(rows),
                "trial_index": int(idx),
                "equilibrium_index": int(pt["eq_idx"]),
                "equilibrium_label": str(pt["spec"]["label"]),
                "pulse": int(pt["spec"]["pulse"]),
                "t_s": float(t_s),
                "naive_rmse": naive_rmse,
                "vae_rmse": vae_rmse,
                "delta_naive_minus_vae": float(naive_rmse - vae_rmse),
            }
        )
        if len(example_store) < int(n_examples):
            example_store.append(
                {
                    "rhop": rhop.copy(),
                    "true": e_true.copy(),
                    "naive": e_naive.copy(),
                    "vae": e_vae_mean.copy(),
                    "label": np.array([rows[-1]["equilibrium_label"], rows[-1]["t_s"]], dtype=object),
                }
            )

    if len(rows) == 0:
        raise ValueError("No valid generated comparison samples; all trials were invalid.")

    naive_rmse_vals = np.asarray([r["naive_rmse"] for r in rows], dtype=float)
    vae_rmse_vals = np.asarray([r["vae_rmse"] for r in rows], dtype=float)
    delta = naive_rmse_vals - vae_rmse_vals

    scatter_path = next_available_path(out / "generated_contextual_vae_vs_naive_rmse_scatter.png")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(naive_rmse_vals, vae_rmse_vals, alpha=0.75, s=24)
    mn = float(min(naive_rmse_vals.min(), vae_rmse_vals.min()))
    mx = float(max(naive_rmse_vals.max(), vae_rmse_vals.max()))
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax.set_xlabel("Naive inversion RMSE")
    ax.set_ylabel("VAE RMSE")
    ax.set_title(f"Per-context comparison ({len(rows)} generated samples)")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"VAE wins: {100.0 * np.mean(delta > 0):.1f}%\nmedian Δ={np.median(delta):.4f}",
        transform=ax.transAxes,
        va="top",
    )
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_show = len(example_store)
    n_cols = 2
    n_rows = int(np.ceil(max(1, n_show) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_show:]:
        ax.axis("off")
    for ax, ex in zip(axes[:n_show], example_store):
        rhop = ex["rhop"]
        valid = np.isfinite(ex["true"]) & np.isfinite(ex["naive"]) & np.isfinite(ex["vae"])
        if not np.any(valid):
            continue
        ax.plot(rhop[valid], ex["true"][valid], color="k", linewidth=2.2, label="Ground truth")
        ax.plot(rhop[valid], ex["naive"][valid], color="tab:blue", linewidth=1.9, label="Naive inversion")
        ax.plot(rhop[valid], ex["vae"][valid], color="tab:red", linewidth=1.9, label="VAE mean")
        eq_label = str(ex["label"][0])
        t_s = float(ex["label"][1])
        ax.set_title(f"{eq_label}, t={t_s:.3f}s")
        ax.set_xlabel("rhop")
        ax.grid(alpha=0.25)
    if n_show > 0:
        axes[0].set_ylabel("emissivity")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Generated contextual comparisons: ground truth vs naive vs VAE", y=1.02)
    fig.tight_layout()
    examples_path = next_available_path(out / "generated_contextual_truth_naive_vae_examples.png")
    fig.savefig(examples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics_csv = out / "generated_contextual_comparison_metrics.csv"
    np.savetxt(
        metrics_csv,
        np.asarray(
            [
                [
                    r["sample_index"],
                    r["trial_index"],
                    r["equilibrium_index"],
                    r["pulse"],
                    r["t_s"],
                    r["naive_rmse"],
                    r["vae_rmse"],
                    r["delta_naive_minus_vae"],
                ]
                for r in rows
            ],
            dtype=float,
        ),
        delimiter=",",
        header="sample_index,trial_index,equilibrium_index,pulse,t_s,naive_rmse,vae_rmse,delta_naive_minus_vae",
        comments="",
    )

    return {
        "num_valid_samples": int(len(rows)),
        "num_requested_samples": int(n_generated_samples),
        "num_equilibrium_points": int(len(points)),
        "median_naive_rmse": float(np.median(naive_rmse_vals)),
        "median_vae_rmse": float(np.median(vae_rmse_vals)),
        "median_delta_naive_minus_vae": float(np.median(delta)),
        "vae_win_fraction": float(np.mean(delta > 0)),
        "scatter_plot": str(scatter_path),
        "examples_plot": str(examples_path),
        "metrics_csv": str(metrics_csv),
    }
