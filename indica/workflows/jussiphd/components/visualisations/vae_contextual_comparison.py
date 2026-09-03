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
from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)
from indica.workflows.jussiphd.components.preprocessing.dataset_creation import PairDataset
from indica.workflows.jussiphd.components.visualisations.vae_generated_visualisations import (
    load_kl_scaling,
    load_vae,
    next_available_path,
    pointwise_band_coverage,
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
    noise_b_test: bool = False,
    noise_count_level: float = 200.0,
    noise_scale_percentile: float = 99.0,
    noise_seed: int | None = 0,
    use_dataset_samples: bool = False,
    meta_path: str | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=None)
    vae = load_vae(model_path)
    kl_scaling = load_kl_scaling(model_path)

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

    generated_samples: list[dict[str, Any]] = []
    if bool(use_dataset_samples):
        b_matrix = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        e_matrix = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
        if b_matrix.ndim == 1:
            b_matrix = b_matrix[None, :]
        if e_matrix.ndim == 1:
            e_matrix = e_matrix[None, :]
        n_rows = int(min(b_matrix.shape[0], e_matrix.shape[0]))
        if n_rows <= 0:
            raise ValueError("No rows available in b/eps dataset for contextual comparison.")
        n_use = int(min(int(n_generated_samples), n_rows))
        selected = rng.choice(n_rows, size=n_use, replace=False)

        meta = None
        if meta_path is not None and Path(meta_path).exists():
            try:
                meta = np.genfromtxt(meta_path, delimiter=",", names=True, dtype=None, encoding=None)
                meta = np.atleast_1d(meta)
            except Exception:
                meta = None
        contexts_by_idx = {int(ctx["eq_idx"]): ctx for ctx in contexts}

        for trial_idx, row_idx in enumerate(selected):
            b_true = b_matrix[int(row_idx)].astype(np.float32).reshape(-1)
            e_true = e_matrix[int(row_idx)].astype(np.float32).reshape(-1)
            rhop = np.linspace(0.0, 1.0, int(e_true.size), dtype=np.float32)

            ctx = points[int(rng.integers(len(points)))]
            eq_idx = int(ctx["eq_idx"])
            eq_label = str(ctx["spec"]["label"])
            pulse = int(ctx["spec"]["pulse"])
            t_s = float(ctx["t_s"])
            model = ctx["model"]
            if meta is not None and int(row_idx) < int(np.size(meta)):
                meta_row = meta[int(row_idx)]
                try:
                    meta_eq = int(meta_row["equilibrium_index"])
                    if meta_eq in contexts_by_idx:
                        ctx_match = contexts_by_idx[meta_eq]
                        target_t = np.asarray(ctx_match["target_t"], dtype=float).reshape(-1)
                        t_guess = float(meta_row["t_s"]) if "t_s" in meta.dtype.names else float(target_t[0])
                        tidx = int(np.argmin(np.abs(target_t - t_guess)))
                        t_s = float(target_t[tidx])
                        eq_idx = int(ctx_match["eq_idx"])
                        eq_label = str(ctx_match["spec"]["label"])
                        pulse = int(ctx_match["spec"]["pulse"])
                        model = ctx_match["model"]
                except Exception:
                    pass

            generated_samples.append(
                {
                    "trial_index": int(trial_idx),
                    "equilibrium_index": int(eq_idx),
                    "equilibrium_label": str(eq_label),
                    "pulse": int(pulse),
                    "t_s": float(t_s),
                    "model": model,
                    "b_true": b_true,
                    "e_true": e_true,
                    "rhop": rhop,
                }
            )
    else:
        reference_model = contexts[0]["model"]
        reference_transform = contexts[0]["transform"]
        overrides = (
            list(config_overrides)
            if config_overrides is not None
            else ["plasma.settings.n_rad=41", "tstart=0.015", "tend=0.160", "dt=0.005"]
        )
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

            generated_samples.append(
                {
                    "trial_index": int(idx),
                    "equilibrium_index": int(pt["eq_idx"]),
                    "equilibrium_label": str(pt["spec"]["label"]),
                    "pulse": int(pt["spec"]["pulse"]),
                    "t_s": float(t_s),
                    "model": model,
                    "b_true": b_true,
                    "e_true": e_true,
                    "rhop": rhop,
                }
            )

    if len(generated_samples) == 0:
        raise ValueError("No generated samples available for contextual comparison.")

    b_clean_mat = np.asarray([s["b_true"] for s in generated_samples], dtype=np.float32)
    b_input_mat = b_clean_mat.copy()
    noise_info: dict[str, Any] | None = None
    if bool(noise_b_test):
        # Match noisy-test-b calibration style: derive scale from the dataset brightness CSV.
        b_reference = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        if b_reference.ndim == 1:
            b_reference = b_reference[None, :]
        scale = float(
            np.percentile(
                np.clip(b_reference, a_min=0.0, a_max=None),
                float(noise_scale_percentile),
            )
        )
        if scale <= 0:
            scale = 1.0
        noise_rng = np.random.default_rng(noise_seed)
        b_input_mat = add_poisson_noise_with_counts(
            values=b_clean_mat,
            count_level=float(noise_count_level),
            scale_value=scale,
            rng=noise_rng,
        ).astype(np.float32)
        noise_info = {
            "enabled": True,
            "count_level": float(noise_count_level),
            "scale_percentile": float(noise_scale_percentile),
            "scale_value": float(scale),
            "scale_source": str(b_path),
            "seed": None if noise_seed is None else int(noise_seed),
        }

    rows: list[dict[str, Any]] = []
    example_store: list[dict[str, np.ndarray]] = []
    vae_sampling_store: list[dict[str, np.ndarray]] = []
    reproj_store: list[dict[str, np.ndarray]] = []
    sampling_cov_weighted_sum = 0.0
    sampling_cov_weight = 0
    for sidx, sample in enumerate(generated_samples):
        b_input = b_input_mat[sidx]
        e_true = sample["e_true"]
        rhop = sample["rhop"]
        model = sample["model"]
        t_s = float(sample["t_s"])

        brightness_single = DataArray(
            b_input[None, :],
            coords=[("t", np.asarray([t_s], dtype=float)), ("channel", np.arange(b_input.shape[0]))],
        )
        try:
            e_naive = (
                calculate_tomo_inversion(
                    brightness_single,
                    model.transform,
                    rhop,
                )
                .isel(t=0)
                .values.astype(np.float32)
            )
        except Exception:
            continue

        # Reproject the naive inverse emissivity back to LOS brightness (b -> inv -> fwd b).
        emissivity_single = DataArray(
            e_naive[None, :],
            coords=[("t", np.asarray([t_s], dtype=float)), ("rhop", rhop)],
        )
        b_reproj_da = model.transform.integrate_on_los(emissivity_single, t=emissivity_single.t)
        if hasattr(b_reproj_da, "isel") and "t" in getattr(b_reproj_da, "dims", ()):
            b_reproj = b_reproj_da.isel(t=0).values.astype(np.float32).reshape(-1)
        else:
            b_reproj = np.asarray(getattr(b_reproj_da, "values", b_reproj_da), dtype=np.float32).reshape(-1)

        b_norm = (b_input - dataset.mu_b) / dataset.sigma_b
        b_t_norm = torch.from_numpy(b_norm.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = torch.randn(int(k_samples), vae.latent_dim)
            b_rep = b_t_norm.expand(int(k_samples), -1)
            e_samps = vae.decode(b_rep, z)
            e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()
        e_vae_mean = e_samps_un.mean(axis=0).astype(np.float32)
        cov_i, cov_n = pointwise_band_coverage(e_true, e_samps_un, central_mass=0.95)
        if np.isfinite(cov_i) and cov_n > 0:
            sampling_cov_weighted_sum += float(cov_i) * float(cov_n)
            sampling_cov_weight += int(cov_n)

        valid = np.isfinite(e_true) & np.isfinite(e_naive) & np.isfinite(e_vae_mean)
        if not np.any(valid):
            continue

        naive_rmse = float(np.sqrt(np.mean((e_naive[valid] - e_true[valid]) ** 2)))
        vae_rmse = float(np.sqrt(np.mean((e_vae_mean[valid] - e_true[valid]) ** 2)))
        rows.append(
            {
                "sample_index": len(rows),
                "trial_index": int(sample["trial_index"]),
                "equilibrium_index": int(sample["equilibrium_index"]),
                "equilibrium_label": str(sample["equilibrium_label"]),
                "pulse": int(sample["pulse"]),
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
            vae_sampling_store.append(
                {
                    "rhop": rhop.copy(),
                    "true": e_true.copy(),
                    "vae_mean": e_vae_mean.copy(),
                    "vae_samples": e_samps_un.astype(np.float32).copy(),
                    "label": np.array([rows[-1]["equilibrium_label"], rows[-1]["t_s"]], dtype=object),
                }
            )
            reproj_store.append(
                {
                    "b_input": b_input.copy(),
                    "b_reproj": b_reproj.copy(),
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
    n_cols = 4
    n_rows = int(np.ceil(max(1, n_show) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4.0 * n_rows), sharex=True, sharey=True)
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

    # Additional view: ground truth + VAE samples + VAE mean.
    n_show_s = len(vae_sampling_store)
    n_cols_s = 4
    n_rows_s = int(np.ceil(max(1, n_show_s) / n_cols_s))
    fig, axes = plt.subplots(
        n_rows_s,
        n_cols_s,
        figsize=(24, 4.0 * n_rows_s),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_show_s:]:
        ax.axis("off")
    for ax, ex in zip(axes[:n_show_s], vae_sampling_store):
        rhop = ex["rhop"]
        e_true = ex["true"]
        e_vae_mean = ex["vae_mean"]
        e_samps_un = ex["vae_samples"]
        ax.plot(rhop, e_true, color="k", linewidth=2.2, label="Ground truth")
        for i in range(e_samps_un.shape[0]):
            lbl = "VAE sample" if i == 0 else None
            ax.plot(rhop, e_samps_un[i], alpha=0.32, linewidth=1.0, label=lbl)
        ax.plot(rhop, e_vae_mean, color="tab:red", linewidth=2.2, label="VAE mean")
        eq_label = str(ex["label"][0])
        t_s = float(ex["label"][1])
        ax.set_title(f"{eq_label}, t={t_s:.3f}s")
        ax.set_xlabel("rhop")
        ax.grid(alpha=0.25)
    if n_show_s > 0:
        axes[0].set_ylabel("emissivity")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    if sampling_cov_weight > 0:
        sampling_cov_pct = 100.0 * (sampling_cov_weighted_sum / float(sampling_cov_weight))
        kl_note = (
            f", KL scaling: {kl_scaling:.3g}"
            if kl_scaling is not None and np.isfinite(kl_scaling)
            else ""
        )
        fig.suptitle(
            "Contextual emissivity sampling: ground truth vs VAE samples "
            f"(95% band coverage: {sampling_cov_pct:.1f}%{kl_note})",
            y=1.02,
        )
    else:
        sampling_cov_pct = float("nan")
        kl_note = (
            f" (KL scaling: {kl_scaling:.3g})"
            if kl_scaling is not None and np.isfinite(kl_scaling)
            else ""
        )
        fig.suptitle(
            "Contextual emissivity sampling: ground truth vs VAE samples" + kl_note,
            y=1.02,
        )
    fig.tight_layout()
    vae_sampling_path = next_available_path(out / "generated_contextual_emissivity_sampling.png")
    fig.savefig(vae_sampling_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Additional diagnostic: measurement vs inverse->forward reprojection in measurement space.
    n_show_b = len(reproj_store)
    n_cols_b = 2
    n_rows_b = int(np.ceil(max(1, n_show_b) / n_cols_b))
    fig, axes = plt.subplots(n_rows_b, n_cols_b, figsize=(12, 3.6 * n_rows_b), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_show_b:]:
        ax.axis("off")
    for ax, ex in zip(axes[:n_show_b], reproj_store):
        channels = np.arange(ex["b_input"].shape[0], dtype=int)
        ax.plot(channels, ex["b_input"], linewidth=1.9, label="Input b")
        ax.plot(channels, ex["b_reproj"], linewidth=1.8, alpha=0.9, label="b -> inv -> fwd b")
        eq_label = str(ex["label"][0])
        t_s = float(ex["label"][1])
        ax.set_title(f"{eq_label}, t={t_s:.3f}s")
        ax.set_xlabel("channel")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.25)
    if n_show_b > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Contextual measurement consistency: input b vs inverse-forward b", y=1.02)
    fig.tight_layout()
    b_reproj_path = next_available_path(out / "generated_contextual_b_vs_inverse_forward_b.png")
    fig.savefig(b_reproj_path, dpi=150, bbox_inches="tight")
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

    result = {
        "num_valid_samples": int(len(rows)),
        "num_requested_samples": int(n_generated_samples),
        "used_dataset_samples": bool(use_dataset_samples),
        "num_equilibrium_points": int(len(points)),
        "median_naive_rmse": float(np.median(naive_rmse_vals)),
        "median_vae_rmse": float(np.median(vae_rmse_vals)),
        "median_delta_naive_minus_vae": float(np.median(delta)),
        "vae_win_fraction": float(np.mean(delta > 0)),
        "scatter_plot": str(scatter_path),
        "examples_plot": str(examples_path),
        "vae_sampling_plot": str(vae_sampling_path),
        "vae_95_band_coverage_pct": (
            float(sampling_cov_pct) if np.isfinite(sampling_cov_pct) else None
        ),
        "vae_kl_scaling": (
            float(kl_scaling) if kl_scaling is not None and np.isfinite(kl_scaling) else None
        ),
        "b_inverse_forward_plot": str(b_reproj_path),
        "metrics_csv": str(metrics_csv),
    }
    if noise_info is not None:
        result["noise"] = noise_info
    return result
