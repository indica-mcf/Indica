"""VAE/tomography visualisations extracted from surrogate notebook cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from indica.workflows.jussiphd.components.data.data_generation import build_real_model_for_pulse
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


def _choose_time_indices(valid_t_indices: list[int], n_pick: int) -> np.ndarray:
    arr = np.asarray(valid_t_indices, dtype=int)
    if arr.size <= n_pick:
        return arr
    pos = np.linspace(0, arr.size - 1, n_pick)
    idx = np.round(pos).astype(int)
    idx = np.clip(idx, 0, arr.size - 1)
    idx = np.unique(idx)
    chosen = arr[idx]
    if chosen.size < n_pick:
        for ti in arr:
            if ti not in chosen:
                chosen = np.append(chosen, ti)
            if chosen.size == n_pick:
                break
    return np.asarray(chosen[:n_pick], dtype=int)


def generate_vae_tomo_visualisations(
    model_path: str,
    b_path: str,
    eps_path: str,
    meta_path: str,
    visualization_slice_pool: list[dict[str, Any]],
    visualization_pulse_to_t_indices: dict[Any, list[int]],
    machine: str = "st40",
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    output_dir: str = ".",
    max_pulses_to_plot: int = 3,
    max_test_samples: int = 50,
    n_times_per_pulse: int = 4,
) -> dict[str, Any]:
    """Generate and save the next notebook visualisations (excluding speed benchmark)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(b_path=b_path, eps_path=eps_path, meta_path=meta_path)
    model = _load_vae(model_path)

    pulse_to_t = {
        int(k): [int(v) for v in vals]
        for k, vals in visualization_pulse_to_t_indices.items()
    }

    pulse_cache: dict[int, dict[str, Any]] = {}

    def load_bundle(pulse: int) -> dict[str, Any]:
        pulse = int(pulse)
        if pulse not in pulse_cache:
            model_pulse = build_real_model_for_pulse(
                pulse=pulse,
                machine=machine,
                instrument=instrument,
                tstart=tstart,
                tend=tend,
                dt=dt,
            )
            bckc, emissivity = model_pulse(return_emissivity=True)
            pulse_cache[pulse] = {
                "model": model_pulse,
                "brightness": bckc["brightness"],
                "emissivity": emissivity,
            }
        return pulse_cache[pulse]

    saved_files: dict[str, list[str] | str] = {
        "tomo_vs_vae_timeslices": [],
        "emissivity_sampling": [],
        "spatial_uncertainty": [],
    }

    # 1) Tomography inversion vs VAE across time slices
    pulse_candidates = list(pulse_to_t.keys())[:max_pulses_to_plot]
    for pulse in pulse_candidates:
        bundle = load_bundle(pulse)
        brightness = bundle["brightness"]
        emissivity = bundle["emissivity"]

        t_indices = np.asarray(pulse_to_t.get(pulse, []), dtype=int)
        tomo_profiles = []
        used_t_indices = []
        for ti in t_indices:
            try:
                tomo_single = calculate_tomo_inversion(
                    brightness.isel(t=[ti]),
                    bundle["model"].transform,
                    emissivity.rhop,
                )
                tomo_profiles.append(tomo_single.isel(t=0).values)
                used_t_indices.append(int(ti))
            except Exception:
                continue

        if not used_t_indices:
            continue

        if len(used_t_indices) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            axes = np.array([axes])
        else:
            n_cols = 2
            n_rows = int(np.ceil(len(used_t_indices) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows), sharex=True, sharey=True)
            axes = np.atleast_1d(axes).ravel()

        for ax in axes[len(used_t_indices):]:
            ax.axis("off")

        for ax, ti, e_tomo in zip(axes[: len(used_t_indices)], used_t_indices, tomo_profiles):
            e_true = emissivity.isel(t=ti).values
            b_t = brightness.isel(t=ti).values.astype(np.float32)
            b_t_norm = torch.from_numpy((b_t - dataset.mu_b) / dataset.sigma_b).unsqueeze(0)

            k = 20
            z = torch.randn(k, model.latent_dim)
            b_rep = b_t_norm.expand(k, -1)
            with torch.no_grad():
                e_samps = model.decode(b_rep, z)
                e_samps_un = e_samps * dataset.sigma_eps + dataset.mu_eps
                e_vae = e_samps_un.mean(dim=0).cpu().numpy()

            ax.plot(emissivity.rhop, e_true, label="True emissivity")
            ax.plot(emissivity.rhop, e_vae, label="VAE mean")
            ax.plot(emissivity.rhop, e_tomo, label="Naive inversion")
            ax.set_title(f"t = {float(brightness.t.values[ti]):.3f}")
            ax.set_xlabel("rhop")
            ax.set_ylabel("emissivity")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Pulse {pulse}: emissivity comparison across time slices", y=1.02)
        fig.tight_layout()

        path = out_dir / f"tomo_vs_vae_timeslices_pulse_{pulse}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["tomo_vs_vae_timeslices"].append(str(path))

    # 2) VAE vs tomography RMSE paired scatter
    vae_rmse = []
    tomo_rmse = []
    test_records = visualization_slice_pool[: int(max_test_samples)]

    for rec in test_records:
        pulse = int(rec["pulse"])
        t_idx = int(rec["t_idx"])
        try:
            bundle = load_bundle(pulse)
        except Exception:
            continue

        measurements = bundle["brightness"]
        emissivity = bundle["emissivity"]

        b_t = measurements.isel(t=t_idx).values.astype(np.float32)
        e_true = emissivity.isel(t=t_idx).values.astype(np.float32)

        try:
            tomo_single = calculate_tomo_inversion(
                measurements.isel(t=[t_idx]),
                bundle["model"].transform,
                emissivity.rhop,
            )
            e_tomo = tomo_single.isel(t=0).values.astype(np.float32)
        except Exception:
            continue

        b_t_norm = torch.from_numpy((b_t - dataset.mu_b) / dataset.sigma_b).unsqueeze(0)
        z = torch.randn(50, model.latent_dim)
        b_rep = b_t_norm.expand(50, -1)
        with torch.no_grad():
            e_samps = model.decode(b_rep, z)
            e_samps_un = e_samps * dataset.sigma_eps + dataset.mu_eps
            e_vae = e_samps_un.mean(dim=0).cpu().numpy()

        valid = np.isfinite(e_tomo) & np.isfinite(e_true) & np.isfinite(e_vae)
        if not np.any(valid):
            continue

        vae_rmse.append(np.sqrt(np.mean((e_vae[valid] - e_true[valid]) ** 2)))
        tomo_rmse.append(np.sqrt(np.mean((e_tomo[valid] - e_true[valid]) ** 2)))

    if len(vae_rmse) > 0:
        vae_rmse = np.asarray(vae_rmse)
        tomo_rmse = np.asarray(tomo_rmse)

        plt.figure(figsize=(5.5, 5))
        plt.scatter(tomo_rmse, vae_rmse, alpha=0.7, s=30)
        mn = min(tomo_rmse.min(), vae_rmse.min())
        mx = max(tomo_rmse.max(), vae_rmse.max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="y = x")
        plt.xlabel("Tomo RMSE")
        plt.ylabel("VAE RMSE")
        plt.title(f"Paired RMSE over {len(vae_rmse)} shared-filtered samples")
        plt.legend()
        plt.tight_layout()

        delta = tomo_rmse - vae_rmse
        plt.gca().text(
            0.02,
            0.98,
            f"VAE wins: {100.0 * np.mean(delta > 0):.1f}%\\nmedian Δ={np.median(delta):.4f}",
            transform=plt.gca().transAxes,
            va="top",
        )

        rmse_path = out_dir / "vae_vs_tomo_rmse_scatter.png"
        plt.savefig(rmse_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_files["vae_vs_tomo_rmse_scatter"] = str(rmse_path)
    else:
        saved_files["vae_vs_tomo_rmse_scatter"] = "no_valid_samples"

    # 3) Emissivity comparison with VAE samples
    for pulse in pulse_candidates:
        bundle = load_bundle(pulse)
        brightness = bundle["brightness"]
        emissivity = bundle["emissivity"]
        t_plot_indices = _choose_time_indices(pulse_to_t.get(pulse, []), n_times_per_pulse)
        if t_plot_indices.size == 0:
            continue

        n_show = int(t_plot_indices.size)
        n_cols = 2
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.5 * n_rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

        for ax in axes[n_show:]:
            ax.axis("off")

        for ax, t_idx in zip(axes[:n_show], t_plot_indices):
            e_true = emissivity.isel(t=t_idx).values.astype(np.float32)
            b_t = brightness.isel(t=t_idx).values.astype(np.float32)
            b_t_norm = torch.from_numpy((b_t - dataset.mu_b) / dataset.sigma_b).unsqueeze(0)

            with torch.no_grad():
                z = torch.randn(20, model.latent_dim)
                b_rep = b_t_norm.expand(20, -1)
                e_samps = model.decode(b_rep, z)
                e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()

            e_vae_mean = e_samps_un.mean(axis=0)

            ax.plot(emissivity.rhop, e_true, color="k", linewidth=2.2, label="Ground truth")
            for i in range(20):
                label = "VAE sample" if i == 0 else None
                ax.plot(emissivity.rhop, e_samps_un[i], alpha=0.45, linewidth=1.1, label=label)
            ax.plot(emissivity.rhop, e_vae_mean, color="tab:red", linewidth=2.2, label="VAE mean")
            ax.set_title(f"t = {float(brightness.t.values[t_idx]):.3f}")
            ax.set_xlabel("rhop")
            ax.grid(alpha=0.25)

        axes[0].set_ylabel("emissivity")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Pulse {pulse}: emissivity comparison at {n_show} time slices", y=1.02)
        fig.tight_layout()

        p = out_dir / f"emissivity_sampling_pulse_{pulse}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["emissivity_sampling"].append(str(p))

    # 4) Spatial uncertainty map (posterior variance vs rhop)
    for pulse in pulse_candidates:
        bundle = load_bundle(pulse)
        brightness = bundle["brightness"]
        emissivity = bundle["emissivity"]
        t_indices = _choose_time_indices(pulse_to_t.get(pulse, []), n_times_per_pulse)
        if t_indices.size == 0:
            continue

        rhop = np.asarray(emissivity.rhop.values, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4.5))

        for t_idx in t_indices:
            t_sel = float(brightness.t.values[t_idx])
            b_t = brightness.isel(t=t_idx).values.astype(np.float32)
            b_t_norm = torch.from_numpy((b_t - dataset.mu_b) / dataset.sigma_b).unsqueeze(0)

            with torch.no_grad():
                z = torch.randn(200, model.latent_dim)
                b_rep = b_t_norm.expand(200, -1)
                e_samps = model.decode(b_rep, z)
                e_samps_un = (e_samps * dataset.sigma_eps + dataset.mu_eps).cpu().numpy()

            e_var = e_samps_un.var(axis=0)
            ax.plot(rhop, e_var, linewidth=2, label=f"t={t_sel:.3f}")

        ax.set_xlabel("rhop")
        ax.set_ylabel("variance")
        ax.set_title(f"Pulse {pulse}: spatial uncertainty (VAE posterior variance)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        p = out_dir / f"spatial_uncertainty_pulse_{pulse}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["spatial_uncertainty"].append(str(p))

    return {
        "saved_files": saved_files,
        "num_cached_pulses": int(len(pulse_cache)),
        "num_shared_records": int(len(visualization_slice_pool)),
    }
