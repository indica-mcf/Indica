"""Configuration helpers for real-vs-synthetic experiment comparison."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from typing import Sequence


DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


@dataclass
class ComparisonConfig:
    """Serializable config for aligned real vs synthetic flow runs."""

    machine: str = "st40"
    instrument: str = "blom_xy1"
    tstart: float = 0.04
    tend: float = 0.14
    dt: float = 0.01
    use_real_equilibrium: bool = True
    real_equilibrium_pulse: int = 13622
    real_equilibrium_verbose: bool = False
    train_fraction: float = 0.8
    batch_size: int = 8
    shuffle: bool = True
    vae_latent_dim: int = 4
    vae_hidden_scaling: int = 8
    vae_n_epochs: int = 25
    vae_lr: float = 1e-3
    metrics_idx: int = 10
    metrics_k_samples: int = 100
    run_visualisations: bool = False

    # Real-flow specific
    pulses: tuple[int, ...] = (13622, 13623, 13624)
    emissivity_instrument: str = "blom_rz1"
    use_all_timepoints_real: bool = True
    revision: int = 0
    node: str | None = None

    # Synthetic-flow specific
    n_generations_synthetic: int = 3000
    use_all_timepoints_synthetic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_real_flow_kwargs(cfg: ComparisonConfig) -> dict[str, Any]:
    return {
        "machine": cfg.machine,
        "instrument": cfg.instrument,
        "emissivity_instrument": cfg.emissivity_instrument,
        "pulses": list(cfg.pulses),
        "tstart": cfg.tstart,
        "tend": cfg.tend,
        "dt": cfg.dt,
        "use_real_equilibrium": cfg.use_real_equilibrium,
        "real_equilibrium_pulse": cfg.real_equilibrium_pulse,
        "real_equilibrium_verbose": cfg.real_equilibrium_verbose,
        "revision": cfg.revision,
        "node": cfg.node,
        "generate_new_data": True,
        "use_all_timepoints": cfg.use_all_timepoints_real,
        "create_training_dataset": True,
        "train_fraction": cfg.train_fraction,
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle,
        "run_vae_training": True,
        "vae_latent_dim": cfg.vae_latent_dim,
        "vae_hidden_scaling": cfg.vae_hidden_scaling,
        "vae_n_epochs": cfg.vae_n_epochs,
        "vae_lr": cfg.vae_lr,
        "run_vae_metrics": True,
        "metrics_idx": cfg.metrics_idx,
        "metrics_k_samples": cfg.metrics_k_samples,
        "run_visualisations": cfg.run_visualisations,
    }


def build_synthetic_flow_kwargs(cfg: ComparisonConfig) -> dict[str, Any]:
    return {
        "machine": cfg.machine,
        "instrument": cfg.instrument,
        "tstart": cfg.tstart,
        "tend": cfg.tend,
        "dt": cfg.dt,
        "use_real_equilibrium": cfg.use_real_equilibrium,
        "real_equilibrium_pulse": cfg.real_equilibrium_pulse,
        "real_equilibrium_verbose": cfg.real_equilibrium_verbose,
        "n_generations": cfg.n_generations_synthetic,
        "generate_new_data": True,
        "use_all_timepoints": cfg.use_all_timepoints_synthetic,
        "create_training_dataset": True,
        "train_fraction": cfg.train_fraction,
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle,
        "run_vae_training": True,
        "vae_latent_dim": cfg.vae_latent_dim,
        "vae_hidden_scaling": cfg.vae_hidden_scaling,
        "vae_n_epochs": cfg.vae_n_epochs,
        "vae_lr": cfg.vae_lr,
        "run_vae_metrics": True,
        "metrics_idx": cfg.metrics_idx,
        "metrics_k_samples": cfg.metrics_k_samples,
        "run_visualisations": cfg.run_visualisations,
    }


def parse_pulses_csv(pulses_csv: str) -> Sequence[int]:
    return tuple(int(p.strip()) for p in pulses_csv.split(",") if p.strip())

