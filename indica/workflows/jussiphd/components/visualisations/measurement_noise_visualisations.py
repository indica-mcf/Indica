"""Visualisations for clean-vs-noisy measurement comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_measurement_noise_comparison(
    clean_b_path: str,
    noisy_b_path: str,
    output_dir: str,
    n_examples: int = 6,
) -> dict[str, Any]:
    """Plot clean and noisy measurement vectors for selected test samples."""
    clean = np.loadtxt(clean_b_path, delimiter=",", dtype=np.float32)
    noisy = np.loadtxt(noisy_b_path, delimiter=",", dtype=np.float32)
    if clean.ndim == 1:
        clean = clean[None, :]
    if noisy.ndim == 1:
        noisy = noisy[None, :]
    if clean.shape != noisy.shape:
        raise ValueError(f"Shape mismatch: clean={clean.shape}, noisy={noisy.shape}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(clean.shape[0])
    n_pick = min(max(1, int(n_examples)), n)
    idx = np.unique(np.round(np.linspace(0, n - 1, n_pick)).astype(int))

    n_cols = 2
    n_rows = int(np.ceil(len(idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(idx):]:
        ax.axis("off")

    channels = np.arange(clean.shape[1], dtype=int)
    for ax, i in zip(axes[: len(idx)], idx):
        ax.plot(channels, clean[i], linewidth=2.0, label="clean b")
        ax.plot(channels, noisy[i], linewidth=1.6, alpha=0.85, label="noisy b")
        ax.set_title(f"sample idx={int(i)}")
        ax.set_xlabel("channel")
        ax.set_ylabel("brightness")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Measurement comparison: clean vs noisy test brightness", y=1.02)
    fig.tight_layout()

    plot_path = out_dir / "test_measurements_clean_vs_noisy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot_path": str(plot_path),
        "num_examples": int(len(idx)),
        "num_samples": int(n),
    }
