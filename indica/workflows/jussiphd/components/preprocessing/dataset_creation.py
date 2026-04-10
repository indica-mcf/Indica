"""Dataset creation utilities extracted from the surrogate notebook."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np


class PairDataset:
    """Torch-style dataset for emissivity/bolometry pairs with normalization stats."""

    def __init__(
        self,
        b_path: str,
        eps_path: str,
        meta_path: str | None = None,
        zero_tol: float = 0.0,
    ) -> None:
        self.b_path = b_path
        self.eps_path = eps_path
        self.meta_path = meta_path
        self.zero_tol = float(zero_tol)

        self.b_slices = np.loadtxt(b_path, delimiter=",", dtype=np.float32)
        self.eps_slices = np.loadtxt(eps_path, delimiter=",", dtype=np.float32)
        if self.b_slices.ndim == 1:
            self.b_slices = self.b_slices[None, :]
        if self.eps_slices.ndim == 1:
            self.eps_slices = self.eps_slices[None, :]
        if len(self.b_slices) != len(self.eps_slices):
            raise ValueError("b_path and eps_path must contain the same number of samples")

        self.mu_b, self.sigma_b = np.mean(self.b_slices), np.std(self.b_slices)
        self.mu_eps, self.sigma_eps = np.mean(self.eps_slices), np.std(self.eps_slices)

        b_zeros = np.count_nonzero(np.abs(self.b_slices) <= self.zero_tol, axis=1)
        eps_zeros = np.count_nonzero(np.abs(self.eps_slices) <= self.zero_tol, axis=1)
        total_dims = self.b_slices.shape[1] + self.eps_slices.shape[1]
        self.sample_nonzero_fraction = 1.0 - (b_zeros + eps_zeros) / total_dims

        self.pulse_quality: dict[int, float] = {}
        self.best_pulse: int | None = None
        if meta_path is not None and os.path.exists(meta_path):
            with open(meta_path, newline="") as f:
                rows = list(csv.reader(f))
            has_header = bool(rows) and rows[0] == ["pulse", "time_s"]
            data_rows = rows[1:] if has_header else rows
            if len(data_rows) == len(self.sample_nonzero_fraction):
                pulse_to_scores: dict[int, list[float]] = {}
                for row, score in zip(data_rows, self.sample_nonzero_fraction):
                    pulse = int(float(row[0]))
                    pulse_to_scores.setdefault(pulse, []).append(float(score))
                self.pulse_quality = {
                    pulse: float(np.mean(scores)) for pulse, scores in pulse_to_scores.items()
                }
                if self.pulse_quality:
                    self.best_pulse = max(self.pulse_quality, key=self.pulse_quality.get)

    def get_top_pulses(self, n: int = 5, min_nonzero_fraction: float = 0.0) -> list[tuple[int, float]]:
        if not self.pulse_quality:
            return []
        return [
            (pulse, score)
            for pulse, score in sorted(
                self.pulse_quality.items(), key=lambda x: x[1], reverse=True
            )
            if score >= min_nonzero_fraction
        ][:n]

    def __len__(self) -> int:
        return len(self.b_slices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        b_slice = self.b_slices[idx]
        eps_slice = self.eps_slices[idx]
        eps_norm = (eps_slice - self.mu_eps) / self.sigma_eps
        b_norm = (b_slice - self.mu_b) / self.sigma_b
        return eps_norm, b_norm


def create_dataset_and_dataloaders(
    b_path: str,
    eps_path: str,
    meta_path: str | None = None,
    zero_tol: float = 0.0,
    train_fraction: float = 0.8,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Create PairDataset and train/test dataloaders from notebook defaults."""
    import torch
    from torch.utils.data import DataLoader

    dataset = PairDataset(
        b_path=b_path,
        eps_path=eps_path,
        meta_path=meta_path,
        zero_tol=zero_tol,
    )

    n_total = len(dataset)
    n_train = int(train_fraction * n_total)
    n_test = n_total - n_train

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_split, test_split = torch.utils.data.random_split(
        dataset,
        lengths=[n_train, n_test],
        generator=generator,
    )

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=shuffle)

    return {
        "dataset": dataset,
        "train_split": train_split,
        "test_split": test_split,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "summary": {
            "num_samples": n_total,
            "num_train": n_train,
            "num_test": n_test,
            "batch_size": batch_size,
            "mu_b": float(dataset.mu_b),
            "sigma_b": float(dataset.sigma_b),
            "mu_eps": float(dataset.mu_eps),
            "sigma_eps": float(dataset.sigma_eps),
            "best_pulse": dataset.best_pulse,
            "top_pulses": dataset.get_top_pulses(n=20),
        },
    }
