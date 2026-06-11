"""Dataset noise-injection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)


def _add_poisson_noise_to_csv(
    csv_path: str,
    quantity_name: str,
    count_level: float = 200.0,
    output_path: str | None = None,
    scale_percentile: float = 99.0,
    seed: int = 0,
) -> dict[str, Any]:
    in_path = Path(csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"{quantity_name} CSV does not exist: {in_path}")

    values = np.loadtxt(in_path, delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]

    rng = np.random.default_rng(int(seed))
    scale_value = float(
        np.percentile(np.clip(values, a_min=0.0, a_max=None), float(scale_percentile))
    )
    if scale_value <= 0:
        scale_value = 1.0

    noisy = add_poisson_noise_with_counts(
        values=values,
        count_level=float(count_level),
        scale_value=scale_value,
        rng=rng,
    )

    if output_path is None:
        out_path = in_path.with_name(
            f"{in_path.stem}_poisson{int(float(count_level))}{in_path.suffix}"
        )
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, noisy.astype(np.float32), delimiter=",")

    return {
        f"input_{quantity_name}_path": str(in_path),
        f"output_{quantity_name}_path": str(out_path),
        "count_level": float(count_level),
        "scale_percentile": float(scale_percentile),
        "scale_value": float(scale_value),
        "shape": tuple(noisy.shape),
    }


def add_poisson_noise_to_eps_csv(
    eps_path: str,
    count_level: float = 200.0,
    output_path: str | None = None,
    scale_percentile: float = 99.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a Poisson-noised emissivity CSV copy and return metadata."""
    return _add_poisson_noise_to_csv(
        csv_path=eps_path,
        quantity_name="eps",
        count_level=count_level,
        output_path=output_path,
        scale_percentile=scale_percentile,
        seed=seed,
    )


def add_poisson_noise_to_b_csv(
    b_path: str,
    count_level: float = 200.0,
    output_path: str | None = None,
    scale_percentile: float = 99.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a Poisson-noised brightness CSV copy and return metadata."""
    return _add_poisson_noise_to_csv(
        csv_path=b_path,
        quantity_name="b",
        count_level=count_level,
        output_path=output_path,
        scale_percentile=scale_percentile,
        seed=seed,
    )
