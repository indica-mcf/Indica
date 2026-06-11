"""Dataset noise-injection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from indica.workflows.jussiphd.components.evaluation.noise_likelihood import (
    add_poisson_noise_with_counts,
)


def add_poisson_noise_to_eps_csv(
    eps_path: str,
    count_level: float = 200.0,
    output_path: str | None = None,
    scale_percentile: float = 99.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a Poisson-noised emissivity CSV copy and return metadata."""
    in_path = Path(eps_path)
    if not in_path.exists():
        raise FileNotFoundError(f"EPS CSV does not exist: {in_path}")

    eps = np.loadtxt(in_path, delimiter=",", dtype=np.float32)
    if eps.ndim == 1:
        eps = eps[None, :]

    rng = np.random.default_rng(int(seed))
    scale_value = float(
        np.percentile(np.clip(eps, a_min=0.0, a_max=None), float(scale_percentile))
    )
    if scale_value <= 0:
        scale_value = 1.0

    eps_noisy = add_poisson_noise_with_counts(
        values=eps,
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
    np.savetxt(out_path, eps_noisy.astype(np.float32), delimiter=",")

    return {
        "input_eps_path": str(in_path),
        "output_eps_path": str(out_path),
        "count_level": float(count_level),
        "scale_percentile": float(scale_percentile),
        "scale_value": float(scale_value),
        "shape": tuple(eps_noisy.shape),
    }
