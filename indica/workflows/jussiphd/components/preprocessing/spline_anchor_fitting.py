"""Utilities for fitting profile matrices into spline-anchor space."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from indica.workflows.jussiphd.plasma_profiler_init import load_bda_config


def load_monospline_anchor_spec(
    profile_name: str,
    *,
    config_name: str = "baseline_spline_tene",
    config_module: str = "indica.configs.workflows.bda",
) -> dict[str, Any]:
    """Load mono-spline knot/end-point spec from BDA config."""
    cfg = load_bda_config(config_name=config_name, config_module=config_module)
    if not hasattr(cfg, "plasma_profiler"):
        raise ValueError(f"Config '{config_name}' has no plasma_profiler section.")

    params_all = OmegaConf.to_container(cfg.plasma_profiler.params)
    if profile_name not in params_all:
        raise KeyError(f"Profile '{profile_name}' not found in plasma_profiler.params.")
    params = params_all[profile_name]

    if "xknots" not in params:
        raise KeyError(f"Profile '{profile_name}' has no xknots.")
    xknots = np.asarray(params["xknots"], dtype=np.float64).reshape(-1)
    if xknots.size < 2:
        raise ValueError(f"Profile '{profile_name}' xknots must have at least 2 entries.")

    y_indices: list[int] = []
    for key in params:
        m = re.fullmatch(r"y(\d+)", str(key))
        if m is not None:
            y_indices.append(int(m.group(1)))
    y_indices = sorted(set(y_indices))
    if len(y_indices) < 2:
        raise ValueError(f"Profile '{profile_name}' must define at least y0 and one edge y.")

    # Use first/last available y-index as core/edge anchor values (e.g., y0 and y5).
    first_idx = y_indices[0]
    legacy_edge_idx = y_indices[-1]
    fixed_start = float(params[f"y{first_idx}"])
    fixed_end = float(params[f"y{legacy_edge_idx}"])

    n_anchors = int(xknots.size)
    anchor_labels = [f"a{i}" for i in range(n_anchors)]
    return {
        "profile_name": profile_name,
        "config_name": config_name,
        "xknots": xknots,
        "n_anchors": n_anchors,
        "fixed_start": fixed_start,
        "fixed_end": fixed_end,
        "fixed_start_param": f"y{first_idx}",
        "fixed_end_param": f"y{legacy_edge_idx}",
        "anchor_labels": anchor_labels,
    }


def _build_piecewise_linear_basis(x: np.ndarray, xknots: np.ndarray) -> np.ndarray:
    """Build basis matrix for piecewise-linear interpolation at given knots."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    knots = np.asarray(xknots, dtype=np.float64).reshape(-1)
    n = int(knots.size)
    m = int(x.size)
    if n < 2:
        raise ValueError("Need at least 2 knots for piecewise-linear basis.")

    b = np.zeros((m, n), dtype=np.float64)
    idx = np.searchsorted(knots, x, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x0 = knots[idx]
    x1 = knots[idx + 1]
    span = np.maximum(x1 - x0, 1e-12)
    w1 = (x - x0) / span
    w0 = 1.0 - w1
    row = np.arange(m, dtype=int)
    b[row, idx] = w0
    b[row, idx + 1] = w1
    return b


def fit_profiles_to_anchor_space(
    profiles: np.ndarray,
    *,
    xknots: np.ndarray,
    fixed_start: float | None,
    fixed_end: float | None,
    x_domain_end: float = 1.0,
) -> dict[str, Any]:
    """Fit rows of profile matrix to anchor vectors with fixed end points."""
    y = np.asarray(profiles, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    if y.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {y.shape}.")

    n_samples, n_points = int(y.shape[0]), int(y.shape[1])
    x_end = float(x_domain_end)
    if not np.isfinite(x_end) or x_end <= 0.0:
        raise ValueError(f"x_domain_end must be positive and finite, got {x_domain_end}.")
    x = np.linspace(0.0, x_end, n_points, dtype=np.float64)
    basis = _build_piecewise_linear_basis(x, np.asarray(xknots, dtype=np.float64))
    n_anchors = int(basis.shape[1])

    anchors = np.full((n_samples, n_anchors), np.nan, dtype=np.float64)
    rmse = np.full(n_samples, np.nan, dtype=np.float64)
    recon = np.full_like(y, np.nan)

    for i in range(n_samples):
        yi = y[i]
        valid = np.isfinite(yi)
        if int(valid.sum()) < max(3, n_anchors):
            continue

        ai = np.zeros(n_anchors, dtype=np.float64)
        fixed_idx: list[int] = []
        if fixed_start is not None:
            ai[0] = float(fixed_start)
            fixed_idx.append(0)
        if fixed_end is not None:
            ai[-1] = float(fixed_end)
            if (n_anchors - 1) not in fixed_idx:
                fixed_idx.append(n_anchors - 1)

        free_idx = [j for j in range(n_anchors) if j not in fixed_idx]
        if not free_idx:
            yhat = basis @ ai
        else:
            rhs = yi.copy()
            if fixed_idx:
                rhs = rhs - basis[:, fixed_idx] @ ai[fixed_idx]
            coef, *_ = np.linalg.lstsq(basis[valid][:, free_idx], rhs[valid], rcond=None)
            ai[free_idx] = coef
            yhat = basis @ ai

        diff = yi[valid] - yhat[valid]
        rmse[i] = float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan
        anchors[i] = ai
        recon[i] = yhat

    ok_mask = np.isfinite(rmse)
    return {
        "anchors": anchors.astype(np.float32),
        "reconstruction": recon.astype(np.float32),
        "rmse": rmse.astype(np.float32),
        "ok_mask": ok_mask,
        "n_input": int(n_samples),
        "n_fit_ok": int(ok_mask.sum()),
        "n_fit_failed": int((~ok_mask).sum()),
        "x_grid_fit": x.astype(np.float32),
    }
