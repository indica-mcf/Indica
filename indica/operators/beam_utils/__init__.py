"""Utilities for neutral beam injection (NBI) modeling."""

from typing import Callable

from .adas_nbi_utils import run_adas
from .analytic_nbi_utils import run_analytic
from .fidasim_utils import run_fidasim

NBI_MODEL_HANDLERS = {
    "FIDASIM": run_fidasim,
    "ANALYTIC": run_analytic,
    "ADAS": run_adas,
}


def get_nbi_model_handler(model_key: str) -> Callable:
    if model_key not in NBI_MODEL_HANDLERS:
        supported = ", ".join(NBI_MODEL_HANDLERS.keys())
        raise ValueError(
            f"Unknown nbi_model '{model_key}'. Supported models: {supported}"
        )
    return NBI_MODEL_HANDLERS[model_key]


__all__ = [
    "adas_nbi_utils",
    "analytic_nbi_utils",
    "fidasim_utils",
    "run_adas",
    "run_analytic",
    "run_fidasim",
    "NBI_MODEL_HANDLERS",
    "get_nbi_model_handler",
]
