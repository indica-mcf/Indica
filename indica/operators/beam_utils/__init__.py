"""Utilities for neutral beam injection (NBI) modeling."""

from typing import Callable

__all__ = [
    "adas_nbi_utils",
    "analytic_nbi_utils",
    "fidasim_utils",
    "NBI_MODEL_HANDLERS",
    "get_nbi_model_handler",
]

NBI_MODEL_HANDLERS = {}
try:
    from .adas_nbi_utils import run_adas

    __all__ += ["run_adas"]
    NBI_MODEL_HANDLERS["ADAS"] = run_adas
except ImportError:
    pass

try:
    from .adas_nbi_utils import run_analytic

    __all__ += ["run_analytic"]
    NBI_MODEL_HANDLERS["ANALYTIC"] = run_analytic
except ImportError:
    pass

try:
    from .adas_nbi_utils import run_fidasim

    __all__ += ["run_fidasim"]
    NBI_MODEL_HANDLERS["FIDASIM"] = run_fidasim
except ImportError:
    pass


def get_nbi_model_handler(model_key: str) -> Callable:
    _model_key = str(model_key).strip().upper()
    if _model_key not in NBI_MODEL_HANDLERS:
        supported = ", ".join(NBI_MODEL_HANDLERS.keys())
        raise ValueError(
            f"Unknown nbi_model '{_model_key}'. Supported models: {supported}"
        )
    return NBI_MODEL_HANDLERS[_model_key]
