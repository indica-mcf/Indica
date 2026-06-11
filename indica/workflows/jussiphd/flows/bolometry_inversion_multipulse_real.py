"""Backward-compatible shim; use experiments.multipulse_real.flow instead."""

from indica.workflows.jussiphd.experiments.multipulse_real.flow import (
    bolometry_inversion_multipulse_real,
    bolometry_inversion_single_real,
)

__all__ = [
    "bolometry_inversion_multipulse_real",
    "bolometry_inversion_single_real",
]


if __name__ == "__main__":
    result = bolometry_inversion_multipulse_real()

