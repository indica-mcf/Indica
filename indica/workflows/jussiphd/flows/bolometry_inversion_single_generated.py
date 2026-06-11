"""Backward-compatible shim; use experiments.single_generated.flow instead."""

from indica.workflows.jussiphd.experiments.single_generated.flow import (
    bolometry_inversion_single_generated,
)

__all__ = [
    "bolometry_inversion_single_generated",
]


if __name__ == "__main__":
    result = bolometry_inversion_single_generated()

