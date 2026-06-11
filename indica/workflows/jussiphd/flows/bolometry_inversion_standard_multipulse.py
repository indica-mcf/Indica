"""Backward-compatible shim; use experiments.standard_multipulse.flow instead."""

from indica.workflows.jussiphd.experiments.standard_multipulse.flow import (
    bolometry_inversion,
)

__all__ = [
    "bolometry_inversion",
]


if __name__ == "__main__":
    result = bolometry_inversion()
    print(result.get("dataset_summary", {}))

