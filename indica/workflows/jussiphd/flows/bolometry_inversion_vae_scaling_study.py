"""Backward-compatible shim; use experiments.vae_scaling_study.flow instead."""

from indica.workflows.jussiphd.experiments.vae_scaling_study.flow import (
    bolometry_inversion_vae_scaling_study,
)

__all__ = [
    "bolometry_inversion_vae_scaling_study",
]


if __name__ == "__main__":
    result = bolometry_inversion_vae_scaling_study()
    import pickle

    with open("scaling_results.pickle", "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

