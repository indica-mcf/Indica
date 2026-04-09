"""Plasma data-generation utilities extracted from the surrogate notebook."""

from __future__ import annotations

from typing import Any, Sequence

from indica.workflows.jussiphd.plasma_profiler_init import (
    build_plasma_profiler,
    load_bda_config,
    sample_prior_parameters,
)


DEFAULT_BDA_OVERRIDES: tuple[str, ...] = (
    "plasma.settings.n_rad=41",
    "tstart=0.04",
    "tend=0.15",
    "dt=0.01",
)


class PlasmaGenerator:
    """Generate random plasma states and run the bolometry forward model."""

    def __init__(
        self,
        model: Any,
        transform: Any,
        config_name: str = "ion_temperature_phantom_run_all_params",
        overrides: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.cfg = load_bda_config(
            config_name=config_name,
            overrides=list(overrides) if overrides is not None else list(DEFAULT_BDA_OVERRIDES),
        )
        self.plasma_profiler = build_plasma_profiler(self.cfg)

    def generate(self) -> Any:
        """Sample prior parameters and build a plasma instance."""
        all_params = sample_prior_parameters(self.cfg)
        self.plasma_profiler(all_params)
        return self.plasma_profiler.plasma

    def run_model(self, target_plasma: Any | None = None) -> tuple[Any, Any]:
        """Run forward model and return (brightness, emissivity)."""
        if target_plasma is not None:
            self.model.set_plasma(target_plasma)
        else:
            self.model.set_plasma(self.plasma_profiler.plasma)

        self.model.set_transform(self.transform)
        bckc, emissivity = self.model(return_emissivity=True)
        measurements = bckc["brightness"]
        return measurements, emissivity
