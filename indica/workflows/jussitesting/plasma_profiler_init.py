import argparse
import logging
from typing import Iterable

from hydra import compose
from hydra import initialize_config_module
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from indica.defaults.load_defaults import load_default_objects
from indica.plasma import Plasma
from indica.plasma import PlasmaProfiler
from indica.workflows.bda_phantom_optimisation import initialise_profilers
from indica.workflows.bda.priors import PriorManager
from indica.workflows.bda.priors import PriorType
from indica.workflows.bda.priors import sample_from_priors


def load_bda_config(
    config_name: str = "ion_temperature_phantom_run",
    config_module: str = "indica.configs.workflows.bda",
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    if overrides is None:
        overrides = []
    with initialize_config_module(version_base=None, config_module=config_module):
        return compose(config_name=config_name, overrides=list(overrides))


def build_plasma_profiler(
    cfg: DictConfig,
    *,
    device: str = "st40",
    equilibrium=None,
    save_phantoms: bool = False,
) -> PlasmaProfiler:
    log = logging.getLogger(__name__)
    if equilibrium is None:
        log.info("Loading default equilibrium for %s", device)
        equilibrium = load_default_objects(device, "equilibrium")

    log.info("Initialising plasma")
    plasma = Plasma(
        tstart=cfg.tstart,
        tend=cfg.tend,
        dt=cfg.dt,
        **cfg.plasma.settings,
    )
    plasma.build_atomic_data()
    plasma.set_equilibrium(equilibrium=equilibrium)

    log.info("Initialising plasma state with PlasmaProfiler")
    profilers = initialise_profilers(
        plasma.rhop,
        profiler_types=cfg.plasma.profiles.profilers,
        profile_names=cfg.plasma.profiles.params.keys(),
        profile_params=OmegaConf.to_container(cfg.plasma.profiles.params),
    )
    plasma_profiler = PlasmaProfiler(
        plasma=plasma,
        profilers=profilers,
    )
    plasma_profiler()
    if save_phantoms:
        plasma_profiler.save_phantoms(phantom=True)

    if hasattr(cfg, "plasma_profiler"):
        log.info("Updating plasma profilers for optimisation")
        profilers = initialise_profilers(
            plasma.rhop,
            profiler_types=cfg.plasma_profiler.profilers,
            profile_names=cfg.plasma_profiler.params.keys(),
            profile_params=OmegaConf.to_container(cfg.plasma_profiler.params),
        )
        plasma_profiler.update_profilers(profilers=profilers)

    return plasma_profiler


def _prior_param_names(prior_manager: PriorManager) -> list[str]:
    param_names: list[str] = []
    for prior in prior_manager.priors.values():
        if prior.type not in (PriorType.BASIC, PriorType.COMPOUND):
            continue
        for label in prior.labels:
            if label not in param_names:
                param_names.append(label)
    return param_names


def sample_prior_parameters(
    cfg: DictConfig,
    names: Iterable[str] | str | None = None,
    *,
    size: int = 1,
    filter_for_profiler: bool = True,
) -> dict[str, float] | dict[str, np.ndarray]:
    """
    Sample parameters from BDA priors.

    If names is None, samples all parameters defined by basic/compound priors.
    """
    prior_manager = PriorManager(**cfg.priors)
    available_names = _prior_param_names(prior_manager)
    allowed_names = None
    if filter_for_profiler:
        profile_params = None
        if hasattr(cfg, "plasma_profiler"):
            profile_params = cfg.plasma_profiler.params
        elif hasattr(cfg, "plasma") and hasattr(cfg.plasma, "profiles"):
            profile_params = cfg.plasma.profiles.params
        if profile_params is not None:
            profile_params = OmegaConf.to_container(profile_params)
            allowed_names = {
                f"{profile_name}.{param_name}"
                for profile_name, params in profile_params.items()
                for param_name in params.keys()
            }

    if names is None:
        param_names = available_names
        if allowed_names is not None:
            param_names = [name for name in param_names if name in allowed_names]
    else:
        if isinstance(names, str):
            param_names = [names]
        else:
            param_names = list(names)
        unknown = [name for name in param_names if name not in available_names]
        if unknown:
            raise ValueError(
                f"Unknown prior parameter name(s): {unknown}. "
                f"Available: {available_names}"
            )
        if allowed_names is not None:
            disallowed = [name for name in param_names if name not in allowed_names]
            if disallowed:
                raise ValueError(
                    "Requested parameter(s) not configured for the profiler: "
                    f"{disallowed}. Available: {sorted(allowed_names)}"
                )

    samples = sample_from_priors(param_names, prior_manager.priors, size=size)
    if size == 1:
        return {name: float(samples[0, idx]) for idx, name in enumerate(param_names)}
    return {name: samples[:, idx] for idx, name in enumerate(param_names)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise a PlasmaProfiler for updating plasma profiles."
    )
    parser.add_argument(
        "--config-name",
        default="ion_temperature_phantom_run",
        help="Hydra config name under indica.configs.workflows.bda",
    )
    parser.add_argument(
        "--config-module",
        default="indica.configs.workflows.bda",
        help="Hydra config module to load",
    )
    parser.add_argument(
        "--device",
        default="st40",
        help="Device name for default equilibrium loading",
    )
    parser.add_argument(
        "--save-phantoms",
        action="store_true",
        help="Save phantom profiles after initialisation",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_bda_config(
        config_name=args.config_name,
        config_module=args.config_module,
    )
    plasma_profiler = build_plasma_profiler(
        cfg,
        device=args.device,
        save_phantoms=args.save_phantoms,
    )

    example = {"electron_temperature.y0": 1500}
    plasma_profiler(example)
    print("PlasmaProfiler ready. Example update applied:", example)


if __name__ == "__main__":
    main()
