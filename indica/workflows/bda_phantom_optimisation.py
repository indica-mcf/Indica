import logging
from pathlib import Path
import pickle
import pprint

import flatdict
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.plasma import Plasma
from indica.models import ThomsonScattering
from indica.plasma import PlasmaProfiler
from indica.plotters.plot_bda import plot_bda
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.profilers.profiler_spline import ProfilerCubicSpline
from indica.profilers.profiler_spline import ProfilerMonoSpline
from indica.workflows.bda.bda_driver import BDADriver
from indica.workflows.bda.model_coordinator import ModelCoordinator
from indica.workflows.bda.optimisers import BOOptimiser
from indica.workflows.bda.optimisers import BOSettings
from indica.workflows.bda.optimisers import EmceeOptimiser
from indica.workflows.bda.optimisers import EmceeSettings
from indica.workflows.bda.priors import PriorManager

ERROR_FUNCTIONS = {
    "efit.wp": lambda x: x * 0.10,
    "xrcs.spectra_raw": lambda x: x * 0.05
    + 0.01 * x.max(dim="wavelength")
    + (
        x.where(
            (x.wavelength < 0.392) & (x.wavelength > 0.388),
        ).std("wavelength")
    ).fillna(0),
    "cxff_pi.ti": lambda x: x * 0 + 0.20 * x.max(dim="channel"),
    "cxff_tws_c.ti": lambda x: x * 0 + 0.10 * x.max(dim="channel"),
    "cxff_tws_b.ti": lambda x: x * 0 + 0.10 * x.max(dim="channel"),
}

INSTRUMENT_MAPPING: dict = {
    "xrcs": HelikeSpectrometer,
    "cxff_pi": ChargeExchangeSpectrometer,
    "cxff_tws_c": ChargeExchangeSpectrometer,
    "cxff_tws_b": ChargeExchangeSpectrometer,
    "smmh1": Interferometer,
    "efit": EquilibriumReconstruction,
    "ts": ThomsonScattering,
}


def initialise_profilers(
    x_coord, profiler_types: dict, profile_names: list, profile_params: dict = None
):
    profilers = {
        "gauss": ProfilerGauss,
        "mono_spline": ProfilerMonoSpline,
        "cubic_spline": ProfilerCubicSpline,
    }

    if profile_params is None:
        profile_params = {}
    _profilers = {
        profile_name: profilers[profiler_types[profile_name]](
            datatype=profile_name.split(":")[0],
            xspl=x_coord,
            parameters=profile_params.get(profile_name, {}),
        )
        for profile_name in profile_names
    }
    return _profilers


def save_pickle(result, filepath):
    Path(filepath).mkdir(parents=True, exist_ok=True)
    with open(filepath + "results.pkl", "wb") as handle:
        pickle.dump(result, handle)


def deep_update(mapping: dict, *updating_mappings: dict) -> dict:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def add_error_to_opt_data(opt_data: dict, error_functions=None, verbose=True):
    if error_functions is None:
        error_functions = ERROR_FUNCTIONS

    opt_data_with_error = {}
    for key, value in opt_data.items():
        if key not in error_functions.keys():
            opt_data_with_error[key] = value
            if verbose:
                print(f"no error function defined for {key}")
            continue
        error = error_functions[key](value)
        opt_data_with_error[key] = value.assign_coords({"error": error})
    return opt_data_with_error


@hydra.main(
    version_base=None,
    config_path="../configs/workflows/bda/",
    config_name="ion_temperature_phantom_run",
)
def bda_phantom_optimisation(  # noqa: C901
    cfg: DictConfig,
    save_results: bool = False,
):
    """
    This workflow takes all of its inputs from a config file created using hydra.
    It uses phantom data and diagnostic models to prepare an optimisation of synthetic
    data. Replace the phantom data with real experimental data to "easily" analyse a
    real scenario.

    It initialises the following objects (some of which are passed to the BDADriver):
        plasma
        profilers
        plasma_profiler
        model_coordinator
        prior_manager
        optimiser_context
    """

    log = logging.getLogger(__name__)
    config = pprint.pformat(dict(cfg))
    log.info("Beginning BDA phantom optimisation")
    log.info("Using a mock equilibrium")
    equilibrium = load_default_objects("st40", "equilibrium")

    log.info("Initialising plasma")
    plasma = Plasma(
        tstart=cfg.tstart,
        tend=cfg.tend,
        dt=cfg.dt,
        **cfg.plasma.settings,
    )
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
    log.info("Saving plasma phantom attributes")
    plasma_profiler.save_phantoms(phantom=True)

    log.info("Updating Plasma Profiler with profilers used for optimisation")
    profilers = initialise_profilers(
        plasma.rhop,
        profiler_types=cfg.plasma_profiler.profilers,
        profile_names=cfg.plasma_profiler.params.keys(),
        profile_params=OmegaConf.to_container(cfg.plasma_profiler.params),
    )
    plasma_profiler.update_profilers(profilers=profilers)

    log.info("Loading transforms from pickle")
    transforms = load_default_objects("st40", "geometry")

    log.info("Using phantom data reader")
    diagnostic_models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.diagnostics}
    reader = ModelCoordinator(
        models=diagnostic_models,
        model_settings=OmegaConf.to_container(cfg.model),
    )
    reader.set_transforms(transforms)
    reader.set_equilibrium(
        equilibrium,
    )
    reader.set_plasma(plasma)
    reader(
        cfg.diagnostics,
        revisions=OmegaConf.to_container(cfg.reader.revisions),
        fetch_equilbrium=False,
        **cfg.reader.filters,
    )

    flat_data = flatdict.FlatDict(reader.binned_data, ".")
    log.info("Applying error to opt_data")
    opt_data = add_error_to_opt_data(flat_data, verbose=False)

    models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.diagnostics}

    log.info("Initialising ModelCoordinator")
    model_coordinator = ModelCoordinator(
        models=models,
        model_settings=deep_update(
            OmegaConf.to_container(cfg.model),
        ),
        verbose=False,
    )
    if "xrcs" in reader.binned_data.keys():
        model_call_kwargs = {
            "xrcs": {
                "norm_spectra": reader.binned_data["xrcs"]["spectra_raw"].max(
                    "wavelength"
                ),
            }
        }
    else:
        model_call_kwargs = {}

    model_coordinator.set_transforms(reader.transforms)
    model_coordinator.set_equilibrium(equilibrium)
    model_coordinator.set_plasma(plasma)

    log.info("initialising PriorManager")
    prior_manager = PriorManager(**cfg.priors)

    if cfg.optimisation.method == "emcee":
        optimiser_settings = EmceeSettings(
            param_names=OmegaConf.to_container(cfg.param_names),
            nwalkers=cfg.optimisation.nwalkers,
            iterations=cfg.optimisation.iterations,
            sample_method=cfg.optimisation.sample_method,
            starting_samples=cfg.optimisation.starting_samples,
            burn_frac=cfg.optimisation.burn_frac,
            stopping_criteria=cfg.optimisation.stopping_criteria,
            stopping_criteria_factor=cfg.optimisation.stopping_criteria_factor,
            stopping_criteria_debug=True,
        )

        log.info("Initialising Ecmee Optimiser Context")
        optimiser_context = EmceeOptimiser(
            optimiser_settings=optimiser_settings,
            prior_manager=prior_manager,
            model_kwargs=model_call_kwargs,
        )
    elif cfg.optimisation.method == "bo":
        optimiser_settings = BOSettings(
            param_names=OmegaConf.to_container(cfg.param_names),
            n_calls=cfg.optimisation.n_calls,
            n_initial_points=cfg.optimisation.n_initial_points,
            acq_func=cfg.optimisation.acq_func,
            xi=cfg.optimisation.xi,
            noise=cfg.optimisation.noise,
            initial_point_generator=cfg.optimisation.initial_point_generator,
            use_previous_best=cfg.optimisation.use_previous_best,
            model_samples=cfg.optimisation.model_samples,
            boundary_samples=cfg.optimisation.boundary_samples,
            posterior_samples=cfg.optimisation.posterior_samples,
        )
        log.info("Initialising BO Optimiser Context")
        optimiser_context = BOOptimiser(
            optimiser_settings=optimiser_settings,
            prior_manager=prior_manager,
            model_kwargs=model_call_kwargs,
        )
    else:
        raise ValueError(
            f"cfg.optimisation.method: {cfg.optimisation.method} not implemented"
        )

    log.info("Initialising BDA Driver")
    driver = BDADriver(
        quant_to_optimise=cfg.quant_to_optimise,
        opt_data=opt_data,
        optimiser_context=optimiser_context,
        plasma_profiler=plasma_profiler,
        model_coordinator=model_coordinator,
        prior_manager=prior_manager,
    )

    log.info("Running BDA Driver")
    _results = driver(config=config)
    if save_results:
        filepath = "./results/phantom/"
        save_pickle(_results, filepath)
        plot_bda(_results, filepath)


if __name__ == "__main__":
    bda_phantom_optimisation()
