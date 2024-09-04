import logging
import pprint

import flatdict
import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import Plasma
from indica.models import ThomsonScattering
from indica.readers.read_st40 import ReadST40
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.bayes_workflow import EmceeOptimiser, BOOptimiser
from indica.workflows.model_coordinator import ModelCoordinator
from indica.workflows.optimiser_context import EmceeSettings, BOSettings
from indica.workflows.pca import pca_workflow
from indica.workflows.plasma_profiler import initialise_gauss_profilers
from indica.workflows.plasma_profiler import PlasmaProfiler
from indica.workflows.priors import PriorManager

ERROR_FUNCTIONS = {
    "ts.ne": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    "ts.te": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    "xrcs.raw_spectra": lambda x: x * 0.05 + 0.01 * x.max(dim="wavelength") + (
        x.where(
            (x.wavelength < 0.392) & (x.wavelength > 0.388),
        ).std("wavelength")
    ).fillna(
        0
    ),
    "cxff_pi.ti": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    "cxff_tws_c.ti": lambda x: x * 0 + 0.10 * x.max(dim="channel"),
}


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
        # TODO: find better way of handling errors for exp + phantom cases
        # if "error" in value.coords:
        #     if verbose:
        #         print(f"{key} contains error: skipping")
        #     opt_data_with_error[key] = value
        #     continue
        error = error_functions[key](value)
        opt_data_with_error[key] = value.assign_coords({"error": error})
    return opt_data_with_error


INSTRUMENT_MAPPING: dict = {
    "xrcs": HelikeSpectrometer,
    "cxff_pi": ChargeExchangeSpectrometer,
    "cxff_tws_c": ChargeExchangeSpectrometer,
    "smmh1": Interferometer,
    "efit": EquilibriumReconstruction,
    "ts": ThomsonScattering,
}


@hydra.main(
    version_base=None,
    config_path="../configs/workflows/bda_run",
    config_name="test_mock",
)
def bda_run(
    cfg: DictConfig,
):
    if cfg.writer.pulse_to_write is None:
        cfg.writer.pulse_to_write = cfg.pulse

    log = logging.getLogger(__name__)
    log.info(f"Beginning BDA for pulse {cfg.pulse}")
    dirname = f"{cfg.pulse}.{cfg.writer.run}"

    log.info("Initialising plasma")
    plasma = Plasma(
        tstart=cfg.tstart,
        tend=cfg.tend,
        dt=cfg.dt,
        **cfg.plasma.settings,
    )

    log.info("Initialising plasma_profiler")
    profilers = initialise_gauss_profilers(
        plasma.rho,
        profile_names=cfg.plasma.profiles.keys(),
        profile_params=OmegaConf.to_container(cfg.plasma.profiles),
    )
    plasma_profiler = PlasmaProfiler(plasma=plasma, profilers=profilers)
    plasma_profiler()

    if cfg.reader.set_ts or cfg.reader.apply_rshift:
        log.info("PPTS reading")
        ppts_reader = ReadST40(
            pulse=cfg.pulse + cfg.reader.ppts_modelling_number,
            tstart=cfg.tstart,
            tend=cfg.tend,
            dt=cfg.dt,
            tree="ppts",
        )
        ppts_reader(
            ["ppts"],
            revisions=OmegaConf.to_container(cfg.reader.revisions),
            fetch_equilbrium=False,
        )

    if cfg.reader.set_ts:
        # interp ppts profiles as some are empty
        log.info("Setting profiles from PPTS")
        ppts_profs = ppts_reader.filtered_data["ppts"]
        ne = (
            ppts_profs["ne_rho"]
            .interpolate_na(dim="t")
            .ffill("t")
            .bfill("t")
            .interp(
                t=plasma.t,
            )
            .interp(rho_poloidal=plasma.rho)
        )
        te = (
            ppts_profs["te_rho"]
            .interpolate_na(dim="t")
            .ffill("t")
            .bfill("t")
            .interp(
                t=plasma.t,
            )
            .interp(rho_poloidal=plasma.rho)
        )
        plasma_profiler.set_profiles(
            {
                "electron_density": ne,
                "electron_temperature": te,
            }
        )
    plasma_profiler.save_phantoms(phantom=cfg.reader.phantom)

    # if binned data is used then interpolating onto equil.t
    # and back to plasma.t causes some time points to be lost
    R_shift = 0.0
    if cfg.reader.apply_rshift:
        R_shift = ppts_reader.raw_data["ppts"]["R_shift"]
        log.info(f"R shift of: {R_shift}")

    if cfg.reader.mock:
        log.info("Using mock equilibrium")
        equilibrium = load_default_objects("st40", "equilibrium")
    else:
        log.info("Reading equilibrium")
        equil_reader = ReadST40(
            pulse=cfg.reader.equilibrium.modelling_number + cfg.pulse,
            tstart=cfg.tstart,
            tend=cfg.tend,
            dt=cfg.dt,
        )
        equil_reader(
            [cfg.reader.equilibrium.code],
            revisions=OmegaConf.to_container(cfg.reader.equilibrium.revisions),
            R_shift=R_shift,
        )
        equilibrium = equil_reader.equilibrium

    plasma.set_equilibrium(equilibrium=equilibrium)

    if cfg.reader.mock:
        log.info("Using mock data reader strategy")
        transforms = load_default_objects("st40", "geometry")
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.model.diagnostics}
        reader = ModelCoordinator(
            models,
            OmegaConf.to_container(cfg.model.settings),
        )
        reader.set_transforms(transforms)
        reader.set_equilibrium(
            equilibrium,
        )
        reader.set_plasma(plasma)

    elif cfg.reader.phantom:
        log.info("Using phantom reader strategy")
        phantom_reader = ReadST40(
            pulse=cfg.pulse,
            tstart=cfg.tstart,
            tend=cfg.tend,
            dt=cfg.dt,
        )
        phantom_reader(
            cfg.diagnostics,
            revisions=OmegaConf.to_container(cfg.data_info.revisions),
            R_shift=R_shift,
            **cfg.reader.filters,
        )
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.pulse_info.diagnostics}
        if "xrcs" in phantom_reader.binned_data.keys():
            more_model_settings = {
                "xrcs": {
                    "window": phantom_reader.binned_data["xrcs"]["intens"].wavelength
                }
            }
        else:
            more_model_settings = {}

        reader = ModelCoordinator(
            models,
            {**OmegaConf.to_container(cfg.model.settings), **more_model_settings},
        )
        reader.set_transforms(phantom_reader.transforms)
        reader.set_equilibrium(
            equilibrium,
        )
        reader.set_plasma(plasma)

    else:
        log.info("Using default reader strategy")
        reader = ReadST40(
            pulse=cfg.pulse,
            tstart=cfg.tstart,
            tend=cfg.tend,
            dt=cfg.dt,
        )

    reader(
        cfg.model.diagnostics,
        revisions=OmegaConf.to_container(cfg.reader.revisions),
        R_shift=R_shift,
        fetch_equilbrium=False,
        **cfg.reader.filters,
    )

    # post processing (TODO: where should this be)
    flat_data = flatdict.FlatDict(reader.binned_data, ".")
    log.info("Applying error to opt_data")
    opt_data = add_error_to_opt_data(flat_data, verbose=False)

    models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.model.diagnostics}
    if "xrcs" in reader.binned_data.keys():
        more_model_settings = {
            "xrcs": {
                "window": reader.binned_data["xrcs"]["raw_spectra"].wavelength,
                "background": reader.binned_data["xrcs"].get("background", 0),
            },
        }
    else:
        more_model_settings = {}
    log.info("Initialising ModelCoordinator")
    model_coordinator = ModelCoordinator(
        models=models,
        model_settings={
            **OmegaConf.to_container(cfg.model.settings),
            **more_model_settings,
        },
        verbose=False,
    )

    model_call_kwargs = {"xrcs": {"norm_y": reader.binned_data["xrcs"]["raw_spectra"].max("wavelength")}
                         }
    model_coordinator.set_transforms(reader.transforms)
    model_coordinator.set_equilibrium(equilibrium)
    model_coordinator.set_plasma(plasma)

    log.info("initialising PriorManager")
    prior_manager = PriorManager(**cfg.priors)

    if any(cfg.optimisation.pca_profiles):
        log.info(f"Using PCA profiles: {cfg.optimisation.pca_profiles}")
        pca_processor, pca_profilers = pca_workflow(
            prior_manager,
            cfg.optimisation.pca_profiles,
            plasma.rho,
            n_components=cfg.optimisation.pca_components,
            num_prof_samples=int(5e3),
        )
        prior_manager.update_priors(pca_processor.compound_priors)
        plasma_profiler.update_profilers(pca_profilers)
        opt_params = [
            param
            for param in cfg.optimisation.param_names
            if param.split(".")[0] not in cfg.optimisation.pca_profiles
        ]
        opt_params.extend(
            prior_manager.get_param_names_for_profiles(cfg.optimisation.pca_profiles)
        )
        log.info(f"optimising with: {opt_params}")
    else:
        opt_params = list(cfg.optimisation.param_names)

    if cfg.optimisation.method == "emcee":
        optimiser_settings = EmceeSettings(
            param_names=opt_params,
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
        optimiser_settings = BOSettings(param_names=opt_params,
                                        n_calls=cfg.optimisation.n_calls,
                                        n_initial_points=cfg.optimisation.n_initial_points,
                                        acq_func=cfg.optimisation.acq_func,
                                        xi=cfg.optimisation.xi,
                                        noise=cfg.optimisation.noise,
                                        initial_point_generator=cfg.optimisation.initial_point_generator,
                                        use_previous_best=cfg.optimisation.use_previous_best,
                                        )
        log.info("Initialising BO Optimiser Context")
        optimiser_context = BOOptimiser(optimiser_settings,
                                        prior_manager,
                                        model_kwargs=model_call_kwargs, )
    else:
        raise ValueError(f"cfg.optimisation.method: {cfg.optimisation.method} not implemented")

    workflow = BayesWorkflow(
        quant_to_optimise=cfg.model.quantities,
        opt_data=opt_data,
        optimiser_context=optimiser_context,
        plasma_profiler=plasma_profiler,
        model_coordinator=model_coordinator,
        prior_manager=prior_manager,
    )

    log.info("Running BDA")
    workflow(
        pulse_to_write=cfg.writer.pulse_to_write,
        run=cfg.writer.run,
        run_info=cfg.writer.run_info,
        best=cfg.writer.best,
        mds_write=cfg.writer.mds_write,
        plot=cfg.writer.plot,
        filepath=f"./results/{dirname}/",
        config=pprint.pformat(dict(cfg)),
    )


if __name__ == "__main__":
    bda_run()
