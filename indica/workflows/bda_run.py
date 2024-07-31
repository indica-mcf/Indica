import logging
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
from indica.models import Plasma
from indica.models import ThomsonScattering
from indica.readers.read_st40 import ReadST40
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.workflows.model_coordinator import ModelCoordinator
from indica.workflows.optimiser_context import OptimiserEmceeSettings
from indica.workflows.pca import pca_workflow
from indica.workflows.plasma_profiler import initialise_gauss_profilers
from indica.workflows.plasma_profiler import PlasmaProfiler
from indica.workflows.priors import PriorManager

ERROR_FUNCTIONS = {
    "ts.ne": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    "ts.te": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    # "xrcs.intens": lambda x: np.sqrt(x)  # Poisson noise
    #                          + (x.where((x.wavelength < 0.392) &
    #                      (x.wavelength > 0.388),
    #                      ).std("wavelength")).fillna(0),  # Background noise
    "cxff_pi.ti": lambda x: x * 0 + 0.10 * x.max(dim="channel"),
    "cxff_tws_c.ti": lambda x: x * 0 + 0.20 * x.max(dim="channel"),
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
    config_name="basic_run",
)
def bda_run(
    cfg: DictConfig,
):
    log = logging.getLogger(__name__)
    log.info(f"Beginning BDA for pulse {cfg.pulse_info.pulse}")
    dirname = f"{cfg.pulse_info.pulse}.{cfg.write_info.run}"

    if cfg.data_info.mock:
        log.info("Using mock plasma")
        plasma = load_default_objects("st40", "plasma")
    else:
        log.info("Initialising plasma")
        plasma = Plasma(
            tstart=cfg.pulse_info.tstart,
            tend=cfg.pulse_info.tend,
            dt=cfg.pulse_info.dt,
            **cfg.plasma_settings,
        )

    log.info("Initialising plasma_profiler")
    profilers = initialise_gauss_profilers(xspl=plasma.rho)
    plasma_profiler = PlasmaProfiler(plasma=plasma, profilers=profilers)
    plasma_profiler(cfg.data_info.profile_params_to_update)

    if cfg.data_info.set_ts or cfg.data_info.apply_rshift:
        log.info("PPTS reading")
        ppts_reader = ReadST40(
            pulse=cfg.pulse_info.pulse + cfg.data_info.ppts_modelling_number,
            tstart=cfg.pulse_info.tstart,
            tend=cfg.pulse_info.tend,
            dt=cfg.pulse_info.dt,
            tree="ppts",
        )
        ppts_reader(
            ["ppts"],
            revisions=OmegaConf.to_container(cfg.data_info.revisions),
            fetch_equilbrium=False,
        )

    if cfg.data_info.set_ts:
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
    plasma_profiler.save_phantoms(phantom=cfg.data_info.phantom)

    # if binned data is used then interpolating onto equil.t
    # and back to plasma.t causes some time points to be lost
    R_shift = 0.0
    if cfg.data_info.apply_rshift:
        R_shift = ppts_reader.raw_data["ppts"]["R_shift"]
        log.info(f"R shift of: {R_shift}")

    if cfg.data_info.mock:
        log.info("Using mock equilibrium")
        equilibrium = load_default_objects("st40", "equilibrium")
    else:
        log.info("Reading equilibrium")
        equil_reader = ReadST40(
            pulse=cfg.data_info.equilibrium.modelling_number + cfg.pulse_info.pulse,
            tstart=cfg.pulse_info.tstart,
            tend=cfg.pulse_info.tend,
            dt=cfg.pulse_info.dt,
        )
        equil_reader(
            [cfg.data_info.equilibrium.code],
            revisions=OmegaConf.to_container(cfg.data_info.equilibrium.revisions),
            R_shift=R_shift,
        )
        equilibrium = equil_reader.equilibrium

    plasma.set_equilibrium(equilibrium=equilibrium)

    # different data handling methods TODO: abstract these to an interface

    if cfg.data_info.mock:
        log.info("Using mock data reader strategy")
        transforms = load_default_objects("st40", "geometry")
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.pulse_info.diagnostics}
        reader = ModelCoordinator(
            models,
            OmegaConf.to_container(cfg.model_info),
        )
        reader.set_transforms(transforms)
        reader.set_equilibrium(
            equilibrium,
        )
        reader.set_plasma(plasma)

    elif cfg.data_info.phantom:  # Currently broken
        log.info("Using phantom reader strategy")
        phantom_reader = ReadST40(
            pulse=cfg.pulse_info.pulse,
            tstart=cfg.pulse_info.tstart,
            tend=cfg.pulse_info.tend,
            dt=cfg.pulse_info.dt,
        )
        phantom_reader(
            cfg.pulse_info.diagnostics,
            filter_coords=cfg.data_info.filter_coords,
            filter_limits=cfg.data_info.filter_limits,
            revisions=OmegaConf.to_container(cfg.data_info.revisions),
            R_shift=R_shift,
        )
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.pulse_info.diagnostics}
        reader = ModelCoordinator(
            models,
            OmegaConf.to_container(cfg.model_info),
        )
        if "xrcs" in phantom_reader.binned_data.keys():
            more_model_settings = {
                "xrcs": {
                    "window": phantom_reader.binned_data["xrcs"]["intens"].wavelength
                }
            }
        else:
            more_model_settings = {}
        reader.init_models(**more_model_settings)
        reader.set_transforms(phantom_reader.transforms)
        reader.set_equilibrium(
            equilibrium,
        )
        reader.set_plasma(plasma)

    else:
        log.info("Using default reader strategy")
        reader = ReadST40(
            pulse=cfg.pulse_info.pulse,
            tstart=cfg.pulse_info.tstart,
            tend=cfg.pulse_info.tend,
            dt=cfg.pulse_info.dt,
        )

    reader(
        cfg.pulse_info.diagnostics,
        filter_coords=cfg.data_info.filter_coords,
        filter_limits=cfg.data_info.filter_limits,
        revisions=OmegaConf.to_container(cfg.data_info.revisions),
        R_shift=R_shift,
        fetch_equilbrium=False,
    )

    # post processing (TODO: where should this be)
    flat_data = flatdict.FlatDict(reader.binned_data, ".")
    log.info("Applying error to opt_data")
    opt_data = add_error_to_opt_data(flat_data, verbose=False)

    models = {diag: INSTRUMENT_MAPPING[diag] for diag in cfg.pulse_info.diagnostics}
    log.info("Initialising ModelCoordinator")
    model_coordinator = ModelCoordinator(
        models=models,
        model_settings=OmegaConf.to_container(cfg.model_info),
        verbose=False,
    )
    if "xrcs" in reader.binned_data.keys():
        more_model_settings = {
            "xrcs": {
                "window": reader.binned_data["xrcs"]["intens"].wavelength,
                "background": reader.binned_data["xrcs"]["background"],
            },
        }
    else:
        more_model_settings = {}

    model_kwargs = {}
    model_coordinator.init_models(**more_model_settings)
    model_coordinator.set_transforms(reader.transforms)
    model_coordinator.set_equilibrium(equilibrium)
    model_coordinator.set_plasma(plasma)

    log.info("initialising PriorManager")
    prior_manager = PriorManager(**cfg.priors)

    if any(cfg.optimisation_info.pca_profiles):
        log.info(f"Using PCA profiles: {cfg.optimisation_info.pca_profiles}")
        pca_processor, pca_profilers = pca_workflow(
            prior_manager,
            cfg.optimisation_info.pca_profiles,
            plasma.rho,
            n_components=cfg.optimisation_info.pca_components,
            num_prof_samples=int(5e3),
        )
        prior_manager.update_priors(pca_processor.compound_priors)
        plasma_profiler.update_profilers(pca_profilers)
        opt_params = [
            param
            for param in cfg.optimisation_info.param_names
            if param.split(".")[0] not in cfg.optimisation_info.pca_profiles
        ]
        opt_params.extend(
            prior_manager.get_param_names_for_profiles(
                cfg.optimisation_info.pca_profiles
            )
        )
        log.info(f"optimising with: {opt_params}")
    else:
        opt_params = list(cfg.optimisation_info.param_names)

    optimiser_settings = OptimiserEmceeSettings(
        param_names=opt_params,
        nwalkers=cfg.optimisation_info.nwalkers,
        iterations=cfg.optimisation_info.iterations,
        sample_method=cfg.optimisation_info.sample_method,
        starting_samples=cfg.optimisation_info.starting_samples,
        burn_frac=cfg.optimisation_info.burn_frac,
        stopping_criteria=cfg.optimisation_info.stopping_criteria,
        stopping_criteria_factor=cfg.optimisation_info.stopping_criteria_factor,
        stopping_criteria_debug=True,
    )

    log.info("Initialising Optimiser Context")
    optimiser_context = EmceeOptimiser(
        optimiser_settings=optimiser_settings,
        prior_manager=prior_manager,
        model_kwargs=model_kwargs,
    )

    workflow = BayesWorkflow(
        quant_to_optimise=cfg.pulse_info.quant_to_optimise,
        opt_data=opt_data,
        optimiser_context=optimiser_context,
        plasma_profiler=plasma_profiler,
        model_coordinator=model_coordinator,
        prior_manager=prior_manager,
    )

    log.info("Running BDA")
    workflow(
        pulse_to_write=cfg.pulse_info.pulse_to_write,
        run=cfg.write_info.run,
        run_info=cfg.write_info.run_info,
        best=cfg.write_info.best,
        mds_write=cfg.write_info.mds_write,
        plot=cfg.write_info.plot,
        filepath=f"./results/{dirname}/",
        config=pprint.pformat(dict(cfg)),
    )


if __name__ == "__main__":
    bda_run()
