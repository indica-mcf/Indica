import importlib
import sys
import flatdict

import numpy as np
from indica.models import *
from indica.readers.read_st40 import ReadST40
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.optimiser_context import OptimiserEmceeSettings
from indica.workflows.pca import pca_workflow
from indica.workflows.plasma_profiler import PlasmaProfiler, initialise_gauss_profilers
from indica.workflows.model_coordinator import ModelCoordinator
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.defaults.load_defaults import load_default_objects
from indica.workflows.priors import PriorManager

ERROR_FUNCTIONS = {
    "ts.ne": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    "ts.te": lambda x: x * 0 + 0.05 * x.max(dim="channel"),
    # "xrcs.intens": lambda x: np.sqrt(x)  # Poisson noise
    #                          + (x.where((x.wavelength < 0.392) &
    #                                     (x.wavelength > 0.388), ).std("wavelength")).fillna(0),  # Background noise
    "cxff_pi.ti": lambda x: x * 0 + 0.10 * x.max(dim="channel"),
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
        if "error" in value.coords:
            if verbose:
                print(f"{key} contains error: overwriting")
            opt_data_with_error[key] = value
            value["error"] = error_functions[key](value)
            continue
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


def bda_run(
        pulse: int = None,
        pulse_to_write: int = None,
        diagnostics: list = None,
        param_names: list = None,
        pca_profiles: list = None,
        pca_components: int = 3,
        quant_to_optimise: list = None,
        phantom=False,
        mock=False,
        tstart=0.02,
        tend=0.05,
        dt=0.01,
        revisions=None,
        filter_limits=None,
        filter_coords=None,
        R_shift=0.0,
        apply_rshift=True,
        profile_params_to_update=None,
        model_init=None,
        plasma_settings=None,
        starting_samples=100,
        iterations=500,
        nwalkers=50,
        stopping_criteria_factor=0.002,
        burn_frac=0.20,
        stopping_criteria="mode",
        sample_method="high_density",
        mds_write=False,
        plot=False,
        best=True,
        run="RUN01",
        run_info="Default run",
        dirname=None,
        set_ts=False,
        **kwargs,
):
    if model_init is None:
        model_init = {}
    if profile_params_to_update is None:
        profile_params_to_update = {}
    if plasma_settings is None:
        plasma_settings = dict(main_ion="h", impurities=("ar",))
    if not all([pulse, diagnostics, param_names, quant_to_optimise, ]):
        raise ValueError("Not all inputs defined")
    if pca_profiles is None:
        pca_profiles = []

    if pulse_to_write is None:
        pulse_to_write = 43000000 + pulse
    if dirname is None:
        dirname = f"{pulse}.{run}"

    # Get plasma set up as phantom / model readers require it
    plasma = Plasma(tstart=tstart, tend=tend, dt=dt, **plasma_settings)
    profilers = initialise_gauss_profilers(xspl=plasma.rho)
    plasma_profiler = PlasmaProfiler(plasma=plasma, profilers=profilers)
    plasma_profiler(profile_params_to_update)

    ppts_reader = ReadST40(pulse=pulse, tstart=tstart, tend=tend, dt=dt, )
    ppts_reader(["ppts"])

    if set_ts:
        # interp ppts profiles as some are empty
        ppts_profs = ppts_reader.filtered_data["ppts"]
        ne = ppts_profs["ne_rho"].interp(t=plasma.t, method="nearest").interp(rho_poloidal=plasma.rho)
        te = ppts_profs["te_rho"].interp(t=plasma.t, method="nearest").interp(rho_poloidal=plasma.rho)
        plasma_profiler.set_profiles({"electron_density": ne,
                                      "electron_temperature": te,
                                      })
    plasma_profiler.save_phantoms(phantom=phantom)

    # if binned data is used then interpolating onto equil.t and back to plasma.t causes some time points to be lost
    if apply_rshift:
        R_shift = ppts_reader.raw_data["ppts"]["R_shift"]

    # different data handling methods TODO: abstract these to an interface
    if phantom:
        phantom_reader = ReadST40(pulse=pulse, tstart=tstart, tend=tend, dt=dt)
        phantom_reader(diagnostics, filter_coords=filter_coords, filter_limits=filter_limits,
                       revisions=revisions, R_shift=R_shift)
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in diagnostics}
        reader = ModelCoordinator(models, model_init, )
        if "xrcs" in phantom_reader.binned_data.keys():
            more_model_settings = {"xrcs": {"window": phantom_reader.binned_data["xrcs"]["intens"].wavelength}}
        else:
            more_model_settings = {}
        reader.init_models(**more_model_settings)
        reader.set_transforms(phantom_reader.transforms)
        reader.set_equilibrium(phantom_reader.equilibrium, )
        reader.set_plasma(plasma)

    elif mock:
        equilibrium = load_default_objects("st40", "equilibrium")
        transforms = load_default_objects("st40", "geometry")
        models = {diag: INSTRUMENT_MAPPING[diag] for diag in diagnostics}
        reader = ModelCoordinator(models, model_init, )
        reader.set_transforms(transforms)
        reader.set_equilibrium(equilibrium, )
        reader.set_plasma(plasma)

    else:
        reader = ReadST40(pulse=pulse, tstart=tstart, tend=tend, dt=dt, )

    reader(diagnostics, filter_coords=filter_coords, filter_limits=filter_limits,
           revisions=revisions, R_shift=R_shift)
    plasma.set_equilibrium(equilibrium=reader.equilibrium)

    # post processing (TODO: where should this be)
    flat_data = flatdict.FlatDict(reader.binned_data, ".")
    opt_data = add_error_to_opt_data(flat_data, verbose=False)

    models = {diag: INSTRUMENT_MAPPING[diag] for diag in diagnostics}
    model_coordinator = ModelCoordinator(
        models=models,
        model_settings=model_init,
        verbose=False,
    )
    if "xrcs" in reader.binned_data.keys():
        more_model_settings = {"xrcs": {
            "window": reader.binned_data["xrcs"]["intens"].wavelength,
            "background": reader.binned_data["xrcs"]["background"],
        },
        }
    else:
        more_model_settings = {}

    model_kwargs = {}
    model_coordinator.init_models(**more_model_settings)
    model_coordinator.set_transforms(reader.transforms)
    model_coordinator.set_equilibrium(reader.equilibrium)
    model_coordinator.set_plasma(plasma)

    prior_manager = PriorManager()

    if any(pca_profiles):
        pca_processor, pca_profilers = pca_workflow(prior_manager, pca_profiles,
                                                    plasma.rho, n_components=pca_components, num_prof_samples=int(5e3))
        prior_manager.update_priors(pca_processor.compound_priors)
        plasma_profiler.update_profilers(pca_profilers)
        opt_params = [param for param in param_names if param.split(".")[0] not in pca_profiles]
        opt_params.extend(prior_manager.get_param_names_for_profiles(pca_profiles))
        print(f"optimising with: {opt_params}")
    else:
        opt_params = param_names

    optimiser_settings = OptimiserEmceeSettings(
        param_names=opt_params,
        nwalkers=nwalkers,
        iterations=iterations,
        sample_method=sample_method,
        starting_samples=starting_samples,
        burn_frac=burn_frac,
        stopping_criteria=stopping_criteria,
        stopping_criteria_factor=stopping_criteria_factor,
        stopping_criteria_debug=True,
    )

    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings, prior_manager=prior_manager,
                                       model_kwargs=model_kwargs,
                                       )

    workflow = BayesWorkflow(
        quant_to_optimise=quant_to_optimise,
        opt_data=opt_data,
        optimiser_context=optimiser_context,
        plasma_profiler=plasma_profiler,
        model_coordinator=model_coordinator,
        prior_manager=prior_manager
    )

    workflow(
        pulse_to_write=pulse_to_write,
        run=run,
        run_info=run_info,
        best=best,
        mds_write=mds_write,
        plot=plot,
        filepath=f"./results/{dirname}/",
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        config_name = "example_pca_with_DIME"
    else:
        config_name = sys.argv[1]

    print(f"using config file: {config_name}")
    config_path = f"indica.workflows.configs.{config_name}"
    config_file = importlib.import_module(config_path)
    bda_run(**config_file.__dict__)
