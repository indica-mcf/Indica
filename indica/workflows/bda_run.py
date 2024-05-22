import importlib
import sys

from indica.models import Plasma
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.optimiser_context import OptimiserEmceeSettings
from indica.workflows.priors import PriorManager
from indica.workflows.plasma_profiler import PlasmaProfiler, initialise_gauss_profilers
from indica.workflows.model_coordinator import ModelCoordinator
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.workflows.data_context import ExpData
from indica.workflows.data_context import MockData
from indica.workflows.data_context import PhantomData



def bda_run(
    pulse=None,
    pulse_to_write=None,
    diagnostics=None,
    param_names=None,
    opt_quantity=None,
    phantom=False,
    mock=False,
    best=True,
    tstart=0.02,
    tend=0.05,
    dt=0.01,
    revisions=None,
    filters=None,
    starting_samples=100,
    iterations=500,
    nwalkers=50,
    stopping_criteria_factor=0.002,
    burn_frac=0.20,
    stopping_criteria="mode",
    sample_method="high_density",
    mds_write=False,
    plot=False,
    run="RUN01",
    run_info="Default run",
    dirname=None,
    set_ts=False,
    profile_params_to_update=None,
    model_init=None,
    plasma_settings=None,
    **kwargs,
):
    if pulse_to_write is None:
        pulse_to_write = 43000000 + pulse

    if filters is None:
        filters = {}
    if revisions is None:
        revisions = {}
    if model_init is None:
        model_init = {}
    if profile_params_to_update is None:
        profile_params_to_update = {}
    if not all([pulse, diagnostics, param_names, opt_quantity, plasma_settings]):
        raise ValueError("Not all inputs defined")

    if dirname is None:
        dirname = f"{pulse}.{run}"

    data_settings = {}
    if phantom:
        data_context = PhantomData(
            pulse=pulse,
            diagnostics=diagnostics,
            tstart=tstart,
            tend=tend,
            dt=dt,
            reader_settings=data_settings,
        )
    elif mock:
        data_context = MockData(
            pulse=pulse,
            diagnostics=diagnostics,
            tstart=tstart,
            tend=tend,
            dt=dt,
            reader_settings=data_settings,
        )
    else:
        data_context = ExpData(
            pulse=pulse,
            diagnostics=diagnostics,
            tstart=tstart,
            tend=tend,
            dt=dt,
            reader_settings=data_settings,
        )

    data_context.read_data()

    plasma = Plasma(**plasma_settings, tstart=tstart, tend=tend, dt=dt, equilibrium=data_context.equilibrium)
    profilers = initialise_gauss_profilers(xspl = plasma.rho)
    plasma_profiler = PlasmaProfiler(plasma=plasma, profilers=profilers)

    plasma_profiler(profile_params_to_update)
    plasma_profiler.save_phantoms(phantoms=data_context.phantoms)

    if set_ts:
        _profs = data_context.binned_data["ppts"]
        plasma_profiler.set_profiles({"electron_density": _profs["ne_rho"],
                                      "electron_temperature": _profs["te_rho"],
                                      })


    model_settings = ModelSettings()
    model_settings.init_kwargs.update(model_init)

    model_coordinator = ModelCoordinator(
        diagnostics=diagnostics,
        plasma=plasma,
        equilibrium=data_context.equilibrium,
        transforms=data_context.transforms,
        model_settings=model_settings,
    )
    model_context.update_model_kwargs(data_context.binned_data)
    model_context.init_models()

    data_context.process_data(
        model_context._build_bckc,
    )

    optimiser_settings = OptimiserEmceeSettings(
        param_names=param_names,
        nwalkers=nwalkers,
        iterations=iterations,
        sample_method=sample_method,
        starting_samples=starting_samples,
        burn_frac=burn_frac,
        stopping_criteria=stopping_criteria,
        stopping_criteria_factor=stopping_criteria_factor,
        stopping_criteria_debug=True,
        prior_manager=prior_manager,
    )

    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(
        quant_to_optimise=None,
        data_context=data_context,
        optimiser_context=optimiser_context,
        plasma_profiler=plasma_profiler,
        model_coordinator=model_coordinator,
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
        config_name = "example_bda"
    else:
        config_name = sys.argv[1]

    print(f"using config file: {config_name}")
    config_path = f"indica.workflows.configs.{config_name}"
    config_file = importlib.import_module(config_path)
    bda_run(**config_file.__dict__)
