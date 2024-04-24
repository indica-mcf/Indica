import importlib
import sys
from indica.workflows.bayes_workflow import BayesBBSettings
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.bayes_workflow import DEFAULT_PRIORS
from indica.workflows.bayes_workflow import DEFAULT_PROFILE_PARAMS
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.workflows.bayes_workflow import ExpData
from indica.workflows.bayes_workflow import ModelContext
from indica.workflows.bayes_workflow import ModelSettings
from indica.workflows.bayes_workflow import OptimiserEmceeSettings
from indica.workflows.bayes_workflow import PhantomData
from indica.workflows.bayes_workflow import PlasmaContext
from indica.workflows.bayes_workflow import PlasmaSettings
from indica.workflows.bayes_workflow import ReaderSettings

def bda_run(
    pulse=None,
    pulse_to_write = None,
    diagnostics=None,
    param_names=None,
    opt_quantity=None,
    phantom=False,
    best=True,
    tstart=0.02,
    tend=0.05,
    dt=0.01,
    revisions=None,
    filters=None,
    starting_samples = 100,
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
    ts_split="LFS",
    ts_R_shift=0,
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

    bayes_settings = BayesBBSettings(
        diagnostics=diagnostics,
        param_names=param_names,
        opt_quantity=opt_quantity,
        priors=DEFAULT_PRIORS,
    )

    data_settings = ReaderSettings(filters=filters, revisions=revisions)
    if phantom:
        data_context = PhantomData(
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

    plasma_settings = PlasmaSettings(**plasma_settings

    )
    plasma_context = PlasmaContext(
        plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
    )
    if profile_params_to_update:
        plasma_context.profile_params.update(profile_params_to_update)

    plasma_context.init_plasma(
        data_context.equilibrium, tstart=tstart, tend=tend, dt=dt
    )

    plasma_context.save_phantom_profiles(phantoms=data_context.phantoms)

    if set_ts:
        plasma_context.set_ts_profiles(data_context, split=ts_split, R_shift=ts_R_shift)

    model_settings = ModelSettings()
    model_settings.init_kwargs.update(model_init)

    model_context = ModelContext(
        diagnostics=diagnostics,
        plasma_context=plasma_context,
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
        param_names=bayes_settings.param_names,
        nwalkers=nwalkers,
        iterations=iterations,
        sample_method=sample_method,
        starting_samples=starting_samples,
        burn_frac=burn_frac,
        stopping_criteria=stopping_criteria,

        stopping_criteria_factor=stopping_criteria_factor,
        stopping_criteria_debug=True,
        priors=bayes_settings.priors,
    )
    optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)

    workflow = BayesWorkflow(
        tstart=tstart,
        tend=tend,
        dt=dt,
        blackbox_settings=bayes_settings,
        data_context=data_context,
        optimiser_context=optimiser_context,
        plasma_context=plasma_context,
        model_context=model_context,
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
