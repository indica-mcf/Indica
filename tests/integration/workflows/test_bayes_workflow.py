from indica.workflows.bayes_workflow import BayesBBSettings
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.bayes_workflow import DEFAULT_PRIORS
from indica.workflows.bayes_workflow import DEFAULT_PROFILE_PARAMS
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.workflows.bayes_workflow import MockData
from indica.workflows.bayes_workflow import ModelContext
from indica.workflows.bayes_workflow import ModelSettings
from indica.workflows.bayes_workflow import OptimiserEmceeSettings
from indica.workflows.bayes_workflow import PlasmaContext
from indica.workflows.bayes_workflow import PlasmaSettings
from indica.workflows.bayes_workflow import ReaderSettings


class TestBayesWorkflow:
    def setup_class(self):
        self.diagnostics = ["cxff_tws_c", "cxff_pi"]
        self.opt_params = [
            "Ti_prof.y0",
            # "Ti_prof.peaking",
            # "Ti_prof.wped",
            # "Ti_prof.wcenter",
        ]
        self.opt_quant = ["cxff_tws_c.ti", "cxff_pi.ti"]

        self.pulse = None
        self.tstart = 0.01
        self.tend = 0.02
        self.dt = 0.01

        self.plasma_settings = PlasmaSettings(
            main_ion="h",
            impurities=("ar", "c"),
            impurity_concentration=(0.001, 0.04),
            n_rad=10,
        )

        self.bayes_settings = BayesBBSettings(
            diagnostics=self.diagnostics,
            param_names=self.opt_params,
            opt_quantity=self.opt_quant,
            priors=DEFAULT_PRIORS,
        )

        self.data_settings = ReaderSettings(filters={}, revisions={})

        self.model_settings = ModelSettings(call_kwargs={"xrcs": {"pixel_offset": 0.0}})

        self.optimiser_settings = OptimiserEmceeSettings(
            param_names=self.bayes_settings.param_names,
            nwalkers=5,
            iterations=2,
            sample_method="random",
            starting_samples=2,
            burn_frac=0,
            stopping_criteria="mode",
            stopping_criteria_factor=0.005,
            stopping_criteria_debug=True,
            priors=self.bayes_settings.priors,
        )

    def test_workflow_runs(self):

        data_context = MockData(
            pulse=self.pulse,
            diagnostics=self.diagnostics,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            reader_settings=self.data_settings,
        )
        data_context.read_data()

        plasma_context = PlasmaContext(
            plasma_settings=self.plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
        )
        plasma_context.init_plasma(
            data_context.equilibrium, self.tstart, self.tend, self.dt
        )
        plasma_context.save_phantom_profiles(phantoms=data_context.phantoms)

        model_context = ModelContext(
            diagnostics=self.diagnostics,
            plasma_context=plasma_context,
            equilibrium=data_context.equilibrium,
            transforms=data_context.transforms,
            model_settings=self.model_settings,
        )
        model_context.update_model_kwargs(data_context.binned_data)
        model_context.init_models()
        data_context.process_data(
            model_context._build_bckc,
        )

        optimiser_context = EmceeOptimiser(optimiser_settings=self.optimiser_settings)

        workflow = BayesWorkflow(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            blackbox_settings=self.bayes_settings,
            data_context=data_context,
            optimiser_context=optimiser_context,
            plasma_context=plasma_context,
            model_context=model_context,
        )
        workflow(
            pulse_to_write=43000001,
            run="TEST",
            mds_write=False,
            plot=False,
            filepath="./results/test/",
        )


if __name__ == "__main__":

    test = TestBayesWorkflow()
    test.setup_class()
    test.test_workflow_runs()
