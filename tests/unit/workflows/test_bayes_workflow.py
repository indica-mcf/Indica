from unittest.mock import MagicMock

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
            "ion_temperature.y0",
            "ion_temperature.peaking",
            "ion_temperature.wped",
            "ion_temperature.wcenter",
        ]
        self.opt_quant = ["cxff_tws_c.ti", "cxff_pi.ti"]

        self.pulse = None
        self.tstart = 0.01
        self.tend = 0.10
        self.dt = 0.01
        self.phantoms = True

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
            nwalkers=10,
            iterations=10,
            sample_method="random",
            starting_samples=10,
            burn_frac=0.05,
            stopping_criteria="mode",
            stopping_criteria_factor=0.005,
            stopping_criteria_debug=True,
            priors=self.bayes_settings.priors,
        )

        self.data_context = MagicMock(
            pulse=self.pulse,
            diagnostics=self.diagnostics,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            reader_settings=self.data_settings,
        )
        self.data_context.opt_data.keys = MagicMock(return_value=self.opt_quant)

        self.plasma_context = MagicMock(
            plasma_settings=self.plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
        )

        self.model_context = MagicMock(
            diagnostics=self.diagnostics,
            plasma_context=self.plasma_context,
            equilibrium=self.data_context.equilibrium,
            transforms=self.data_context.transforms,
            model_settings=self.model_settings,
        )

        self.optimiser_context = MagicMock(optimiser_settings=self.optimiser_settings)

        self.workflow = MagicMock(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            blackbox_settings=self.bayes_settings,
            data_context=self.data_context,
            optimiser_context=self.optimiser_context,
            plasma_context=self.plasma_context,
            model_context=self.model_context,
        )

    def test_data_context_initialises(self):
        data_context = MockData(
            pulse=self.pulse,
            diagnostics=self.diagnostics,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            reader_settings=self.data_settings,
        )
        data_context.read_data()
        data_context.process_data(
            self.model_context._build_bckc,
        )
        assert True

    def test_plasma_context_initialises(self):
        plasma_context = PlasmaContext(
            plasma_settings=self.plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
        )
        plasma_context.init_plasma(
            equilibrium=self.data_context.equilibrium,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
        )
        plasma_context.save_phantom_profiles(phantoms=self.data_context.phantoms)
        assert True

    def test_model_context_initialises(self):
        model_context = ModelContext(
            diagnostics=self.diagnostics,
            plasma_context=self.plasma_context,
            equilibrium=self.data_context.equilibrium,
            transforms=self.data_context.transforms,
            model_settings=self.model_settings,
        )
        model_context.update_model_kwargs(self.data_context.binned_data)
        model_context.init_models()
        assert True

    def test_optimiser_context_initialises(self):
        optimiser_context = EmceeOptimiser(optimiser_settings=self.optimiser_settings)
        optimiser_context.init_optimiser(MagicMock())
        assert True

    def test_workflow_initialises(self):
        workflow = BayesWorkflow(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            blackbox_settings=self.bayes_settings,
            data_context=self.data_context,
            optimiser_context=self.optimiser_context,
            plasma_context=self.plasma_context,
            model_context=self.model_context,
        )
        workflow.dummy = ""
        assert True


if __name__ == "__main__":
    test = TestBayesWorkflow()
    test.setup_class()
    test.test_workflow_initialises()
