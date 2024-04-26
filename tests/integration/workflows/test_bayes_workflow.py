from indica.defaults.read_write_defaults import load_default_objects
from indica.workflows.bayes_workflow import BayesWorkflow
from indica.workflows.bayes_workflow import DEFAULT_DIAG_NAMES
from indica.workflows.bayes_workflow import DEFAULT_PROFILE_PARAMS
from indica.workflows.bayes_workflow import EmceeOptimiser
from indica.workflows.bayes_workflow import MockData
from indica.workflows.bayes_workflow import ModelContext
from indica.workflows.bayes_workflow import ModelSettings
from indica.workflows.bayes_workflow import OptimiserEmceeSettings
from indica.workflows.bayes_workflow import PlasmaContext
from indica.workflows.bayes_workflow import PlasmaSettings
from indica.workflows.bayes_workflow import ReaderSettings
from unittest.mock import MagicMock

config = dict(
    tstart = 0.05,
    tend = 0.10,
    dt = 0.01,
)


class TestBayesWorkflow:
    def setup_class(self):
        self.plasma = load_default_objects("st40", "plasma")
        self.equilibrium = load_default_objects("st40", "equilibrium")
        self.plasma.set_equilibrium(self.equilibrium)

    def test_mockdata_initialises(self):

        reader_settings = ReaderSettings()
        data_context = MockData(
            pulse=None,
            diagnostics=DEFAULT_DIAG_NAMES,
            reader_settings=reader_settings,
            **config
        )
        data_context.read_data()

    def test_plasma_context_initialises(self):
        plasma_settings = PlasmaSettings()
        plasma_context = PlasmaContext(
            plasma_settings=plasma_settings, profile_params=DEFAULT_PROFILE_PARAMS
        )
        plasma_context.init_plasma(
            self.equilibrium, **config,
        )
        plasma_context.save_phantom_profiles(phantoms=True)


    def test_model_context_initialises(self):

        model_settings = ModelSettings()
        model_context = ModelContext(
            diagnostics=DEFAULT_DIAG_NAMES,
            plasma_context=MagicMock(),
            equilibrium=self.equilibrium,
            transforms=MagicMock(),
            model_settings=model_settings,
        )

        model_context.update_model_kwargs(MagicMock())
        model_context.init_models()

    def test_optimiser_context_initialises(self):
        optimiser_settings = OptimiserEmceeSettings(param_names=MagicMock(), priors=MagicMock())
        optimiser_context = EmceeOptimiser(optimiser_settings=optimiser_settings)


    def test_bayes_workflow_initialises(self):
        # bayes_settings = MagicMock()
        workflow = BayesWorkflow(
            **config,
            blackbox_settings=MagicMock(),
            data_context=MagicMock(),
            optimiser_context=MagicMock(),
            plasma_context=MagicMock(),
            model_context=MagicMock(),

        )


if __name__ == "__main__":

    test = TestBayesWorkflow()
    test.setup_class()
