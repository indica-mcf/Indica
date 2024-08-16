from hydra import compose
from hydra import initialize_config_module

from indica.workflows.priors import PriorManager
from indica.workflows.priors import sample_from_priors


class TestPriors:
    def setup_class(self):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.workflows.priors"
        ):
            self.cfg = compose(config_name="config")

    def test_prior_manager_initalises_with_config(self):
        PriorManager(**self.cfg)
        assert True

    def test_prior_evaluates(self):
        pm = PriorManager(**self.cfg)
        pm.ln_prior({"electron_density.y0": 1e20, "electron_density.y1": 1e19})
        assert True

    def test_sampling_from_priors(self):
        pm = PriorManager(**self.cfg)
        sample_from_priors(["electron_density.y0", "electron_density.y1"], pm.priors)
        assert True
