from hydra import compose
from hydra import initialize_config_module

from indica.workflows.bda_phantom_optimisation import bda_phantom_optimisation


class TestBDARun:
    def test_emcee_run(
        self,
    ):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.workflows.bda"
        ):
            cfg = compose(
                config_name="test_emcee",
            )
        bda_phantom_optimisation(cfg)

    def test_bo_run(
        self,
    ):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.workflows.bda"
        ):
            cfg = compose(
                config_name="test_bo",
            )
        bda_phantom_optimisation(cfg)
