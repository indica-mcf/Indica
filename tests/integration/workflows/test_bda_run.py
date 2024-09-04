from hydra import compose
from hydra import initialize_config_module

from indica.workflows.bda_run import bda_run


class TestBDARun:
    def test_emcee_run(
        self,
    ):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.workflows.bda_run"
        ):
            cfg = compose(
                config_name="test_emcee",
            )
        bda_run(cfg)

    def test_bo_run(
        self,
    ):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.workflows.bda_run"
        ):
            cfg = compose(
                config_name="test_bo",
            )
        bda_run(cfg)


if __name__ == "__main__":
    test = TestBDARun()
    test.test_bo_run()
