from pathlib import Path

from hydra import compose
from hydra import initialize

from indica.workflows.bda_run import bda_run

PROJECT_PATH = Path(__file__).parent.parent
CONFIG_PATH = f"{PROJECT_PATH}/configs"


class TestBDARun:
    def test_bda_run(
        self,
    ):
        with initialize(
            version_base=None, config_path=f"{CONFIG_PATH}/workflows/bda_run"
        ):
            cfg = compose(
                config_name="mock_run",
            )
        bda_run(cfg)


if __name__ == "__main__":
    test = TestBDARun()
    test.test_bda_run()
