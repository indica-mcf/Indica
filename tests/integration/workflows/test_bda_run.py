from hydra import compose
from hydra import initialize

from indica.workflows.bda_run import bda_run


class TestBDARun:
    def test_bda_run(
        self,
    ):
        with initialize(
            version_base=None, config_path="../../../indica/configs/workflows/bda_run"
        ):
            cfg = compose(
                config_name="mock_run", overrides=["write_info.mds_write=False"]
            )
        bda_run(cfg)


if __name__ == "__main__":
    test = TestBDARun()
    test.test_bda_run()
