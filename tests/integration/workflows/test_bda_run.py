from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir

import indica
from indica.workflows.bda_run import bda_run

PROJECT_PATH = Path(indica.__file__)
CONFIG_PATH = f"{PROJECT_PATH}/../configs"

# TODO: Can't get configs to work properly in CI/CD
# class TestBDARun:
#     def test_bda_run(
#         self,
#     ):
#         with initialize_config_dir(
#             version_base=None, config_dir=f"{CONFIG_PATH}/workflows/bda_run"
#         ):
#             cfg = compose(
#                 config_name="mock_run",
#             )
#         bda_run(cfg)
#
#
# if __name__ == "__main__":
#     test = TestBDARun()
#     test.test_bda_run()
