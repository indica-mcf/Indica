import importlib
from indica.workflows.bda_run import bda_run

class TestBDARun:
    """
    Mock Equilibrium is broken in reader!
    """
    def teststub(self):
        return
    # def test_bda_run(self):
    #     config_path = "indica.workflows.configs.example_bda"
    #     config_file = importlib.import_module(config_path)
    #     bda_workflow = bda_run(**config_file.__dict__)


if __name__ == "__main__":

    test = TestBDARun()
    # test.test_bda_run()
