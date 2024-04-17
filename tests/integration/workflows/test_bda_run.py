from indica.workflows.configs import test_bda
from indica.workflows.bda_run import bda_run

class TestBDARun:
    def test_bda_run(self):
        self.bda_workflow = bda_run(test_bda.__dict__)


if __name__ == "__main__":

    test = TestBDARun()
    test.test_bda_run()
