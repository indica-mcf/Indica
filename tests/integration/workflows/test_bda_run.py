from indica.workflows.bda_run import bda_run


class TestBDARun:
    def test_bda_run(
        self,
    ):
        bda_run()


if __name__ == "__main__":
    test = TestBDARun()
    test.test_bda_run()
