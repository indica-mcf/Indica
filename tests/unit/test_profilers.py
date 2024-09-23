from hydra import compose
from hydra import initialize_config_module

from indica.profilers.profiler_gauss import ProfilerGauss


class TestProfilerGauss:
    def setup_class(self):
        with initialize_config_module(
            version_base=None, config_module="indica.configs.profilers"
        ):
            cfg = compose(config_name="profiler_gauss")
        self.cfg = cfg
        self.datatypes = cfg.keys()

    def test_initialise_with_default_parameters_for_all_datatypes(self):
        for datatype in self.datatypes:
            _ = ProfilerGauss(datatype=datatype)

    def test_plots(self):
        for datatype in self.datatypes:
            prof = ProfilerGauss(datatype=datatype)
            prof.plot()

    def test_calls(self):
        for datatype in self.datatypes:
            prof = ProfilerGauss(datatype=datatype)
            prof.__call__()
            assert hasattr(prof, "ydata")

    def test_extra_parameters_during_init(self):
        for datatype in self.datatypes:
            prof = ProfilerGauss(datatype=datatype, parameters={"dummy_value": -1})
            assert getattr(prof, "dummy_value") == -1

    def test_overwriting_defaults_during_init(self):
        for datatype in self.datatypes:
            prof = ProfilerGauss(datatype=datatype, parameters={"y0": -1})
            assert getattr(prof, "y0") == -1


if __name__ == "__main__":
    test = TestProfilerGauss()
    test.setup_class()
