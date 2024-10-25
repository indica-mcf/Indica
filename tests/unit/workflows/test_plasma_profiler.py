import pytest

from indica import PlasmaProfiler
from indica.defaults.load_defaults import load_default_objects
from indica.profilers.profiler_gauss import initialise_gauss_profilers


class TestPlasmaProfiler:
    def setup_method(self):
        self.plasma = load_default_objects("st40", "plasma")
        self.profilers = initialise_gauss_profilers(
            self.plasma.rhop,
            [
                "electron_density",
                "ion_temperature",
                "impurity_density:ar",
            ],
        )

        self.plasma_profiler = PlasmaProfiler(
            plasma=self.plasma,
            profilers=self.profilers,
        )

    def test_change_profiler_params(self):
        self.plasma_profiler({"electron_density.y0": 1})
        assert self.plasma_profiler.profilers["electron_density"].y0 == 1
        self.plasma_profiler({"impurity_density:ar.y0": 1})
        assert self.plasma_profiler.profilers["impurity_density:ar"].y0 == 1

    def test_change_plasma_profiles(self):
        self.plasma_profiler({"electron_density.y0": 1.02e19})
        assert all(self.plasma.electron_density.sel(rhop=0) == 1.02e19)

        self.plasma_profiler({"impurity_density:ar.y0": 1.02e19})
        assert all(self.plasma.impurity_density.sel(element="ar", rhop=0) == 1.02e19)

    def test_change_plasma_profiles_outside_time_range_fails(self):
        with pytest.raises(KeyError):
            self.plasma_profiler({"electron_density.y0": 1.02e19}, t=10)


if __name__ == "__main__":
    test = TestPlasmaProfiler()
    test.setup_method()
