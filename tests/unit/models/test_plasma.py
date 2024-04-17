import numpy as np
import copy
from indica.models.plasma import example_run


class TestPlasma:
    def setup_class(self):
        self.example_plasma = example_run()

    def test_example_plasma_initializes(self):
        assert hasattr(self, "example_plasma")

    def test_example_plasma_has_equilibrium(self):
        assert hasattr(self.example_plasma, "equilibrium")

    def test_example_plasma_volume_is_non_zero(self):
        assert len(np.where(self.example_plasma.volume > 0)[0]) != 0

    def test_all_fz_is_non_zero(self):
        for impurity in self.example_plasma.impurities:
            _fz = self.example_plasma.fz[impurity]
            assert len(np.where(_fz > 0)[0]) != 0

    def test_downstream_quantites_are_non_zero(self):
        downstream_quantities = ["ptot", "wp", "ion_density", "zeff"]
        for quantity in downstream_quantities:
            _quantity = getattr(self.example_plasma, quantity)
            assert len(np.where(_quantity > 0)[0]) != 0


    @property
    def plasma(self):
        # For doing state based testing
        return copy.deepcopy(self.example_plasma)


if __name__ == "__main__":
    test = TestPlasma()
    test.setup_class()
