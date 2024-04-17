import numpy as np

from indica.equilibrium import Equilibrium
from indica.equilibrium import fake_equilibrium_data
from indica.models.plasma import Plasma, example_run


class TestPlasma:
    def setup_class(self):

        self.example_plasma = example_run()

    def test_example_plasma_initializes(self):
        assert hasattr(self, "example_plasma")

    def test_example_plasma_has_equilibrium(self):
        assert hasattr(self.example_plasma, "equilibrium")

    def test_example_plasma_volume_is_non_zero(self):
        assert len(np.where(self.example_plasma.volume > 0)[0]) != 0



    # def test_fz_is_non_zero(self):
    #     _fz = self.plasma.fz[self.impurities[0]]
    #     assert len(np.where(_fz > 0)[0]) != 0
    #

if __name__ == "__main__":
    test = TestPlasma()
    test.setup_class()
