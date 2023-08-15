import numpy as np

from indica.equilibrium import Equilibrium
from indica.equilibrium import fake_equilibrium_data
from indica.models.plasma import Plasma


class TestPlasma:
    def setup_class(self):
        self.tstart = 0.0
        self.tend = 0.1
        self.dt = 0.02
        self.impurities = ("c", "ar", "he")
        self.impurity_concentration = (0.01, 0.001, 0.01)

        self.plasma = Plasma(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            main_ion="h",
            impurities=self.impurities,
            impurity_concentration=self.impurity_concentration,
        )
        self.equilibrium_data = fake_equilibrium_data(
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            machine_dims=self.plasma.machine_dimensions,
        )
        self.equilibrium = Equilibrium(self.equilibrium_data)
        self.plasma.set_equilibrium(equilibrium=self.equilibrium)

    # def setup_method(self):
    #     self.plasma.electron_density = 1  # set profiles
    #     return

    def teardown_method(self):
        self.plasma.initialize_variables()

    def test_plasma_initializes(self):
        assert hasattr(self, "plasma")

    def test_equilibrium_initializes(self):
        assert hasattr(self, "equilibrium")

    def test_plasma_volume_non_zero(self):
        _volume = self.plasma.volume
        assert len(np.where(_volume > 0)[0]) != 0

    # def test_fz_is_non_zero(self):
    #     _fz = self.plasma.fz[self.impurities[0]]
    #     assert len(np.where(_fz > 0)[0]) != 0
    #
    # def test_lz_is_non_zero(self):
    #     _lz_tot = self.plasma.lz_tot[self.impurities[0]]
    #     assert len(np.where(_lz_tot > 0)[0]) != 0

    # def test_fz_one_time_point(self):
    #     return
    #
    # def test_fz_keys_match_elements(self):
    #     return


# class TestPlasmaProfiles:
#
#     def setup_class(self):
#
#     def test_update_profiles
#
#     def test_assign_profiles
#
#
#
#
# class TestCacheDependency:
#
#     def setup_class(self):
#         return
if __name__ == "__main__":
    test = TestPlasma()
    test.setup_class()
