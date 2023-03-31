import numpy as np

from indica.models.plasma import Plasma
from indica.equilibrium import fake_equilibrium_data, Equilibrium


class TestPlasmaInit:

    def setup_class(self):
        self.tstart = 0.
        self.tend = 0.1
        self.dt = 0.02
        self.plasma = Plasma(tstart=self.tstart, tend=self.tend, dt=self.dt, main_ion="h",
                             impurities=("c", "ar", "he"), impurity_concentration=(0.01, 0.001, 0.01),)
        self.equilibrium_data = fake_equilibrium_data(
            tstart=self.tstart, tend=self.tend, dt=self.dt, machine_dims=self.plasma.machine_dimensions
        )
        self.equilibrium = Equilibrium(self.equilibrium_data)
        self.plasma.set_equilibrium(equilibrium=self.equilibrium)

    def teardown_method(self):
        self.plasma.initialize_variables(tstart=self.tstart, tend=self.tend, dt=self.dt)

    def test_plasma_initializes(self):
        assert hasattr(self, "plasma")

    def test_plasma_volume_non_zero(self):
        _volume = self.plasma.volume
        assert len(np.where(_volume > 0)[0]) != 0



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
