import numpy as np

from indica.equilibrium import fake_equilibrium


class TestEquilibrium:
    def setup_class(self):
        self.tstart = 70
        self.tend = 90
        self.equilib = fake_equilibrium(tstart=self.tstart, tend=self.tend)
        self.rho = np.linspace(0.0, 1.0, 5)
        self.t = np.array([75.0, 77.5, 80.0])

    def test_cross_sectional_area(self):
        # Testing single input to cross_sectional_area()
        single_area, _ = self.equilib.cross_sectional_area(self.rho[2], self.t[1])
        single_area_actual = self.equilib.area.interp(
            rho_poloidal=self.rho[2], t=self.t[1]
        )

        # Testing multiple inputs to cross_sectional_area()
        multi_area, _ = self.equilib.cross_sectional_area(self.rho, self.t)
        multi_area_actual = self.equilib.area.interp(rho_poloidal=self.rho, t=self.t)

        # Compare with 0.01 absolute tolerance and 5% relative tolerance
        # Compare with 0.01 absolute tolerance and 5% relative tolerance
        assert np.isclose(single_area, single_area_actual, atol=1e-2, rtol=5e-2)
        assert np.allclose(multi_area, multi_area_actual, atol=1e-2, rtol=5e-2)

    def test_enclosed_volume(self):
        # Testing single input to cross_sectional_volume()
        single_volume, _ = self.equilib.enclosed_volume(self.rho[2], self.t[1])
        single_volume_actual = self.equilib.volume.interp(
            rho_poloidal=self.rho[2], t=self.t[1]
        )

        # Testing multiple inputs to cross_sectional_volume()
        multi_volume, _ = self.equilib.enclosed_volume(self.rho, self.t)
        multi_volume_actual = self.equilib.volume.interp(
            rho_poloidal=self.rho, t=self.t
        )

        # Compare with 0.01 absolute tolerance and 5% relative tolerance
        assert np.isclose(single_volume, single_volume_actual, atol=1e-2, rtol=5e-2)
        assert np.allclose(multi_volume, multi_volume_actual, atol=1e-2, rtol=5e-2)

    # def test_Btot():
    # """To be re-implemented"""

    def test_R_hfs_1d(self):
        rhfs, t_new = self.equilib.R_hfs(self.rho, self.t)
        assert np.all(t_new == self.t)
        assert np.all(rhfs.coords["rho_poloidal"] == self.rho)

    def test_R_hfs_2d(self):
        rhfs, t_new = self.equilib.R_hfs(self.rho, self.t)
        assert np.all(t_new == self.t)
        assert np.all(rhfs.coords["rho_poloidal"] == self.rho)
