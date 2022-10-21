import numpy as np
from tests.unit.fake_equilibrium import FakeEquilibrium
import xarray as xr

from indica.asymmetric_quantity import AsymmetricQuantity
from indica.utilities import coord_array


def test_to_R_z():
    rho_poloidal = coord_array(np.array([0, 0.2, 0.4, 0.75, 1]), "rho_poloidal")
    lfs = xr.DataArray(
        np.array([4, 3, 2, 1, 0]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )
    asym = xr.DataArray(
        np.array([0, 1, 2, 1, 0]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )
    equil = FakeEquilibrium()
    quant = AsymmetricQuantity(lfs, asym, equil)

    R = coord_array(np.linspace(1.83, 3.9, 10), "R")
    z = coord_array(np.linspace(-1.75, 2.0, 10), "z")
    quant.to_R_z(R, z)
