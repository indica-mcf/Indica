import numpy as np
from tests.unit.fake_equilibrium import FakeEquilibrium
import xarray as xr

from indica.asymmetric_quantity import AsymmetricQuantity
from indica.converters import FluxSurfaceCoordinates
from indica.utilities import coord_array


def test_to_rho_theta():
    """
    Test to_rho_theta for certain theta values and interpolation is sensible.

    LFS values and asymmetry parameter are chosen such that the density profile is
    a monotonically decreasing function of rho_poloidal. Interpolated values
    are checked to follow this monotonicity.
    """
    # rho points chosen so that result will include these
    # avoids verifying interpolation
    rho_poloidal = coord_array(np.array([0, 0.2, 0.4, 0.8, 1]), "rho_poloidal")
    lfs = xr.DataArray(
        np.array([4, 3, 2, 1, 0]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )
    asym = xr.DataArray(
        np.array([0, 1, 2, 3, 4]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )
    equil = FakeEquilibrium()
    quant = AsymmetricQuantity(lfs, asym, equil)

    rho = coord_array(np.linspace(0, 1, 11), "rho_poloidal")
    theta = coord_array(np.linspace(-np.pi, np.pi, 11), "theta")
    result_rho_theta = quant.to_rho_theta(rho, theta)

    # Check no NaNs or infs
    assert np.all(np.isfinite(result_rho_theta))

    # set up flux surfaces for expected values
    flux_surfaces = FluxSurfaceCoordinates(kind="poloidal")
    flux_surfaces.set_equilibrium(equil)

    # check lfs is as expected
    result_lfs = result_rho_theta.sel(theta=0, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    np.testing.assert_allclose(result_lfs, lfs)

    # check hfs is as expected
    result_hfs = result_rho_theta.sel(theta=np.pi, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    R_hfs, _ = flux_surfaces.convert_to_Rz(rho, np.pi, 0)
    R_lfs, _ = flux_surfaces.convert_to_Rz(rho, 0, 0)
    expected_hfs = lfs * np.exp(asym * (R_hfs**2 - R_lfs**2))
    np.testing.assert_allclose(result_hfs, expected_hfs)

    # check if some arbitrary theta value is as expected
    theta_arb = theta.data[4]
    result_arb = result_rho_theta.sel(theta=theta_arb, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    R_arb, _ = flux_surfaces.convert_to_Rz(rho, theta_arb, 0)
    R_lfs, _ = flux_surfaces.convert_to_Rz(rho, 0, 0)
    expected_arb = lfs * np.exp(asym * (R_arb**2 - R_lfs**2))
    np.testing.assert_allclose(result_arb, expected_arb)
    # interpolation check: values should be descending
    assert np.all(
        result_arb.data == result_arb.sortby(result_arb, ascending=False).data
    )
