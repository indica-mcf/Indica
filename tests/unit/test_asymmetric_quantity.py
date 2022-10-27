import numpy as np
import pytest
from tests.unit.fake_equilibrium import FakeEquilibrium
import xarray as xr

from indica.asymmetric_quantity import AsymmetricQuantity
from indica.converters import FluxSurfaceCoordinates
from indica.utilities import coord_array


@pytest.fixture
def rho_poloidal():
    return coord_array(np.array([0, 0.2, 0.4, 0.8, 1]), "rho_poloidal")


@pytest.fixture
def lfs_vals(rho_poloidal):
    return xr.DataArray(
        np.array([4, 3, 2, 1, 0]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )


@pytest.fixture
def asym_param(rho_poloidal):
    return xr.DataArray(
        np.array([0, 1, 2, 3, 4]),
        dims="rho_poloidal",
        coords=dict(rho_poloidal=rho_poloidal),
    )


@pytest.fixture
def equil():
    return FakeEquilibrium()


@pytest.fixture
def asym_quant(lfs_vals, asym_param, equil):
    return AsymmetricQuantity(lfs_vals, asym_param, equil)


def test_to_rho_theta(rho_poloidal, lfs_vals, asym_param, equil, asym_quant):
    """
    Test to_rho_theta for certain theta values and interpolation is sensible.

    LFS values and asymmetry parameter are chosen such that the density profile is
    a monotonically decreasing function of rho_poloidal. Interpolated values
    are checked to follow this monotonicity.
    """

    # rho chosen to contain values from rho_poloidal
    rho = coord_array(np.linspace(0, 1, 11), "rho_poloidal")
    theta = coord_array(np.linspace(-np.pi, np.pi, 11), "theta")
    result_rho_theta = asym_quant.to_rho_theta(rho, theta)

    # Check no NaNs or infs
    assert np.all(np.isfinite(result_rho_theta))

    # set up flux surfaces for expected values
    flux_surfaces = FluxSurfaceCoordinates(kind="poloidal")
    flux_surfaces.set_equilibrium(equil)

    # check lfs is as expected
    result_lfs = result_rho_theta.sel(theta=0, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    np.testing.assert_allclose(result_lfs, lfs_vals)

    # check hfs is as expected
    result_hfs = result_rho_theta.sel(theta=np.pi, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    R_hfs, _ = flux_surfaces.convert_to_Rz(rho, np.pi, 0)
    R_lfs, _ = flux_surfaces.convert_to_Rz(rho, 0, 0)
    expected_hfs = lfs_vals * np.exp(asym_param * (R_hfs**2 - R_lfs**2))
    np.testing.assert_allclose(result_hfs, expected_hfs)

    # check if some arbitrary theta value is as expected
    theta_arb = theta.data[4]
    result_arb = result_rho_theta.sel(theta=theta_arb, method="nearest").interp(
        {"rho_poloidal": rho_poloidal}
    )
    R_arb, _ = flux_surfaces.convert_to_Rz(rho, theta_arb, 0)
    R_lfs, _ = flux_surfaces.convert_to_Rz(rho, 0, 0)
    expected_arb = lfs_vals * np.exp(asym_param * (R_arb**2 - R_lfs**2))
    np.testing.assert_allclose(result_arb, expected_arb)
    # interpolation check: values should be descending
    assert np.all(
        result_arb.data == result_arb.sortby(result_arb, ascending=False).data
    )


# need second asym_quant for arithmetic:
asym_quant2 = asym_quant


def test_add(asym_quant, asym_quant2):
    """
    Test adding two asymmetric quantities is identical to adding two 2D profiles
    """
    rho = coord_array(np.linspace(0, 1, 11), "rho_poloidal")
    theta = coord_array(np.linspace(-np.pi, np.pi, 11), "theta")

    result = (asym_quant + asym_quant2).to_rho_theta(rho, theta)
    expected = asym_quant.to_rho_theta(rho, theta)
    +asym_quant2.to_rho_theta(rho, theta)

    np.assert_allclose(result, expected)
