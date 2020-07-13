"""Test calculations using equilibrium data."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import sampled_from
import numpy as np
from pytest import approx

from src.equilibrium import Equilibrium
from .strategies import arbitrary_coordinates


@composite
def equilibrium_data(draw):
    """Generate the data necessary to create an equilibrium object.

    """
    # TODO: implement this
    return {}


@composite
def equilibria(draw):
    """Generate :py:class:`src.equilibrium.Equilibrium` objects for
    testing purposes.

    """
    # TODO: Create version of this routine that sets the offset
    return Equilibrium(draw(equilibrium_data()), sess=MagicMock())


@composite
def flux_types(draw):
    return sampled_from("toroidal", "poloidal")


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, None)),
    flux_types(),
)
def test_lfs_rad(equilib_dat, coords, ftype):
    """Check R_lfs and R_hfs actually on low-flux surfaces and have
    appropriate flux surface values.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    rho, theta, time = coords
    R_lfs, t = equilib.R_lfs(rho, time, ftype)
    z_mag = equilib_dat["zmag"].interp(t=time)
    R_mag = equilib_dat["Rmag"].interp(t=time)  # TODO: add R-shift
    assert np.all(rho == approx(equilib.flux_coords(R_lfs, z_mag, time)[0]))
    assert np.all(R_lfs <= R_mag)


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, None)),
    flux_types(),
)
def test_hfs_rad(equilib_dat, coords, ftype):
    """Check R_lfs and R_hfs actually on high-flux surfaces and have
    appropriate flux surface values.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    rho, theta, time = coords
    R_hfs, t = equilib.R_hfs(rho, time, ftype)
    z_mag = equilib_dat["zmag"].interp(t=time)
    R_mag = equilib_dat["Rmag"].interp(t=time)  # TODO: add R-shift
    assert np.all(rho == approx(equilib.flux_coords(R_hfs, z_mag, time)[0]))
    assert np.all(R_hfs >= R_mag)


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, None)),
    flux_types(),
)
def test_flux_spatial_conversion(equilib, coords, ftype):
    """Check flux and spatial coordinate conversions can invert each other.

    """
    rho, theta, time = coords
    R, z, t = equilib.spatial_coords(rho, theta, time, ftype)
    rho2, theta2, t = equilib.flux_coords(R, z, t, ftype)
    assert np.all(rho2 == approx(rho))
    assert np.all(theta2 == approx(theta))
    assert t is time


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, None)),
    flux_types(),
    flux_types(),
)
def test_flux_flux_conversion(equilib, coords, ftype1, ftype2):
    """Check conversions back and forth between different types of flux
    coordinates can invert properly.

    """
    rho, theta, time = coords
    rho_tmp, theta_tmp, t = equilib.convert_flux_coords(
        rho, theta, time, ftype1, ftype2
    )
    rho2, theta2, t = equilib.convert_flux_coords(rho_tmp, theta_tmp, t, ftype2, ftype1)
    assert np.all(rho2 == approx(rho))
    assert np.all(theta2 == approx(theta))
    assert t is time


# Check Btot and minor rad using some sort of dummy data (e.g., some factor
# times distance from magnetic axis?)

# Test use of offset picker

# Check provenance is correct
