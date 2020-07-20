"""Test calculations using equilibrium data."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import fixed_dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import sampled_from
import numpy as np
from pytest import approx

from src.equilibrium import Equilibrium
from .fake_equilibrium import FakeEquilibrium
from .fake_equilibrium import FLUX_TYPES
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
    return draw(sampled_from(FLUX_TYPES))


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, None)),
    flux_types(),
)
def test_lfs_rad(equilib_dat, coords, ftype):
    """Check R_lfs actually on low-flux surfaces and have appropriate flux
    surface values.

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
    """Check R_hfs actually on high-flux surfaces and have appropriate
    flux surface values.

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


@given(
    floats(0.1, 10.0),
    floats(-10.0, 10.0),
    fixed_dictionaries(
        {
            "poloidal_a": floats(0.1, 10.0),
            "poloidal_b": floats(0.1, 10.0),
            "poloidal_alpha": floats(-0.001, 0.01),
            "toroidal_a": floats(0.1, 10.0),
            "toroidal_b": floats(0.1, 10.0),
            "toroidal_alpha": floats(-0.001, 0.01),
        }
    ),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, 200.0)),
)
def test_enclosed_volume(Rmag, zmag, parameters, coords):
    """Check volumes enclosed within flux surfaces are what is expected
    (when using fake flux surfaces which are elliptical)."""
    parameters.update({"poloidal_n": 1, "toroidal_n": 1})
    equilib = FakeEquilibrium(Rmag, zmag, **parameters)
    rho, _, time = coords
    vol_pol = (
        np.pi
        * parameters["poloidal_a"]
        * parameters["poloidal_b"]
        * rho ** 2
        * (1 + parameters["poloidal_alpha"] * time) ** 2
    )
    vol_tor = (
        np.pi
        * parameters["toroidal_a"]
        * parameters["toroidal_b"]
        * rho ** 2
        * (1 + parameters["toroidal_alpha"] * time) ** 2
    )
    volume, t = equilib.enclosed_volume(rho, time, "toroidal")
    assert np.all(volume == approx(vol_tor))
    assert t is time
    volume, t = equilib.enclosed_volume(rho, time, "poloidal")
    assert np.all(volume == approx(vol_pol))
    assert t is time


# Test use of offset picker

# Check provenance is correct
