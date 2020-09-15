"""Tests coordinate systems based on magnetic field strength."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
import numpy as np
from pytest import approx

from indica.converters import MagneticCoordinates
from indica.equilibrium import Equilibrium
from ..fake_equilibrium import fake_equilibria
from ..fake_equilibrium import FakeEquilibrium
from ..strategies import arbitrary_coordinates
from ..strategies import monotonic_series
from ..strategies import sane_floats


@composite
def magnetic_coordinate_arguments(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_points=2, max_points=20
):
    z = draw(floats(*domain[1]))
    n = draw(integers(min_points, max_points))
    Bmax = draw(floats(1e-3, 1e6))
    Bstart = draw(floats(1e-8, Bmax / 10))
    Bstop = draw(floats(2 * Bstart, Bmax))
    Bvals = draw(monotonic_series(Bstart, Bstop, n))
    t = draw(monotonic_series(*domain[2], draw(integers(2, 20))),)
    Rvals = draw(monotonic_series(*domain[0], draw(integers(10, 50))),)
    return z, Bvals, Rvals, t


@composite
def magnetic_coordinates(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_points=2, max_points=20
):
    z, B, R, t = draw(magnetic_coordinate_arguments(domain, min_points, max_points))
    if domain:
        Rmin = domain[0][0]
        tmin = domain[2][0]
    else:
        Rmin = R.min()
        tmin = t.min()

    if Rmin < 0:
        Bcoeff = draw(floats(min(0.1, -0.01 / Rmin), min(10.0, -0.99 / Rmin)))
    else:
        Bcoeff = draw(floats(0.1, 10.0))
    if tmin < 0:
        B_alpha = draw(floats(0.0, min(1.0, -0.99 / tmin)))
    else:
        B_alpha = draw(floats(0.0, 1.0))
    equilib = draw(
        fake_equilibria(
            draw(sane_floats()), draw(sane_floats()), Btot_alpha=B_alpha, Btot_b=Bcoeff
        )
    )
    result = MagneticCoordinates(z, B, R, t)
    result.set_equilibrium(equilib)
    result.default_x1 = result.convert_from_Rz()[0][0]
    return result


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0)),
)
def test_magnetic_from_Rz_mock(transform_args, coords, expected_result):
    """Test transform of data to magnetic field coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.Btot.return_value = expected_result[0], expected_result[2]
    transform = MagneticCoordinates(*transform_args)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.Btot.assert_called_with(*coords)
    assert result[0] is expected_result[0]
    assert result[1] == approx(coords[1] - transform_args[0])
    assert result[2] is expected_result[2]


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates((0.0, None, -1e3), (1.0, None, 1e3)),
    sane_floats(),
    sane_floats(),
    floats(0.1, 1e3),
    floats(1e-5, 10.0),
    floats(-9e-4, 9e-4),
)
def test_magnetic_to_Rz_fake_on_los(
    transform_args, coords, Rmag, zmag, Btot_a, Btot_b, Btot_alpha
):
    """Test transform of data from magnetic field coordinates along the
    line of sight.

    """
    z_los, default_Bvals, default_R, default_t = transform_args
    R_expected_factor, x2, time = coords
    R_expected = R_expected_factor * (default_R[-1] - default_R[0]) + default_R[0]
    B = (1 + Btot_alpha * time) * Btot_a / (1 + Btot_b * R_expected) + z_los - zmag
    transform = MagneticCoordinates(z_los, default_Bvals, default_R, default_t)
    transform.set_equilibrium(
        FakeEquilibrium(Rmag, zmag, Btot_a=Btot_a, Btot_b=Btot_b, Btot_alpha=Btot_alpha)
    )
    #    print("Sent to _convert_to_Rz", B, None, time)
    R, z, t = transform.convert_to_Rz(B, None, time)
    assert np.all(R == approx(R_expected * np.ones_like(time), abs=1e-6, rel=1e-6))
    assert z == approx(z_los)
    assert t is time


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates((0.0, -2.0, -1e3), (1.0, 2.0, 1e3)),
    sane_floats(),
    sane_floats(),
    floats(0.1, 1e3),
    floats(1e-5, 10.0),
    floats(-9e-4, 9e-4),
)
def test_magnetic_to_Rz_fake(
    transform_args, coords, Rmag, zmag, Btot_a, Btot_b, Btot_alpha
):
    """Test transform of data from magnetic field coordinates for arbitrary
    vertical position."""
    z_los, default_Bvals, default_R, default_t = transform_args
    R_expected_factor, x2, time = coords
    R_expected = R_expected_factor * (default_R[-1] - default_R[0]) + default_R[0]
    B = (1 + Btot_alpha * time) * Btot_a / (1 + Btot_b * R_expected) + z_los + x2 - zmag
    transform = MagneticCoordinates(z_los, default_Bvals, default_R, default_t)
    transform.set_equilibrium(
        FakeEquilibrium(Rmag, zmag, Btot_a=Btot_a, Btot_b=Btot_b, Btot_alpha=Btot_alpha)
    )
    R, z, t = transform.convert_to_Rz(B, x2, time)
    assert np.all(
        R
        == approx(
            R_expected * np.ones_like(x2) * np.ones_like(time), abs=1e-6, rel=1e-6
        )
    )
    assert np.all(z == approx(z_los + x2))
    assert t is time
