"""Tests coordinate systems based on magnetic field strength."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
import numpy as np
from pytest import approx

from src.converters import MagneticCoordinates
from src.equilibrium import Equilibrium
from ..fake_equilibrium import fake_equilibria
from ..fake_equilibrium import FakeEquilibrium
from ..strategies import arbitrary_coordinates
from ..strategies import monotonic_series
from ..strategies import sane_floats


@composite
def magnetic_coordinate_arguments(draw, min_points=2, max_points=100):
    z = draw(floats(-10.0, 10.0))
    n = draw(integers(min_points, max_points))
    Bmax = draw(floats(1e-3, 1e6))
    Bstart = draw(floats(1e-8, Bmax / 10))
    Bstop = draw(floats(2 * Bstart, Bmax))
    Bvals = draw(monotonic_series(Bstart, Bstop, n))
    tstart = draw(floats(0.0, 500.0))
    tstop = draw(floats(tstart, 1000.0, exclude_min=True))
    t = draw(
        one_of(
            floats(tstart, tstop),
            monotonic_series(tstart, tstop, draw(integers(2, 100))),
        )
    )
    if isinstance(t, np.ndarray):
        t = np.expand_dims(t, (0, 1))
    return z, Bvals, t


@composite
def magnetic_coordinates(draw, domain=None, min_points=2, max_points=100):
    z, B, t = draw(magnetic_coordinate_arguments(min_points, max_points))
    if domain:
        Rmin = domain[0][0]
        tmin = domain[2][0]
    else:
        Rmin = 1.0
        tmin = 1.0

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
    result = MagneticCoordinates(z, B, t)
    result.set_equilibrium(equilib)
    return result


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0)),
)
def test_magnetic_from_Rz_mock(transform_args, coords, expected_result):
    """Test transform of data to magnetic field coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.Btot.return_value = expected_result
    transform = MagneticCoordinates(*transform_args)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, transform_args[0])
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
    assert result[2] is expected_result[2]


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates((1e-5, None, -1e3), (1e5, None, 1e3)),
    floats(),
    sane_floats(),
    floats(1.0, 1e5),
    sane_floats(),
    floats(-1e-3, 1e-3),
)
def test_magnetic_to_Rz_fake(
    transform_args, coords, Rmag, zmag, Btot_a_factor, Btot_b, Btot_alpha
):
    """Test transform of data from magnetic field coordinates."""
    z_los, default_Bvals, default_t = transform_args
    if z_los < zmag:
        z_los, zmag = zmag, z_los
    B, x2, time = coords
    Btot_a = Btot_a_factor * B.max()
    transform = MagneticCoordinates(z_los, default_Bvals, default_t)
    transform.set_equilibrium(
        FakeEquilibrium(Rmag, zmag, Btot_a=Btot_a, Btot_b=Btot_b, Btot_alpha=Btot_alpha)
    )
    R, z, t = transform.convert_to_Rz(*coords)
    assert np.all(
        (1 + Btot_b * R) * (B + zmag - z_los) == approx((1 + Btot_alpha * t) * Btot_a)
    )
    assert z is x2
    assert t is time
