"""Tests coordinate systems based on magnetic field strength."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
import numpy as np
from pytest import approx

from indica.converters import MagneticCoordinates
from indica.equilibrium import Equilibrium
from indica.utilities import coord_array
from ..fake_equilibrium import fake_equilibria
from ..fake_equilibrium import FakeEquilibrium
from ..strategies import arbitrary_coordinates
from ..strategies import machine_dimensions
from ..strategies import monotonic_series
from ..strategies import sane_floats


@composite
def magnetic_coordinate_arguments(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    R_dims = (
        draw(floats(min(0.0, domain[0][0]), domain[0][0])),
        draw(floats(domain[0][1], max(4.0, domain[0][1] + 1.0))),
    )
    z_dims = draw(machine_dimensions())[1]
    return draw(floats(*domain[1])), (R_dims, z_dims)


@composite
def magnetic_coordinates(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    z, machine_dims = draw(magnetic_coordinate_arguments(domain))
    if domain:
        Rmin = domain[0][0]
    else:
        Rmin = machine_dims[0][0]
    if Rmin < 0:
        Bcoeff = draw(floats(min(0.1, -0.01 / Rmin), min(10.0, -0.99 / Rmin)))
    else:
        Bcoeff = draw(floats(0.1, 10.0))
    B_alpha = draw(floats(0.0, 1.0))
    equilib = draw(
        fake_equilibria(
            draw(floats(*domain[0])),
            draw(floats(*domain[1])),
            coord_array(
                draw(monotonic_series(*domain[2], draw(integers(2, 20))),), "t"
            ),
            Btot_alpha=B_alpha,
            Btot_b=Bcoeff,
        )
    )
    result = MagneticCoordinates(z, machine_dims)
    result.set_equilibrium(equilib)
    result.default_x1 = result.convert_from_Rz()[0][0]
    return result


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), max_side=3),
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


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates((0.00001, None, 0.0), (1.0, None, 1e3), max_side=3),
    sane_floats(),
    sane_floats(),
    floats(0.1, 1e3),
    floats(1e-5, 10.0),
    floats(-9e-4, 9e-4),
)
@settings(deadline=500)
def test_magnetic_to_Rz_fake_on_los(
    transform_args, coords, Rmag, zmag, Btot_a, Btot_b, Btot_alpha
):
    """Test transform of data from magnetic field coordinates along the
    line of sight.

    """
    z_los, machine_dims = transform_args
    R_expected_factor, x2, time = coords
    R_expected = (
        R_expected_factor * (machine_dims[0][1] - machine_dims[0][0])
        + machine_dims[0][0]
    )
    B = (1 + Btot_alpha * time) * Btot_a / (1 + Btot_b * R_expected) + z_los - zmag
    transform = MagneticCoordinates(z_los, machine_dims)
    transform.set_equilibrium(
        FakeEquilibrium(Rmag, zmag, Btot_a=Btot_a, Btot_b=Btot_b, Btot_alpha=Btot_alpha)
    )
    #    print("Sent to _convert_to_Rz", B, None, time)
    R, z = transform.convert_to_Rz(B, 0.0, time)
    assert np.all(R == approx(R_expected * np.ones_like(time), abs=1e-6, rel=1e-6))
    assert z == approx(z_los)


@given(
    magnetic_coordinate_arguments(),
    arbitrary_coordinates((0.00001, -2.0, 0.0), (1.0, 2.0, 1e3), max_side=3),
    sane_floats(),
    sane_floats(),
    floats(0.1, 1e3),
    floats(1e-5, 10.0),
    floats(-9e-4, 9e-4),
)
@settings(deadline=None)
def test_magnetic_to_Rz_fake(
    transform_args, coords, Rmag, zmag, Btot_a, Btot_b, Btot_alpha
):
    """Test transform of data from magnetic field coordinates for arbitrary
    vertical position."""
    z_los, machine_dims = transform_args
    R_expected_factor, x2, time = coords
    R_expected = (
        R_expected_factor * (machine_dims[0][1] - machine_dims[0][0])
        + machine_dims[0][0]
    )
    B = (1 + Btot_alpha * time) * Btot_a / (1 + Btot_b * R_expected) + z_los + x2 - zmag
    transform = MagneticCoordinates(z_los, machine_dims)
    transform.set_equilibrium(
        FakeEquilibrium(Rmag, zmag, Btot_a=Btot_a, Btot_b=Btot_b, Btot_alpha=Btot_alpha)
    )
    R, z = transform.convert_to_Rz(B, x2, time)
    assert np.all(
        R
        == approx(
            R_expected * np.ones_like(x2) * np.ones_like(time), abs=1e-6, rel=1e-6
        )
    )
    assert np.all(z == approx(z_los + x2))
