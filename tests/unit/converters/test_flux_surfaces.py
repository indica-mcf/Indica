"""Tests transforms to/from coordinate systems based on flux surfaces."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
import numpy as np

from src.converters import FluxSurfaceCoordinates
from src.equilibrium import Equilibrium
from ..strategies import arbitrary_coordinates
from ..test_equilibrium import flux_types


@composite
def flux_coordinate_arguments(draw):
    rho, theta, t = draw(arbitrary_coordinates())
    bshape = t.shape if isinstance(t, np.ndarray) else ()
    R, z, t = draw(arbitrary_coordinates(base_shape=bshape))
    return draw(flux_types()), rho, theta, R, z, t


@composite
def flux_coordinates(draw):
    result = FluxSurfaceCoordinates(*draw(flux_coordinate_arguments()))
    # TODO: create some sort of fake equilibrium which makes useable transforms
    result.set_equilibrium(Mock())
    return result


@given(
    flux_coordinate_arguments(),
    arbitrary_coordinates(),
    arbitrary_coordinates((0.0, 0.0, None), (1.0, 2 * np.pi, None)),
)
def test_flux_from_Rz(transform_args, coords, expected_result):
    """Test transform of data to flux coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.flux_coords.return_value = expected_result
    transform = FluxSurfaceCoordinates(*transform_args)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, transform_args[0])
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
    assert result[2] is expected_result[2]


@given(
    flux_coordinate_arguments(),
    arbitrary_coordinates((0.0, 0.0, None), (1.0, 2 * np.pi, None)),
    arbitrary_coordinates(),
)
def test_flux_to_Rz(transform_args, coords, expected_result):
    """Test transform of data from flux coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.flux_coords.return_value = expected_result
    transform = FluxSurfaceCoordinates(*transform_args)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, transform_args[0])
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
    assert result[2] is expected_result[2]
