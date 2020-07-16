"""Tests transforms to/from coordinate systems based on flux surfaces."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
import numpy as np

from src.converters import FluxSurfaceCoordinates
from src.equilibrium import Equilibrium
from ..fake_equilibrium import fake_equilibria
from ..strategies import arbitrary_coordinates
from ..test_equilibrium import flux_types


@composite
def flux_coordinate_arguments(draw, min_side=1, min_dims=0):
    """Generate the parameters needed to instantiate
    :py:class:`src.converters.FluxSurfaceCoordinates`.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_dims : integer
        The minimum number of dimensions for the default coordinate arrays.
        (Not available for all coordinate systems.)

    Returns
    -------
    flux_kind : str
        What sort of flux is being used (e.g., ``"poloidal"``, ``"toroidal"``)
    rho : ndarray
        Default coordinates for flux surface value
    theta : ndarray
        Default coordinates for angle
    R : ndarray
        Default coordinates for major radius
    z : ndarray
        Default coordinates for vertical position
    t : ndarray
        Default coordinates for time

    """
    rho, theta, t = draw(
        arbitrary_coordinates(
            (0.0, 0.0, None),
            (1.0, 2 * np.pi, None),
            min_side=min_side,
            min_dims=min_dims,
        )
    )
    bshape = t.shape if isinstance(t, np.ndarray) else ()
    R, z, t = draw(
        arbitrary_coordinates(base_shape=bshape, min_side=min_side, min_dims=min_dims)
    )
    return draw(flux_types()), rho, theta, R, z, t


@composite
def flux_coordinates(draw, min_side=1, min_dims=0):
    """Generates :py:class:`src.converters.FluxSurfaceCoordinates` objects.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_dims : integer
        The minimum number of dimensions for the default coordinate arrays.
        (Not available for all coordinate systems.)

    """
    result = FluxSurfaceCoordinates(
        *draw(flux_coordinate_arguments(min_side, min_dims))
    )
    Rmag = draw(floats(0.1, 10.0))
    zmag = draw(floats(-10.0, 10.0))
    result.set_equilibrium(draw(fake_equilibria(Rmag, zmag)))
    return result


@given(
    flux_coordinate_arguments(),
    arbitrary_coordinates(),
    arbitrary_coordinates((0.0, 0.0, None), (1.0, 2 * np.pi, None)),
)
def test_flux_from_Rz_mock(transform_args, coords, expected_result):
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
def test_flux_to_Rz_mock(transform_args, coords, expected_result):
    """Test transform of data from flux coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.spatial_coords.return_value = expected_result
    transform = FluxSurfaceCoordinates(*transform_args)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, transform_args[0])
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
    assert result[2] is expected_result[2]
