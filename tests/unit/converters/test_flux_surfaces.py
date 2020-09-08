"""Tests transforms to/from coordinate systems based on flux surfaces."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from ..fake_equilibrium import fake_equilibria
from ..fake_equilibrium import flux_types
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates


@composite
def flux_coordinate_arguments(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=2
):
    """Generate the parameters needed to instantiate
    :py:class:`indica.converters.FluxSurfaceCoordinates`.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)

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
        basis_coordinates(
            (0.0, 0.0, domain[2][0]), (1.0, 2 * np.pi, domain[2][1]), min_side=min_side,
        )
    )
    R, z, t = draw(
        basis_coordinates(
            tuple(dim[0] for dim in domain), tuple(dim[1] for dim in domain), min_side
        )
    )
    return draw(flux_types()), rho, theta, R, z, t


@composite
def flux_coordinates(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=2):
    """Generates :py:class:`indica.converters.FluxSurfaceCoordinates` objects.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)

    """
    result = FluxSurfaceCoordinates(*draw(flux_coordinate_arguments(domain, min_side)))
    Rmag = draw(floats(*domain[0]))
    if abs(Rmag) < 1e-10:
        sign = 1 if Rmag == 0.0 else np.sign(Rmag)
        Rmag += sign * 0.1 * (domain[0][1] - domain[0][0])
    zmag = draw(floats(*domain[1]))
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
    result = transform.convert_to_Rz(*coords)
    equilib.spatial_coords.assert_called_with(*coords, transform_args[0])
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
    assert result[2] is expected_result[2]
