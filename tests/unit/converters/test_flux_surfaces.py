"""Tests transforms to/from coordinate systems based on flux surfaces."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
import numpy as np

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.utilities import coord_array
from ..fake_equilibrium import fake_equilibria
from ..fake_equilibrium import flux_types
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates


@composite
def flux_coordinates(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    """Generates :py:class:`indica.converters.FluxSurfaceCoordinates` objects."""
    result = FluxSurfaceCoordinates(draw(flux_types()))
    Rmag = draw(floats(*domain[0]))
    if abs(Rmag) < 1e-10:
        sign = 1 if Rmag == 0.0 else np.sign(Rmag)
        Rmag += sign * 0.1 * (domain[0][1] - domain[0][0])
    zmag = draw(floats(*domain[1]))
    result.set_equilibrium(draw(fake_equilibria(Rmag, zmag)))
    return result


@composite
def flux_coordinates_and_axes(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=2, max_side=12
):
    """Generates :py:class:`indica.converters.FluxSurfaceCoordinates` objects,
    along with axes.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    max_side : integer
        The maximum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)

    """
    transform = draw(flux_coordinates(domain))
    x1_vals, x2_vals, t_vals = map(
        np.ravel,
        draw(
            basis_coordinates(
                (0.0, 0.0, 0.0), (1.0, 2 * np.pi, 120.0), min_side, max_side
            )
        ),
    )
    return (
        transform,
        coord_array(x1_vals, transform.x1_name),
        coord_array(x2_vals, transform.x2_name),
        coord_array(t_vals, "t"),
    )


@given(
    flux_types(),
    arbitrary_coordinates(xarray=True),
    arbitrary_coordinates((0.0, 0.0, None), (1.0, 2 * np.pi, None), xarray=True),
)
def test_flux_from_Rz_mock(kind, coords, expected_result):
    """Test transform of data to flux coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.flux_coords.return_value = expected_result
    transform = FluxSurfaceCoordinates(kind)
    transform.set_equilibrium(equilib)
    result = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, kind)
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]


@given(
    flux_types(),
    arbitrary_coordinates((0.0, 0.0, None), (1.0, 2 * np.pi, None), xarray=True),
    arbitrary_coordinates(xarray=True),
)
def test_flux_to_Rz_mock(kind, coords, expected_result):
    """Test transform of data from flux coordinates."""
    equilib = Mock(spec=Equilibrium)
    equilib.spatial_coords.return_value = expected_result
    transform = FluxSurfaceCoordinates(kind)
    transform.set_equilibrium(equilib)
    result = transform.convert_to_Rz(*coords)
    equilib.spatial_coords.assert_called_with(*coords, kind)
    assert result[0] is expected_result[0]
    assert result[1] is expected_result[1]
