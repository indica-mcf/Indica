"""Test enclosed volume coordinate systems."""

from unittest.mock import Mock

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
import numpy as np
from xarray.testing import assert_allclose

from indica.converters import EnclosedVolumeCoordinates
from .test_flux_surfaces import flux_coordinates
from ..strategies import arbitrary_coordinates


@composite
def enclosed_volume_coordinates(draw, min_side=1, min_dims=0):
    """Generates :py:class:`indica.converters.EnclosedVolumeCoordinates` objects.

    Parameters
    ----------
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_dims : integer
        The minimum number of dimensions for the default coordinate arrays.
        (Not available for all coordinate systems.)

    """
    return EnclosedVolumeCoordinates(draw(flux_coordinates(min_side, min_dims)))


@settings(report_multiple_bugs=False)
@given(
    flux_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 2 * np.pi, 1e3), xarray=True),
)
def test_convert_to_Rz(flux_transform, coords):
    """Test conversion of enclosde-volume coordinates to R,z coordinates."""
    rho, theta, t = coords
    vol, t = flux_transform.equilibrium.enclosed_volume(
        rho, t, flux_transform.flux_kind
    )
    R_expected, z_expected, t_expected = flux_transform.convert_to_Rz(*coords)
    transform = EnclosedVolumeCoordinates(flux_transform)
    R, z, t = transform.convert_to_Rz(vol, coords[1], t)
    assert_allclose(R, R_expected)
    assert_allclose(z, z_expected)
    assert t is t_expected


@given(
    flux_coordinates(),
    arbitrary_coordinates(xarray=True),
    arbitrary_coordinates(xarray=True),
    arbitrary_coordinates(xarray=True),
)
def test_convert_from_Rz(flux_transform, coords, flux_coords, expected_coords):
    """Test conversion of R,z coordinates to enclosed-volume coordinates."""
    equilib = Mock()
    flux_transform.set_equilibrium(equilib, force=True)
    equilib.flux_coords.return_value = flux_coords
    equilib.enclosed_volume.return_value = (expected_coords[0], expected_coords[2])
    transform = EnclosedVolumeCoordinates(flux_transform)
    vol, theta, t = transform.convert_from_Rz(*coords)
    equilib.flux_coords.assert_called_with(*coords, flux_transform.flux_kind)
    equilib.enclosed_volume.assert_called_with(
        flux_coords[0], flux_coords[2], flux_transform.flux_kind
    )
    assert vol is expected_coords[0]
    assert theta is flux_coords[1]
    assert t is expected_coords[2]
