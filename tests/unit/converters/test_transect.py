"""Test conversions on transect coordinate system"""

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import text
import numpy as np
from pytest import approx
from xarray import DataArray

from indica.converters import TransectCoordinates
from indica.utilities import coord_array
from ..strategies import monotonic_series
from ..strategies import sane_floats


@composite
def transect_coordinates_parameters(
    draw,
    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    min_points=2,
    max_points=20,
    coord_name=None,
):
    """Generates the parameters needed to instantiate
    :py:class:`indica.converters.TransectCoordinates` objects.

    Parameters
    ----------
    min_points: int
        The minimum number of (R,z) pairs in the transect
    max_points: int
        The maximum number of (R,z) pairs in the transect
    coord_name: Optional[str]
        Name of the coordinate for the R_vals and z_vals

    Returns
    -------
    R_vals: ndarray
        The major radius positions of datapoints along the transect.
    z_vals: ndarray
        The vertical positions of datapoints along the transect.

    """
    num = draw(integers(min_points, max_points))
    ticks = draw(monotonic_series(0.0, 1.0, num))
    R_start = draw(floats(*domain[0]))
    R_stop = draw(
        floats(*domain[0]).filter(lambda x: x != approx(R_start, rel=1e-3, abs=1e-3))
    )
    z_start = draw(floats(*domain[1]))
    z_stop = draw(
        floats(*domain[1]).filter(lambda x: x != approx(z_start, rel=1e-3, abs=1e-3))
    )
    R_vals = R_start + (R_stop - R_start) * ticks
    z_vals = z_start + (z_stop - z_start) * ticks
    if coord_name is None:
        cname = draw(text(min_size=1))
    else:
        cname = coord_name
    return DataArray(R_vals, dims=cname), DataArray(z_vals, dims=cname)


@composite
def transect_coordinates_and_axes(
    draw,
    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    min_points=2,
    max_points=100,
    coord_name=None,
):
    """Generates :py:class:`indica.converters.TransectCoordinates` objects and
    (x1, x2, t) axes.

    Parameters
    ----------
    min_points: int
        The minimum number of (R,z) pairs in the transect
    max_points: int
        The maximum number of (R,z) pairs in the transect
    coord_name: Optional[str]
        Name of the x1 coordinate for this system

    """
    R_vals, z_vals = draw(
        transect_coordinates_parameters(domain, min_points, max_points, coord_name)
    )
    transform = TransectCoordinates(R_vals, z_vals)
    x1 = coord_array(DataArray(np.arange(len(R_vals))), transform.x1_name)
    x2 = DataArray(0)
    t = DataArray(0)
    return transform, x1, x2, t


@composite
def transect_coordinates(
    draw,
    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    min_points=2,
    max_points=100,
    coord_name=None,
):
    """Generates :py:class:`indica.converters.TransectCoordinates` objects.

    Parameters
    ----------
    min_points: int
        The minimum number of (R,z) pairs in the transect
    max_points: int
        The maximum number of (R,z) pairs in the transect
    coord_name: Optional[str]
        Name of the x1 coordinate for this system

    """
    return draw(
        transect_coordinates_and_axes(domain, min_points, max_points, coord_name)
    )[0]


@given(transect_coordinates_parameters(), floats(0.0, 1.0), floats())
def test_transect_zero(params, position, time):
    """Test that values along transect always have 2nd coordinate 0"""
    Rvals, zvals = params
    transform = TransectCoordinates(Rvals, zvals)
    R = Rvals[0] + position * (Rvals[-1] - Rvals[0])
    z = zvals[0] + position * (zvals[-1] - zvals[0])
    i, zprime = transform.convert_from_Rz(R, z, time)
    tol = max(1e-20 / position, 1e-10) if position != 0.0 else 1e-10
    assert zprime == approx(0.0, abs=tol)


@given(
    transect_coordinates_parameters(),
    floats(0.0, 1.0, exclude_max=True),
    sane_floats(),
    floats(),
)
def test_transect_returns_indices(params, position, z_offset, time):
    """Check (R, z) pairs return expected tick label"""
    Rvals, zvals = params
    transform = TransectCoordinates(Rvals, zvals)
    index = int(position * (len(Rvals) - 1))
    i, z = transform.convert_from_Rz(Rvals[index], zvals[index] + z_offset, time)
    assert i == approx(index)
    assert z == approx(z_offset)


@given(
    transect_coordinates_parameters(),
    floats(0.0, 1.0, exclude_max=True),
    sane_floats(),
    floats(),
)
def test_transect_returns_Rz(params, position, z_offset, time):
    """Check positions along transect fall between appropriate (R, z) pairs"""
    Rvals, zvals = params
    transform = TransectCoordinates(Rvals, zvals)
    i = position * (len(Rvals) - 1)
    index = int(i)
    R, z = transform.convert_to_Rz(i, z_offset, time)
    if Rvals[-1] > Rvals[0]:
        assert R <= Rvals[index + 1]
        assert Rvals[index] <= R
    else:
        assert R >= Rvals[index + 1]
        assert Rvals[index] >= R
    if zvals[-1] > zvals[0]:
        assert z <= zvals[index + 1] + z_offset
        assert zvals[index] + z_offset <= z
    else:
        assert z >= zvals[index + 1] + z_offset
        assert zvals[index] + z_offset >= z
