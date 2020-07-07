"""Test conversions on transect coordinate system"""

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from pytest import approx

from src.converters import TransectCoordinates
from ..strategies import monotonic_series
from ..strategies import sane_floats


@composite
def transect_coordinates(draw, min_points=2, max_points=100):
    """Generates :py:class:`src.converters.TransectCoordinates` objects.

    Parameters
    ----------
    min_points: int
        The minimum number of (R,z) pairs in the transect
    max_points: int
        The maximum number of (R,z) pairs in the transect

    Returns
    transform: TransectCoordinates
        The coordinate transform object.
    R_vals: ndarray
        The major radius positions of datapoints along the transect.
    z_vals: ndarray
        The vertical positions of datapoints along the transect.
    """
    num = draw(integers(min_points, max_points))
    ticks = draw(monotonic_series(0.0, 1.0, num))
    R_start = draw(sane_floats())
    R_stop = draw(
        sane_floats().filter(lambda x: x != approx(R_start, rel=1e-3, abs=1e-3))
    )
    z_start = draw(sane_floats())
    z_stop = draw(
        sane_floats().filter(lambda x: x != approx(z_start, rel=1e-3, abs=1e-3))
    )
    R_vals = R_start + (R_stop - R_start) * ticks
    z_vals = z_start + (z_stop - z_start) * ticks
    return TransectCoordinates(R_vals, z_vals), R_vals, z_vals


@given(transect_coordinates(), floats(0.0, 1.0), floats())
def test_transect_zero(coords, position, time):
    """Test that values along transect always have 2nd coordinate 0"""
    transform, Rvals, zvals = coords
    R = Rvals[0] + position * (Rvals[-1] - Rvals[0])
    z = zvals[0] + position * (zvals[-1] - zvals[0])
    i, zprime, t = transform.convert_from_Rz(R, z, time)
    tol = max(1e-22 / position, 1e-12) if position != 0.0 else 1e-20
    assert zprime == approx(0.0, abs=tol)


@given(
    transect_coordinates(), floats(0.0, 1.0, exclude_max=True), sane_floats(), floats()
)
def test_transect_returns_indices(coords, position, z_offset, time):
    """Check (R, z) pairs return expected tick label"""
    transform, Rvals, zvals = coords
    index = int(position * (len(Rvals) - 1))
    i, z, t = transform.convert_from_Rz(Rvals[index], zvals[index] + z_offset, time)
    assert i == approx(index)
    assert z == approx(z_offset)


@given(
    transect_coordinates(), floats(0.0, 1.0, exclude_max=True), sane_floats(), floats()
)
def test_transect_returns_Rz(coords, position, z_offset, time):
    """Check positions along transect fall between appropriate (R, z) pairs"""
    transform, Rvals, zvals = coords
    i = position * (len(Rvals) - 1)
    index = int(i)
    R, z, t = transform.convert_to_Rz(i, z_offset, time)
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
