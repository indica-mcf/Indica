"""Tests for line-of-sight coordinate transforms."""

from itertools import product
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import one_of
import numpy as np
from pytest import approx
from pytest import mark

from indica.converters import LinesOfSightTransform
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates
from ..strategies import machine_dimensions
from ..strategies import monotonic_series
from ..strategies import smooth_functions


def inside_machine(coords, dimensions, boundary=True):
    """Tests whether the provided coordinates fall inside the tokamak.

    Parameters
    ----------
    coords : Tuple[float, float]
        The (R, z) coordinates of the location to be tests
    dimensions: Tuple[Tuple[float, float], Tuple[float, float]]
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``.
    boundary: bool
        Whether to count the boundary of the tokamak as "inside".

    """
    if boundary:
        return np.logical_and(
            dimensions[0][0] <= coords[0],
            np.logical_and(
                coords[0] <= dimensions[0][1],
                np.logical_and(
                    dimensions[1][0] <= coords[1], coords[1] <= dimensions[1][1]
                ),
            ),
        )
    else:
        return np.logical_and(
            dimensions[0][0] + 1e-12 < coords[0],
            np.logical_and(
                coords[0] < dimensions[0][1] - 1e-12,
                np.logical_and(
                    dimensions[1][0] + 1e-12 < coords[1],
                    coords[1] < dimensions[1][1] - 1e-12,
                ),
            ),
        )


@composite
def parallel_los_coordinates(
    draw, machine_dims=None, min_los=2, max_los=10, min_num=2, max_num=20
):
    """Generates :py:class:`indica.converters.LinesOfSightTransform`  objects
    where lines of sight are parallel, either horizontally or vertically.

    Parameters
    ----------
    machine_dims: Tuple[Tuple[float, float], Tuple[float, float]], optional
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. If absent will draw values.
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    min_num: int
        The minimum number of intervals in which to divide the lines of sight
    max_num: int
        The maximum number of intervals in which to divide the lines of sight

    Returns
    -------
    transform: TransectCoordinates
        The coordinate transform object.
    vertical: bool
        Whether the lines of sight are vertical (True) or horizontal (False)
    R_vals: ndarray
        The major radius positions of datapoints on the line of sight grid.
    z_vals: ndarray
        The vertical positions of datapoints on the line of sight grid.

    """
    if not machine_dims:
        machine_dims = draw(machine_dimensions())
    if machine_dims[0][0] < 0:
        offset = abs(machine_dims[0][0])
        machine_dims = (
            (machine_dims[0][0] + offset, machine_dims[0][1] + offset),
            machine_dims[1],
        )
    if draw(booleans()):
        R_stop, R_start = machine_dims[0]
    else:
        R_start, R_stop = machine_dims[0]
    if draw(booleans()):
        z_stop, z_start = machine_dims[1]
    else:
        z_start, z_stop = machine_dims[1]
    vertical = draw(booleans())
    num_los = draw(integers(min_los, max_los))
    num_intervals = draw(integers(min_num, max_num))
    if vertical:
        R_vals = R_start_vals = R_stop_vals = draw(
            monotonic_series(R_start, R_stop, num_los)
        )
        z_start_vals = np.ones(num_los) * z_start
        z_stop_vals = np.ones(num_los) * z_stop
        z_vals = np.linspace(z_start, z_stop, num_intervals + 1)
    else:
        z_vals = z_start_vals = z_stop_vals = draw(
            monotonic_series(z_start, z_stop, num_los)
        )
        R_start_vals = np.ones(num_los) * R_start
        R_stop_vals = np.ones(num_los) * R_stop
        R_vals = np.linspace(R_start, R_stop, num_intervals + 1)
    transform = LinesOfSightTransform(
        R_start_vals,
        z_start_vals,
        np.zeros_like(R_start_vals),
        R_stop_vals,
        z_stop_vals,
        np.zeros_like(R_stop_vals),
        num_intervals,
        machine_dims,
    )
    transform.set_equilibrium(MagicMock())
    return transform, vertical, R_vals, z_vals


@composite
def los_coordinates_parameters(
    draw, domain=None, min_los=2, max_los=10, min_num=2, max_num=10, default_Rz=True,
):
    """Generates the arguments needed to instantiate a
    :py:class:`indica.converters.LinesOfSightTransform` object with lines of
    sight radiating from a point.

    Parameters
    ----------
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        A tuple giving the range of R,z,t values for which the transform is
        guaranteed to work: ``((Rmin, Rmax), (zmin, zmax)``. Will
        be used to constrain size of Tokamak.
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    min_num: int
        The minimum number of intervals in which to divide the lines of sight
    max_num: int
        The maximum number of intervals in which to divide the lines of sight

    Returns
    -------
    R_start
        1-D array of major radii of the start for each line-of-sight.
    z_start
        1-D array of vertical positions of the start for each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    num_intervals
        The number of intervals in the default grid for the second coordinate.
        Note that there will be one more points in the grid than this.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``.
    default_R
        Default R-grid to use when converting from the R-z coordinate system.
    default_z
        Default z-grid to use when converting from the R-z coordinate system.

    """
    focus_R = 2.5 + draw(floats(-2.4, 7.5))
    if domain and focus_R >= domain[0][0] and focus_R <= domain[0][1]:
        zdown, zup = domain[1]
        focus_z = draw(
            one_of(
                floats(zup, max(zup + 1.0, 10.0), exclude_min=True),
                floats(min(zdown - 1.0, -10.0), zdown, exclude_max=True),
            )
        )
    else:
        focus_z = draw(floats(-10.0, 10.0))
    if domain:
        Rdown, Rup = domain[0]
        zdown, zup = domain[1]
        if Rdown < 0 or Rup < 0:
            raise ValueError(
                "LinesOfSightTransform does not support domains with R < 0."
            )
        t1 = np.arctan2(domain[1][0] - focus_z, domain[0][0] - focus_R)
        t2 = np.arctan2(domain[1][1] - focus_z, domain[0][0] - focus_R)
        t3 = np.arctan2(domain[1][0] - focus_z, domain[0][1] - focus_R)
        t4 = np.arctan2(domain[1][1] - focus_z, domain[0][1] - focus_R)
        thetas = sorted([t1, t2, t3, t4])
        if thetas[-1] - thetas[0] > np.pi:
            thetas = sorted([-t if t < 0 else t for t in thetas])
        theta_min = draw(floats(-np.pi, thetas[0]))
        theta_max = draw(floats(thetas[-1], 2 * np.pi))
        d1 = np.sqrt((focus_R - domain[0][0]) ** 2 + (focus_z - focus_z[1][0]) ** 2)
        d2 = np.sqrt((focus_R - domain[0][1]) ** 2 + (focus_z - focus_z[1][0]) ** 2)
        d3 = np.sqrt((focus_R - domain[0][0]) ** 2 + (focus_z - focus_z[1][1]) ** 2)
        d4 = np.sqrt((focus_R - domain[0][1]) ** 2 + (focus_z - focus_z[1][1]) ** 2)
        max_dist = min(d1, d2, d3, d4)
        R1 = 1.5 + draw(floats(-1.5, Rdown - 1.5))
        R2 = draw(floats(Rup, max(1e3, 2 * Rup)))
        z1 = draw(floats(min(-1e-3, 2 * zdown), zdown))
        z2 = draw(floats(zup, max(1e3, 2 * zup)))
    else:
        theta_min = 5 * np.pi / 4 + draw(floats(-np.pi, np.pi))
        theta_max = theta_min + np.pi / 4 + draw(floats(-np.pi, np.pi))
        diff = theta_max - theta_min
        if abs(diff) < 0.1:
            theta_max = theta_min + np.sign(diff) * 0.1
        max_dist = 0.5
        R1 = 1.5 + draw(floats(-1.5, focus_R - 1.55, exclude_max=True))
        R2 = draw(floats(focus_R + 0.05, 1e3, exclude_min=True))
        z1 = draw(floats(-1e3, focus_z - 0.1, exclude_max=True))
        z2 = draw(floats(focus_z + 0.1, 1e3, exclude_min=True))
    machine_dims = ((R1, R2), (z1, z2))
    angles = draw(
        monotonic_series(theta_min, theta_max, draw(integers(min_los, max_los)))
    )
    start_distance = draw(floats(0.0, max_dist))
    lengths = draw(
        arrays(
            np.float,
            len(angles),
            elements=floats(-0.4, 1.0, exclude_max=True).map(lambda x: x + 0.5),
            fill=just(0.5),
        )
    )
    R_start = focus_R + start_distance * np.cos(angles)
    z_start = focus_z + start_distance * np.sin(angles)
    R_stop = focus_R + (start_distance + lengths) * np.cos(angles)
    z_stop = focus_z + (start_distance + lengths) * np.sin(angles)
    if default_Rz:
        default_R, default_z, _ = draw(
            basis_coordinates(
                (
                    machine_dims[0][0],
                    machine_dims[1][0],
                    domain[2][0] if domain else None,
                ),
                (
                    machine_dims[0][1],
                    machine_dims[1][1],
                    domain[2][1] if domain else None,
                ),
            )
        )
    else:
        default_R, default_z = None, None
    toroidal_skew = draw(booleans())
    T_start = np.zeros_like(R_start)
    if toroidal_skew:
        skew = draw(smooth_functions((theta_min, theta_max), 0.1))
        T_stop = skew(angles)
    else:
        T_stop = np.zeros_like(R_stop)
    return (
        R_start,
        z_start,
        T_start,
        R_stop,
        z_stop,
        T_stop,
        draw(integers(min_num, max_num)),
        machine_dims,
        default_R,
        default_z,
    )


@composite
def los_coordinates(
    draw, domain=None, min_los=2, max_los=10, min_num=2, max_num=10, default_Rz=True,
):
    """Generates :py:class:`indica.converters.LinesOfSightTransform` objects
    with lines of sight radiating from a point.

    At present this point is on the edge of the Tokamak, for reasons of
    convenience.

    Parameters
    ----------
    domain: Tuple[Tuple[float, float], Tuple[float, float]], optional
        A tuple giving the range of R-z values for which the transform is
        guaranteed to work: ``((Rmin, Rmax), (zmin, zmax)``. If absent will
        draw values.
    machine_dims: Tuple[Tuple[float, float], Tuple[float, float]], optional
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. If absent will draw values.
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    min_num: int
        The minimum number of intervals in which to divide the lines of sight
    max_num: int
        The maximum number of intervals in which to divide the lines of sight

    Returns
    -------
    transform: TransectCoordinates
        The coordinate transform object.
    machine_dims: Tuple[Tuple[float, float], Tuple[float, float]]
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``.

    """
    result = LinesOfSightTransform(
        *draw(
            los_coordinates_parameters(
                domain, min_los, max_los, min_num, max_num, default_Rz
            )
        )
    )
    result.set_equilibrium(MagicMock())
    return result


# Ignore warnings when an empty array
pytestmark = mark.filterwarnings("ignore:invalid value encountered in true_divide")


# TODO: consider converting these tests assuming parallel LoS to work with
# general ones


@given(
    parallel_los_coordinates(),
    floats(0.0, 1.0, exclude_max=True),
    floats(0.0, 1.0),
    floats(),
)
def test_parallel_los_to_Rz(coords, position1, position2, time):
    """Checks positions fall between appropriate lines of sight."""
    transform, vertical, Rvals, zvals = coords
    if vertical:
        i = position1 * (len(Rvals) - 1)
        perp_index = int(position2 * (len(zvals) - 1))
    else:
        i = position1 * (len(zvals) - 1)
        perp_index = int(position2 * (len(Rvals) - 1))
    los_index = int(i)
    R, z, t = transform.convert_to_Rz(i, position2, time)
    R_index = los_index if vertical else perp_index
    if Rvals[-1] > Rvals[0]:
        assert R - Rvals[R_index + 1] <= 1e-5
        assert Rvals[R_index] - R <= 1e-5
    else:
        assert R - Rvals[R_index + 1] >= -1e-5
        assert Rvals[R_index] - R >= -1e-5
    z_index = perp_index if vertical else los_index
    if zvals[-1] > zvals[0]:
        assert z <= zvals[z_index + 1]
        assert zvals[z_index] <= z
    else:
        assert z >= zvals[z_index + 1]
        assert zvals[z_index] >= z


@given(
    parallel_los_coordinates(), floats(),
)
def test_parallel_los_from_Rz(coords, time):
    """Checks R,z points along linse of sight have correct channel number."""
    transform, vertical, Rvals, zvals = coords
    for (i, R), (j, z) in product(enumerate(Rvals), enumerate(zvals)):
        ch, pos, t = transform.convert_from_Rz(R, z, time)
        if vertical:
            assert ch == approx(i, abs=1e-5, rel=1e-2)
            assert pos == approx(
                (z - zvals[0]) / (zvals[-1] - zvals[0]), abs=1e-5, rel=1e-2
            )
        else:
            assert ch == approx(j, abs=1e-5, rel=1e-2)
            assert pos == approx(
                (R - Rvals[0]) / (Rvals[-1] - Rvals[0]), abs=1e-5, rel=1e-2
            )


@given(
    los_coordinates(),
    floats(0.0, 1.0, exclude_max=True),
    floats(0.0, 1.0, exclude_min=True),
    integers(2, 50),
    floats(),
)
def test_los_uniform_distances(transform, start, end, steps, time):
    """Test distances are uniform along lines of sight"""
    samples = np.linspace(start, end, steps)
    distance, t = transform.distance(1, x2=samples, t=time)
    assert np.all(np.isclose(distance[1:, :], distance[0, :], 1e-6, 1e-12))


@given(los_coordinates(), integers(2, 50), floats())
def test_los_distances(transform, npoints, time):
    """Tests distances along the line of sight are correct."""
    lengths = np.expand_dims(
        np.sqrt(
            (transform.R_end - transform.R_start) ** 2
            + (transform.z_end - transform.z_start) ** 2
            + (transform.T_end - transform.T_start) ** 2
        ),
        1,
    )
    samples = np.linspace(0.0, 1.0, npoints)
    distance, t = transform.distance(1, x2=samples, t=time)
    assert distance == approx(lengths * samples)


@given(los_coordinates_parameters(), floats())
def test_los_end_points(parameters, time):
    """Test end of all lines fall on edge or outside of reactor dimensions"""
    transform = LinesOfSightTransform(*parameters)
    dims = parameters[7]
    R, z, t = transform.convert_to_Rz(x2=1.0, t=time)
    assert np.all(np.logical_not(inside_machine((R, z), dims, False)))


@given(los_coordinates_parameters(), arbitrary_coordinates())
def test_los_default_Rz(parameters, Rz_defaults):
    """Test expected defaults are used in transforms for R and z"""
    R_default, z_default, _ = Rz_defaults
    transform = LinesOfSightTransform(*parameters[:-2], R_default, z_default)
    x1, x2, t = transform.convert_from_Rz(R_default, z_default)
    x1_default, x2_deafult, t = transform.convert_from_Rz()
    assert np.all(x1 == x1_default)
    assert np.all(x2 == x2_deafult)
