"""Tests for line-of-sight coordinate transforms."""

from itertools import product

from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists
import numpy as np
from pytest import approx

from src.converters import LinesOfSightTransform
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates
from ..strategies import monotonic_series
from ..strategies import sane_floats


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
        return (
            dimensions[0][0] <= coords[0]
            and coords[0] <= dimensions[0][1]
            and dimensions[1][0] <= coords[1]
            and coords[1] <= dimensions[1][1]
        )
    else:
        return (
            dimensions[0][0] < coords[0]
            and coords[0] < dimensions[0][1]
            and dimensions[1][0] < coords[1]
            and coords[1] < dimensions[1][1]
        )


@composite
def machine_dimensions(draw):
    """Generates tuples describing the size of a tokamak."""
    R1 = draw(sane_floats())
    R2 = draw(sane_floats().filter(lambda x: x != approx(R1, rel=1e-3, abs=1e-3)))
    z1 = draw(sane_floats())
    z2 = draw(sane_floats().filter(lambda x: x != approx(z1, rel=1e-3, abs=1e-3)))
    return ((min(R1, R2), max(R1, R2)), (min(z1, z2), max(z1, z2)))


@composite
def parallel_los_coordinates(
    draw, machine_dims=None, min_los=2, max_los=100, min_num=2, max_num=100
):
    """Generates :py:class:`src.converters.LinesOfSightTransform`  objects
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
    if draw(booleans()):
        R_start, R_stop = machine_dimensions[0]
    else:
        R_stop, R_start = machine_dimensions[0]
    if draw(booleans()):
        z_start, z_stop = machine_dimensions[1]
    else:
        z_stop, z_start = machine_dimensions[1]
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
        R_stop_vals,
        z_stop_vals,
        num_intervals,
        machine_dimensions,
    )
    return transform, vertical, R_vals, z_vals


@composite
def los_coordinates_parameters(
    draw, machine_dims=None, min_los=2, max_los=100, min_num=2, max_num=100
):
    """Generates the arguments needed to instantiate a
    :py:class:`src.converters.LinesOfSightTransform` object with lines of
    sight radiating from a point.

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
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.
    default_R
        Default R-grid to use when converting from the R-z coordinate system.
    default_z
        Default z-grid to use when converting from the R-z coordinate system.

    """
    # TODO: consider making this more general
    if not machine_dims:
        machine_dims = draw(machine_dimensions())
    edge = draw(integers(0, 3))
    start_length = draw(
        floats(
            0.0,
            (
                machine_dims[0][1] - machine_dims[0][0]
                if edge % 2 == 0
                else machine_dims[1][1] - machine_dims[1][0]
            )
            * 0.1,
        )
    )
    if edge == 0:
        R_origin = draw(
            floats(machine_dims[0][0] + start_length, machine_dims[0][1] - start_length)
        )
        z_origin = machine_dims[1][1]
        theta_min = np.pi
        theta_max = 2 * np.pi
    elif edge == 1:
        R_origin = machine_dims[0][0]
        z_origin = draw(
            floats(machine_dims[1][0] + start_length, machine_dims[0][1] - start_length)
        )
        theta_min = -0.5 * np.pi
        theta_max = 0.5 * np.pi
    elif edge == 2:
        R_origin = draw(
            floats(machine_dims[0][0] + start_length, machine_dims[0][1] - start_length)
        )
        z_origin = machine_dims[1][0]
        theta_min = 0.0
        theta_max = np.pi
    else:
        R_origin = machine_dims[0][1]
        z_origin = draw(
            floats(machine_dims[1][0] + start_length, machine_dims[1][1] - start_length)
        )
        theta_min = 0.5 * np.pi
        theta_max = 1.5 * np.pi
    angles = np.array(
        draw(
            lists(
                floats(theta_min, theta_max),
                min_size=min_los,
                max_size=max_los,
                unique=True,
            ).map(sorted)
        )
    )
    z_start = z_origin + start_length * np.tan(angles)
    R_start = R_origin + start_length * np.tan(angles)
    z_stop = np.array(
        [
            z_start
            if theta == 0.0 or theta == np.pi
            else draw(
                floats(
                    z_start,
                    machine_dims[1][1] if np.tan(theta) > 1 else machine_dims[1][0],
                )
            )
            for theta in angles
        ]
    )
    R_stop = [
        draw(floats(R_start, machine_dims[0][1]))
        if theta == 0.0
        else draw(floats(machine_dims[0][0], R_start))
        if theta == np.pi
        else R_origin + (z_stop - z_origin) / np.tan(theta)
        for theta in angles
    ]
    default_R, default_z, _ = draw(
        basis_coordinates(
            (machine_dims[0][0], machine_dims[1][0], None),
            (machine_dims[0][1], machine_dims[1][1], None),
        )
    )
    return (
        R_start,
        z_start,
        R_stop,
        z_stop,
        draw(integers(min_num, max_num)),
        machine_dims,
        default_R,
        default_z,
    )


@composite
def los_coordinates(
    draw, machine_dims=None, min_los=2, max_los=100, min_num=2, max_num=100
):
    """Generates :py:class:`src.converters.LinesOfSightTransform` objects
    with lines of sight radiating from a point.

    At present this point is on the edge of the Tokamak, for reasons of
    convenience.

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
    machine_dims: Tuple[Tuple[float, float], Tuple[float, float]]
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``.

    """
    return LinesOfSightTransform(
        *draw(
            los_coordinates_parameters(machine_dims, min_los, max_los, min_num, max_num)
        )
    )


# TODO: consider converting these tests assuming parallel Los to work with
# general ones


@given(
    parallel_los_coordinates(),
    floats(0.0, 1.0, exclude_max=True),
    sane_floats(),
    floats(),
)
def tests_parallel_los_to_Rz(coords, position1, position2, time):
    """Checks positions fall between appropriate lines of sight."""
    transform, vertical, Rvals, zvals = coords
    if vertical:
        i = position1 * (len(Rvals) - 1)
        perp_index = len(zvals) if position2 == 1.0 else np.argwhere(position2 < zvals)
    else:
        i = position1 * (len(zvals) - 1)
        perp_index = len(Rvals) if position2 == 1.0 else np.argwhere(position2 < Rvals)
    los_index = int(i)
    R, z, t = transform.convert_to_Rz(i, position2, time)
    R_index = los_index if vertical else perp_index
    if Rvals[-1] > Rvals[0]:
        assert R <= Rvals[R_index + 1]
        assert Rvals[R_index] <= R
    else:
        assert R >= Rvals[R_index + 1]
        assert Rvals[R_index] >= R
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
def tests_parallel_los_from_Rz(coords, time):
    """Checks R,z points along linse of sight have correct channel number."""
    transform, vertical, Rvals, zvals = coords
    for (i, R), (j, z) in product(enumerate(Rvals), enumerate(zvals)):
        ch, pos, t = transform.convert_from_Rz(R, z, time)
        if vertical:
            assert ch == approx(i)
        else:
            assert ch == approx(j)


@given(
    los_coordinates(),
    floats(0.0, 1.0, exclude_max=True),
    floats(0.0, 1.0, exclude_min=True),
    integers(1, 50),
    floats(),
)
def test_los_uniform_distances(transform, start, end, steps, time):
    """Test distances are uniform along lines of sight"""
    samples = np.expand_dims(np.linspace(start, end, steps), 1)
    distance, t = transform.distance(x2=samples, t=time)
    assert np.all(distance[:, 1:] == approx(distance[:, 0]))


@given(los_coordinates_parameters(), floats())
def test_los_end_points(parameters, time):
    """Test end of all lines fall on edge or outside of reactor dimensions"""
    transform = LinesOfSightTransform(*parameters)
    dims = parameters[5]
    R, z, t = transform.convert_to_Rz(x2=1.0, t=time)
    assert np.all(inside_machine((R, z), dims, False))


@given(los_coordinates_parameters(), arbitrary_coordinates())
def test_los_default_Rz(parameters, Rz_defaults):
    """Test expected defaults are used in transforms for R and z"""
    R_default, z_default = Rz_defaults
    transform = LinesOfSightTransform(*parameters[:-2], R_default, z_default)
    x1, x2, t = transform.convert_from_Rz(R_default, z_default)
    x1_default, x2_deafult, t = transform.convert_from_Rz()
    assert np.all(x1 == x1_default)
    assert np.all(x2 == x2_deafult)
