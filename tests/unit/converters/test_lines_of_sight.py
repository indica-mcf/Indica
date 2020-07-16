"""Tests for line-of-sight coordinate transforms."""

from itertools import product
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
import numpy as np
from pytest import approx

from src.converters import LinesOfSightTransform
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates
from ..strategies import machine_dimensions
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


def get_wall_intersection(R_start, z_start, angles, dimensions):
    """Returns the location where a line of sight interects with the Tokamak
    wall. This will be whichever wall the starting point is further away from.

    """
    distances = [
        np.abs(dimensions[1][1] - z_start),
        np.abs(dimensions[0][0] - R_start),
        np.abs(dimensions[1][0] - z_start),
        np.abs(dimensions[0][1] - R_start),
    ]
    nearest_wall = np.argmix(distances, 0)
    tan_theta = np.tan(angles)
    choices_horizontal = np.where(
        angles % (2 * np.pi) == 0, 0, np.where(angles % (2 * np.pi) == np.pi, 1, 2)
    )
    choices_vertical = np.where(
        angles % (2 * np.pi) == np.pi / 2,
        0,
        np.where(angles % (2 * np.pi) == 3 * np.pi / 2, 1, 2),
    )
    intersections = [
        (
            np.choose(
                choices_horizontal,
                [
                    dimensions[0][1],
                    dimensions[0][0],
                    R_start + (dimensions[1][1] - z_start) / tan_theta,
                ],
            ),
            dimensions[1][1],
        ),
        (
            dimensions[0][0],
            np.choose(
                choices_vertical,
                [
                    dimensions[1][1],
                    dimensions[1][0],
                    z_start + tan_theta * (dimensions[0][0] - R_start),
                ],
            ),
        ),
        (
            np.choose(
                choices_horizontal,
                [
                    dimensions[0][1],
                    dimensions[0][0],
                    R_start + (dimensions[1][0] - z_start) / tan_theta,
                ],
            ),
            dimensions[1][0],
        ),
        (
            dimensions[0][1],
            np.choose(
                choices_vertical,
                [
                    dimensions[1][1],
                    dimensions[1][0],
                    z_start + tan_theta * (dimensions[0][1] - R_start),
                ],
            ),
        ),
    ]
    tol = 1e-12
    in_tokamak = [
        dimensions[i % 2][0] * (1 + tol)
        <= intersections[i][i % 2]
        <= dimensions[i % 2][1] * (1 + tol)
        for i in range(4)
    ]
    nearest_wall_p1 = nearest_wall + 1
    intersect_wall = np.argmax(
        in_tokamak[:nearest_wall] + in_tokamak[nearest_wall_p1:], 0
    )
    return [intersections[wall][i] for i, wall in enumerate(intersect_wall)]


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
        R_start, R_stop = machine_dims[0]
    else:
        R_stop, R_start = machine_dims[0]
    if draw(booleans()):
        z_start, z_stop = machine_dims[1]
    else:
        z_stop, z_start = machine_dims[1]
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
        machine_dims,
    )
    transform.set_equilibrium(MagicMock())
    return transform, vertical, R_vals, z_vals


@composite
def los_coordinates_parameters(
    draw, domain=None, min_los=2, max_los=100, min_num=2, max_num=100
):
    """Generates the arguments needed to instantiate a
    :py:class:`src.converters.LinesOfSightTransform` object with lines of
    sight radiating from a point.

    Parameters
    ----------
    domain: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        A tuple giving the range of R,z,t values for which the transform is
        guaranteed to work: ``((Rmin, Rmax), (zmin, zmax), (tmin, tmax)``. Will
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
    # TODO: consider making this more general

    def theta_range(origin, hard_min, hard_max, domain, edge):
        min_i = 0 if edge == 0 or edge == 1 else 1
        max_i = 0 if edge == 1 or edge == 2 else 1
        min_j = 0 if edge == 1 or edge == 2 else 1
        max_j = 0 if edge == 2 or edge == 3 else 1
        if domain:
            theta_min = np.arctan2(
                (origin[1] - domain[1][min_j]), (origin[0] - domain[0][min_i])
            )
            theta_max = np.arctan2(
                (origin[1] - domain[1][max_j]), (origin[0] - domain[0][max_i])
            )
            if theta_min < theta_max:
                theta_max += 2 * np.pi
        else:
            theta1 = draw(floats(hard_min, hard_max))
            theta2 = draw(
                floats(hard_min, hard_max).filter(
                    lambda x: x != approx(theta1, abs=1e-3, rel=1e-3)
                )
            )
            theta_max = max(theta1, theta2)
            theta_min = min(theta1, theta2)
        return theta_min, theta_max

    if not domain:
        machine_dims = draw(machine_dimensions())
    else:
        machine_dims = (
            (
                draw(floats(max_value=domain[0][0], allow_infinity=False)),
                draw(floats(min_value=domain[0][1], allow_infinity=False)),
            ),
            (
                draw(floats(max_value=domain[1][0], allow_infinity=False)),
                draw(floats(min_value=domain[1][1], allow_infinity=False)),
            ),
        )
    edge = draw(integers(0, 3))
    start_length = draw(
        floats(
            0.0,
            min(
                machine_dims[0][1] - machine_dims[0][0],
                machine_dims[1][1] - machine_dims[1][0],
            )
            * 0.05,
        )
    )
    if edge == 0:
        centre = (machine_dims[0][1] + machine_dims[0][0]) / 2
        half_width = (machine_dims[0][1] - machine_dims[0][0]) / 2
        R_origin = centre + draw(floats(-0.8 * half_width, 0.8 * half_width))
        z_origin = machine_dims[1][1]
        theta_min, theta_max = theta_range(
            (R_origin, z_origin), np.pi, 2 * np.pi, domain, edge
        )
    elif edge == 1:
        centre = (machine_dims[1][1] + machine_dims[1][0]) / 2
        half_width = (machine_dims[1][1] - machine_dims[1][0]) / 2
        R_origin = machine_dims[0][0]
        z_origin = centre + draw(floats(-0.8 * half_width, 0.8 * half_width))
        theta_min, theta_max = theta_range(
            (R_origin, z_origin), -0.5 * np.pi, 0.5 * np.pi, domain, edge
        )
    elif edge == 2:
        centre = (machine_dims[0][1] + machine_dims[0][0]) / 2
        half_width = (machine_dims[0][1] - machine_dims[0][0]) / 2
        R_origin = centre + draw(floats(-0.8 * half_width, 0.8 * half_width))
        z_origin = machine_dims[1][0]
        theta_min, theta_max = theta_range(
            (R_origin, z_origin), 0.0, np.pi, domain, edge
        )
    else:
        centre = (machine_dims[1][1] + machine_dims[1][0]) / 2
        half_width = (machine_dims[1][1] - machine_dims[1][0]) / 2
        R_origin = machine_dims[0][1]
        z_origin = centre + draw(floats(-0.8 * half_width, 0.8 * half_width))
        theta_min, theta_max = theta_range(
            (R_origin, z_origin), 0.5 * np.pi, 1.5 * np.pi, domain, edge
        )
    angles = draw(
        monotonic_series(theta_min, theta_max, draw(integers(min_los, max_los)))
    )
    z_start = z_origin + start_length * np.cos(angles)
    R_start = R_origin + start_length * np.sin(angles)
    stop = get_wall_intersection(R_start, z_start, angles, machine_dims)
    distance = (
        1
        - arrays(
            np.float, len(angles), elements=floats(0.0, 1.0, exclude_max=True), fill=0.0
        )
    ) * np.sqrt((R_start - stop[:, 0]) ** 2 + (z_start - stop[:, 1]))
    R_stop = R_start + distance * np.cos(angles)
    z_stop = z_start + distance * np.sin(angles)
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
def los_coordinates(draw, domain=None, min_los=2, max_los=100, min_num=2, max_num=100):
    """Generates :py:class:`src.converters.LinesOfSightTransform` objects
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
        *draw(los_coordinates_parameters(domain, min_los, max_los, min_num, max_num))
    )
    result.set_equilibrium(MagicMock())
    return result


# TODO: consider converting these tests assuming parallel LoS to work with
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
            assert pos == approx(z / (zvals[-1] - zvals[0]))
        else:
            assert ch == approx(j)
            assert pos == approx(R / (Rvals[-1] - Rvals[0]))


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
    R_default, z_default, _ = Rz_defaults
    transform = LinesOfSightTransform(*parameters[:-2], R_default, z_default)
    x1, x2, t = transform.convert_from_Rz(R_default, z_default)
    x1_default, x2_deafult, t = transform.convert_from_Rz()
    assert np.all(x1 == x1_default)
    assert np.all(x2 == x2_deafult)
