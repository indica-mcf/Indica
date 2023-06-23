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
from hypothesis.strategies import text
import numpy as np
from pytest import approx
from pytest import mark
from xarray import DataArray
from xarray.testing import assert_allclose

from indica.converters import LinesOfSightTransform
from indica.utilities import coord_array
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
        x_stop, x_start = machine_dims[0]
    else:
        x_start, x_stop = machine_dims[0]
    if draw(booleans()):
        z_stop, z_start = machine_dims[1]
    else:
        z_start, z_stop = machine_dims[1]
    vertical = draw(booleans())
    num_los = draw(integers(min_los, max_los))
    num_intervals = draw(integers(min_num, max_num))
    if vertical:
        x_start_vals = x_stop_vals = draw(monotonic_series(x_start, x_stop, num_los))
        x_vals = DataArray(x_start_vals, dims="x")
        z_start_vals = np.ones(num_los) * z_start
        z_stop_vals = np.ones(num_los) * z_stop
        z_vals = DataArray(np.linspace(z_start, z_stop, num_intervals + 1), dims="z")
    else:
        z_start_vals = z_stop_vals = draw(monotonic_series(z_start, z_stop, num_los))
        z_vals = DataArray(z_start_vals, dims="z")
        x_start_vals = np.ones(num_los) * x_start
        x_stop_vals = np.ones(num_los) * x_stop
        x_vals = DataArray(np.linspace(x_start, x_stop, num_intervals + 1), dims="x")
    transform = LinesOfSightTransform(
        x_start_vals,
        z_start_vals,
        np.zeros_like(x_start_vals),
        x_stop_vals,
        z_stop_vals,
        np.zeros_like(x_stop_vals),
        draw(text()),
        machine_dims,
    )
    transform.set_equilibrium(MagicMock())
    return transform, vertical, x_vals, z_vals


@composite
def los_coordinates_parameters(
    draw,
    domain=None,
    min_los=2,
    max_los=10,
    domain_as_dims=False,
    toroidal_skew=None,
    name=None,
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
    domain_as_dims: bool
        If True, use the domain as the machine dimensions.
    toroidal_skew: Optional[bool]
        Whether to include any toroidal angle to the lines of sight. Defaults
        to drawing a value to decide.
    name: Optional[str]
        The name of the camera these lines of sight are for.

    Returns
    -------
    x_start
        1-D array of x positions of the start for each line-of-sight.
    z_start
        1-D array of z positions of the start for each line-of-sight.
    y_start
        1-D array of y positions of the start for each line-of-sight.
    x_end
        1-D array of x positions of the end for each line-of-sight.
    z_end
        1-D array of z positions of the end for each line-of-sight.
    y_end
        1-D array of y positions of the end for each line-of-sight.
    name: Optional[str]
        The name of the camera these lines of sight are for.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (xmin, xmax)``.

    """
    focus_x = 2.5 + draw(floats(-2.4, 7.5))
    if domain and focus_x >= domain[0][0] and focus_x <= domain[0][1]:
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
        R_down, R_up = domain[0]
        z_down, z_up = domain[1]
        if R_down < 0 or R_up < 0:
            raise ValueError(
                "LinesOfSightTransform does not support domains with R < 0."
            )
        t1 = np.arctan2(domain[1][0] - focus_z, domain[0][0] - focus_x)
        t2 = np.arctan2(domain[1][1] - focus_z, domain[0][0] - focus_x)
        t3 = np.arctan2(domain[1][0] - focus_z, domain[0][1] - focus_x)
        t4 = np.arctan2(domain[1][1] - focus_z, domain[0][1] - focus_x)
        thetas = sorted([t1, t2, t3, t4])
        if thetas[-1] - thetas[0] > np.pi:
            thetas = sorted([np.pi - t if t < 0 else t for t in thetas])
        theta_range = thetas[-1] - thetas[0]
        theta_min = draw(
            floats(
                thetas[0] + 0.01 * theta_range,
                thetas[0] + 0.85 * theta_range,
                exclude_min=True,
                exclude_max=True,
            )
        )
        theta_max = thetas[-1] - draw(
            floats(
                0.01 * theta_range,
                thetas[-1] - theta_min - 0.05 * theta_range,
                exclude_min=True,
                exclude_max=True,
            )
        )
        d1 = np.sqrt((focus_x - domain[0][0]) ** 2 + (focus_z - domain[1][0]) ** 2)
        d2 = np.sqrt((focus_x - domain[0][1]) ** 2 + (focus_z - domain[1][0]) ** 2)
        d3 = np.sqrt((focus_x - domain[0][0]) ** 2 + (focus_z - domain[1][1]) ** 2)
        d4 = np.sqrt((focus_x - domain[0][1]) ** 2 + (focus_z - domain[1][1]) ** 2)
        max_dist = min(d1, d2, d3, d4)
        R_1 = 1.5 + draw(floats(-1.5, R_down - 1.5))
        R_2 = draw(floats(R_up, max(1e3, 2 * R_up)))
        z_1 = draw(floats(min(-1e-3, 2 * z_down), z_down))
        z_2 = draw(floats(z_up, max(1e3, 2 * z_up)))
    else:
        theta_min = 5 * np.pi / 4 + draw(
            floats(-np.pi, np.pi, exclude_min=True, exclude_max=True)
        )
        theta_max = (
            theta_min
            + np.pi / 4
            + draw(floats(-np.pi, np.pi, exclude_min=True, exclude_max=True))
        )
        diff = theta_max - theta_min
        if abs(diff) < 0.1:
            theta_max = theta_min + np.sign(diff) * 0.1
        max_dist = 0.5
        R_1 = 1.5 + draw(floats(-1.5, focus_x - 1.55, exclude_max=True))
        R_2 = draw(floats(focus_x + 0.05, 1e3, exclude_min=True))
        z_1 = draw(floats(-1e3, focus_z - 0.1, exclude_max=True))
        z_2 = draw(floats(focus_z + 0.1, 1e3, exclude_min=True))
    if domain_as_dims:
        machine_dims = domain
    else:
        machine_dims = ((R_1, R_2), (z_1, z_2))
    angles = draw(
        monotonic_series(theta_min, theta_max, draw(integers(min_los, max_los)))
    )
    start_distance = draw(floats(0.0, max_dist))
    lengths = draw(
        arrays(
            np.float64,
            len(angles),
            elements=floats(-0.4, 1.0, exclude_max=True).map(lambda x: x + 0.5),
            fill=just(0.5),
        )
    )
    x_start = focus_x + start_distance * np.cos(angles)
    z_start = focus_z + start_distance * np.sin(angles)
    x_stop = focus_x + (start_distance + lengths) * np.cos(angles)
    z_stop = focus_z + (start_distance + lengths) * np.sin(angles)
    if toroidal_skew is None:
        toroidal_skew = draw(booleans())
    y_start = np.zeros_like(x_start)
    if toroidal_skew:
        skew = draw(smooth_functions((theta_min, theta_max), 0.1))
        y_stop = skew(angles)
    else:
        y_stop = np.zeros_like(x_stop)
    if name is None:
        name = draw(text())
    return (
        x_start,
        z_start,
        y_start,
        x_stop,
        z_stop,
        y_stop,
        name,
        machine_dims,
    )


@composite
def los_coordinates(
    draw,
    domain=None,
    min_los=2,
    max_los=10,
    domain_as_dims=False,
    toroidal_skew=None,
    name=None,
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
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    domain_as_dims: bool
        If True, use the domain as the machine dimensions.
    toroidal_skew: Optional[bool]
        Whether to include any toroidal angle to the lines of sight. Defaults
        to drawing a value to decide.
    name: Optional[str]
        The name of the camera these lines of sight are for.

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
                domain,
                min_los,
                max_los,
                domain_as_dims,
                toroidal_skew,
                name,
            )
        )
    )
    result.set_equilibrium(MagicMock())
    return result


@composite
def los_coordinates_and_axes(
    draw,
    domain=None,
    min_los=2,
    max_los=10,
    domain_as_dims=False,
    toroidal_skew=None,
    name=None,
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
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    domain_as_dims: bool
        If True, use the domain as the machine dimensions.
    toroidal_skew: Optional[bool]
        Whether to include any toroidal angle to the lines of sight. Defaults
        to drawing a value to decide.
    name: Optional[str]
        The name of the camera these lines of sight are for.

    Returns
    -------
    transform: TransectCoordinates
        The coordinate transform object.
    machine_dims: Tuple[Tuple[float, float], Tuple[float, float]]
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``.

    """
    transform = draw(
        los_coordinates(
            domain,
            min_los,
            max_los,
            domain_as_dims,
            toroidal_skew,
            name,
        )
    )
    x1 = coord_array(np.arange(len(transform.x_start)), transform.x1_name)
    x2 = DataArray(0)
    t = DataArray(0)
    return transform, x1, x2, t


# Ignore warnings when an empty array
pytestmark = mark.filterwarnings(
    "ignore:(invalid value|divide by zero) encountered in true_divide"
)


# TODO: consider converting these tests assuming parallel LoS to work with
# general ones


@given(
    parallel_los_coordinates(),
    floats(0.0, 1.0, exclude_max=True),
    floats(0.0, 1.0, exclude_max=True),
    floats(),
)
def test_parallel_los_to_Rz(coords, position1, position2, time):
    """Checks positions fall between appropriate lines of sight."""
    transform, vertical, x_vals, z_vals = coords
    if vertical:
        i = position1 * (len(x_vals) - 1)
        perp_index = int(position2 * (len(z_vals) - 1))
    else:
        i = position1 * (len(z_vals) - 1)
        perp_index = int(position2 * (len(x_vals) - 1))
    los_index = int(i)
    x, z = transform.convert_to_Rz(i, position2, time)
    x_index = los_index if vertical else perp_index
    if x_vals[-1] > x_vals[0]:
        assert x - x_vals[x_index + 1] <= 1e-5
        assert x_vals[x_index] - x <= 1e-5
    else:
        assert x - x_vals[x_index + 1] >= -1e-5
        assert x_vals[x_index] - x >= -1e-5
    z_index = perp_index if vertical else los_index
    if z_vals[-1] > z_vals[0]:
        assert z <= z_vals[z_index + 1]
        assert z_vals[z_index] <= z
    else:
        assert z >= z_vals[z_index + 1]
        assert z_vals[z_index] >= z


@given(
    parallel_los_coordinates(),
    floats(),
)
@mark.xfail(reason="Conversion from R-z is not reliably implemented.")
def test_parallel_los_from_Rz(coords, time):
    """Checks R,z points along linse of sight have correct channel number."""
    transform, vertical, x_vals, z_vals = coords
    for (i, x), (j, z) in product(enumerate(x_vals), enumerate(z_vals)):
        ch, pos = transform.convert_from_Rz(x, z, time)
        if vertical:
            assert ch == approx(i, abs=1e-5, rel=1e-2)
            assert pos == approx(
                (z - z_vals[0]) / (z_vals[-1] - z_vals[0]), abs=1e-5, rel=1e-2
            )
        else:
            assert ch == approx(j, abs=1e-5, rel=1e-2)
            assert pos == approx(
                (x - x_vals[0]) / (x_vals[-1] - x_vals[0]), abs=1e-5, rel=1e-2
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
    lines = DataArray(np.arange(len(transform.x_start)), dims="index")
    samples = DataArray(np.linspace(start, end, steps), dims="x2")
    distance = transform.distance("x2", lines, samples, time)
    assert np.all(
        np.isclose(
            distance.sel(index=slice(1, None)), distance.sel(index=0), 1e-6, 1e-12
        )
    )


@given(los_coordinates(), integers(2, 50), floats())
def test_los_distances(transform, npoints, time):
    """Tests distances along the line of sight are correct."""
    lengths = DataArray(
        np.sqrt(
            (transform.x_end - transform.x_start) ** 2
            + (transform.z_end - transform.z_start) ** 2
            + (transform.y_end - transform.y_start) ** 2
        ),
        dims="index",
    )
    lines = DataArray(np.arange(len(transform.x_start)), dims="index")
    samples = DataArray(np.linspace(0.0, 1.0, npoints), dims="x2")
    distance = transform.distance("x2", lines, samples, time)
    assert_allclose(distance, lengths * samples)


@given(los_coordinates_parameters(), floats())
def test_los_end_points(parameters, time):
    """Test end of all lines fall on edge or outside of reactor dimensions"""
    transform = LinesOfSightTransform(*parameters)
    lines = DataArray(np.arange(len(transform.x_start)), dims="index")
    dims = parameters[7]
    x, z = transform.convert_to_Rz(lines, 1.0, time)
    assert np.all(np.logical_not(inside_machine((x, z), dims, False)))
