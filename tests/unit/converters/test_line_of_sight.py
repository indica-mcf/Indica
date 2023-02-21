"""Tests for line-of-sight coordinate transforms."""

from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray

from indica import equilibrium
from indica.converters import line_of_sight
from indica.models.equilibrium import fake_equilibrium_data
from indica.utilities import intersection


def default_inputs():
    """Default inputs for a single line of sight, no time dependence"""
    x1 = 0.0
    x2 = DataArray(np.linspace(0.0, 1.0, 350, dtype=float))
    t = 0.0

    return x1, x2, t


def load_line_of_sight_default():
    origin = np.array([[3.8, -2.0, 0.5], [3.8, -2.0, 0.0]])
    direction = np.array([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))
    name = "los_test"
    los = line_of_sight.LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        machine_dimensions=machine_dims,
        name=name,
    )
    return los, machine_dims


def load_equilibrium_default():
    data = fake_equilibrium_data()
    equil = equilibrium.Equilibrium(
        data,
        sess=MagicMock(),
    )
    return equil


def _test_check_rho():
    """To be implemented"""


def test_convert_to_xy(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2[0]
    t = 0

    # Test method
    x, y = los.convert_to_xy(x1, x2, t)
    _, z = los.convert_to_Rz(x1, x2, t)

    assert np.all(x.values <= np.max([los.x_start, los.x_end]))
    assert np.all(x >= np.min([los.x_start, los.x_end]))
    assert np.all(y <= np.max([los.y_start, los.y_end]))
    assert np.all(y >= np.min([los.y_start, los.y_end]))
    assert np.all(z <= np.max([los.z_start, los.z_end]))
    assert np.all(z >= np.min([los.z_start, los.z_end]))

    if debug:
        print(f"x = {x}")
        print(f"y = {y}")
        print(f"z = {z}")


# Test convert_to_Rz method
def test_convert_to_Rz(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2
    t = 0

    # Test method
    R_, z_ = los.convert_to_Rz(x1, x2, t)

    x, y = los.convert_to_xy(x1, x2, t)
    R = np.sqrt(x**2 + y**2)

    # R and z are as expected=
    assert all(R == R_)

    if debug:
        print(f"R = {R}")


# Test distance method
def test_distance(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()
    x1 = 0
    x2 = los.x2
    t = 0

    # Test method
    dist = los.distance("los_position", x1, x2, t)
    dls = [dist[i + 1] - dist[i] for i in range(len(dist) - 1)]

    # dl is identical along the line of sight up to 1 per million
    assert all(np.abs(dls - dls[0]) < (dls[0] * 1.0e-6))

    if debug:
        print(f"dist = {dist}")
    return


# Test distance method
def test_set_dl(debug=False):
    # Load line-of-sight default
    los, machine_dims = load_line_of_sight_default()

    # Test inputs
    dl = 0.002

    # Test method
    los.set_dl(dl)

    assert np.abs(dl - los.dl) < 1.0e-4

    if debug:
        print(f"x2 = {los.x2}")
        print(f"dl_out = {los.dl}")
    return


# Test script for intersections
def test_intersections(debug=False):
    """Test script for intersections"""

    # Test parallel lines -> should return an empty list
    line_1_x = np.array([0.0, 1.0])
    line_1_y = np.array([1.0, 2.0])
    line_2_x = np.array([0.0, 1.0])
    line_2_y = np.array([2.0, 3.0])

    rx, zx, _, _ = intersection(line_1_x, line_1_y, line_2_x, line_2_y)
    assert len(rx) == 0
    assert len(zx) == 0

    if debug:
        print(rx)
        print(zx)

    # Test intersecting lines - should return list of len=1
    line_3_x = np.array([0.0, 1.0])
    line_3_y = np.array([2.0, 1.0])
    rx, zx, _, _ = intersection(line_1_x, line_1_y, line_3_x, line_3_y)
    assert len(rx) != 0
    assert len(zx) != 0

    if debug:
        print(rx)
        print(zx)


# Test LOS missing vessel
def test_missing_los():
    # Line of sight origin tuple
    origin = np.array(
        [
            [4.0, -2.0, 0.5],
        ]
    )  # [xyz]

    # Line of sight direction
    direction = np.array(
        [
            [0.0, 1.0, 0.0],
        ]
    )  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    try:
        _ = line_of_sight.LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            machine_dimensions=machine_dims,
            name=name,
        )
    except ValueError:
        # Value Error since the LOS does not intersect with machine dimensions
        print("LOS initialisation failed with ValueError as expected")
