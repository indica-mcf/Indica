"""Tests for line-of-sight coordinate transforms."""

import numpy as np

from indica.converters import TransectCoordinates
from indica.defaults.load_defaults import load_default_objects


def default_inputs():
    """Default inputs for a single line of sight, no time dependence"""
    nchannels = 11
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)
    return x_positions, y_positions, z_positions


def load_transect_default():
    x_positions, y_positions, z_positions = default_inputs()
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))
    name = "transect_test"
    transect = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        name=name,
        machine_dimensions=machine_dims,
    )
    return transect, machine_dims


def load_equilibrium_default(machine: str = "st40"):
    equilibrium = load_default_objects(machine, "equilibrium")
    return equilibrium


def test_convert_to_xy():
    x_positions, y_positions, z_positions = default_inputs()
    R_positions = np.sqrt(x_positions**2 + y_positions**2)
    transform, machine_dims = load_transect_default()
    x1 = transform.x1
    x2 = transform.x2
    t = 0

    x, y = transform.convert_to_xy(x1, x2, t)
    R, z = transform.convert_to_Rz(x1, x2, t)

    assert np.array_equal(x, x_positions)
    assert np.array_equal(y, y_positions)
    assert np.array_equal(z, z_positions)
    assert np.array_equal(R, R_positions)


def test_convert_to_Rz():
    x_positions, y_positions, z_positions = default_inputs()
    transform, machine_dims = load_transect_default()
    x1 = transform.x1
    x2 = transform.x2
    t = 0

    R, z = transform.convert_to_Rz(x1, x2, t)
    R_positions = np.sqrt(x_positions**2 + y_positions**2)
    assert np.array_equal(R, R_positions)


def test_convert_to_rho_theta():
    # TODO: Tricky, as it depends on the resolution of the equilibrium...
    transform, machine_dims = load_transect_default()
    equilibrium = load_equilibrium_default()
    transform.set_equilibrium(equilibrium)

    t = np.mean(equilibrium.rhop.t.mean())
    rho, theta = transform.convert_to_rho_theta(t=t)
    _rho = equilibrium.rhop.sel(t=t).interp(R=transform.R, z=transform.z)

    assert np.all(np.array(np.abs(rho - _rho)) < 1.0e-2)
