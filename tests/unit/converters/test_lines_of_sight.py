"""Tests for line-of-sight coordinate transforms."""

import sys
sys.path.insert(0, "/home/jonathan.wood/git_home/Indica")
sys.path.insert(0, "../")
sys.path.remove("/home/marco.sertoli/python/Indica")
from indica.converters import lines_of_sight
from indica.converters import flux_surfaces
from indica import equilibrium
from test_equilibrium_single import equilibrium_dat_and_te
from unittest.mock import MagicMock
from xarray import DataArray

import numpy as np
from matplotlib import pyplot as plt


def convert_to_rho(plot=False):
    # Line of sight origin tuple
    origin = (3.8, -2.0, 0.5)  # [xyz]

    # Line of sight direction
    direction = (-1.0, 0.0, 0.0)  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Equilibrium
    data, Te = equilibrium_dat_and_te()
    offset = MagicMock(side_effect=[(0.02, False), (0.02, True)])
    equil = equilibrium.Equilibrium(
        data,
        Te,
        sess=MagicMock(),
        offset_picker=offset,
    )

    # Flux Transform
    flux_coord = flux_surfaces.FluxSurfaceCoordinates("poloidal")
    flux_coord.set_equilibrium(equil)

    # Set-up line of sight class
    los = lines_of_sight.LinesOfSightTransform(
        origin[0], origin[1], origin[2], direction[0], direction[1], direction[2],
        machine_dimensions=machine_dims, name=name
    )

    # Assign flux transform
    los.assign_flux_transform(flux_coord)

    # Convert_to_rho method
    los.convert_to_rho()

    if plot:
        # centre column
        th = np.linspace(0.0, 2 * np.pi, 1000)
        x_cc = machine_dims[0][0] * np.cos(th)
        y_cc = machine_dims[0][0] * np.sin(th)

        # IVC
        x_ivc = machine_dims[0][1] * np.cos(th)
        y_ivc = machine_dims[0][1] * np.sin(th)

        plt.figure()
        plt.plot(los.x2, los.rho[0].sel(t=77.0, method='nearest'), 'b')
        plt.ylabel('rho')

        plt.figure()
        plt.plot(x_cc, y_cc, 'k--')
        plt.plot(x_ivc, y_ivc, 'k--')
        plt.plot(los.x_start, los.y_start, 'ro', label='start')
        plt.plot(los.x_end, los.y_end, 'bo', label='end')
        plt.plot(los.x, los.y, 'g', label='los')
        plt.legend()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show(block=True)

    return


def test_methods(debug=False):
    # Line of sight origin tuple
    origin = (3.8, -2.0, 0.5)  # [xyz]

    # Line of sight direction
    direction = (-1.0, 0.0, 0.0)  # [xyz]

    # machine dimensions
    machine_dims = ((1.83, 3.9), (-1.75, 2.0))

    # name
    name = "los_test"

    # Set-up line of sight class
    los = lines_of_sight.LinesOfSightTransform(
        origin[0], origin[1], origin[2], direction[0], direction[1], direction[2],
        machine_dimensions=machine_dims, name=name
    )

    # Inputs for testing methods...
    R_test = DataArray(2.5)  # Does not work as an array
    Z_test = DataArray(0.5)  # Does not work as an array
    x1 = 0.0  # does nothing
    x2 = DataArray(np.linspace(0.0, 1.0, 350, dtype=float))  # index along line of sight, must be a DataArray
    t = 0.0  # does nothing

    # Test method #1
    r_, z_ = los.convert_to_Rz(x1, x2, t)
    if debug:
        print(f'r_ = {r_}')
        print(f'z_ = {z_}')

    # Check method #2: convert_from_Rz, inputs: "R", "Z", "t"
    _, x2_out2 = los.convert_from_Rz(R_test, Z_test, t)
    if debug:
        print(f'x2_out2 = {x2_out2}')

    # Check method #3: distance, inputs: "x1", "x2", "t"
    dist = los.distance('dim_0', x1, x2, t)
    if debug:
        print(f'dist = {dist}')

    return
