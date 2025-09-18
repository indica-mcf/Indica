from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates


def line_of_sight_example(make_plot=True):
    # Dummy line-of-sight
    machine_dims = ((0.15, 0.85), (-0.75, 0.75))
    origin_x = np.array([1.0, 1.0, 1.0], dtype=float)
    origin_y = np.array([0.0, 0.0, 0.0], dtype=float)
    origin_z = np.array([0.0, 0.0, 0.0], dtype=float)
    direction_x = np.array([-0.8, -0.8, -0.8], dtype=float)
    direction_y = np.array([0.4, 0.1, 0.0], dtype=float)
    direction_z = np.array([0.0, 0.0, 0.0], dtype=float)
    name = "dummy_los"

    # Optional inputs for the spot
    focal_length = -0.5  # meter
    spot_width = 0.010  # meter
    spot_height = 0.015  # meter
    spot_shape = "round"
    beamlets_method = "adaptive"
    n_beamlets = 25
    plot_beamlets = True

    los_transform = LineOfSightTransform(
        origin_x,
        origin_y,
        origin_z,
        direction_x,
        direction_y,
        direction_z,
        name=name,
        dl=0.01,
        spot_width=spot_width,
        spot_height=spot_height,
        spot_shape=spot_shape,
        beamlets_method=beamlets_method,
        n_beamlets=n_beamlets,
        focal_length=focal_length,
        machine_dimensions=machine_dims,
        passes=1,
        plot_beamlets=plot_beamlets,
    )

    if make_plot:
        # Plotting...
        cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(los_transform.x1), dtype=float))

        plt.figure()
        th = np.linspace(0, 2 * np.pi, 1000)
        x_ivc = machine_dims[0][1] * np.cos(th)
        y_ivc = machine_dims[0][1] * np.sin(th)
        x_cc = machine_dims[0][0] * np.cos(th)
        y_cc = machine_dims[0][0] * np.sin(th)

        plt.plot(x_cc, y_cc, c="k", lw=2.0)
        plt.plot(x_ivc, y_ivc, c="k", lw=2.0)
        for x1 in los_transform.x1:
            for beamlet in los_transform.beamlets:
                x = los_transform.x.sel(channel=x1, beamlet=beamlet)
                y = los_transform.y.sel(channel=x1, beamlet=beamlet)

                plt.plot(x, y, c=cols[x1])

        plt.tight_layout()

        plt.figure()

        plt.plot(
            [machine_dims[0][1], machine_dims[0][1]],
            [machine_dims[1][0], machine_dims[1][1]],
            c="k",
            lw=2.0,
        )

        plt.plot(
            [machine_dims[0][0], machine_dims[0][0]],
            [machine_dims[1][0], machine_dims[1][1]],
            c="k",
            lw=2.0,
        )

        for x1 in los_transform.x1:
            for beamlet in los_transform.beamlets:
                R = los_transform.R.sel(channel=x1, beamlet=beamlet)
                z = los_transform.z.sel(channel=x1, beamlet=beamlet)

                plt.plot(R, z, c=cols[x1])

        plt.tight_layout()
        plt.show(block=True)

    return los_transform


def cxrs_transform_example(nchannels: int):
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)

    transect_transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        "pi",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
    )
    return transect_transform


def helike_transform_example(nchannels):
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.2, -0.5, nchannels)
    los_start = np.array([[0.9, 0, 0]] * los_end.shape[0])
    los_start[:, 2] = -0.1
    origin = los_start
    direction = los_end - los_start

    los_transform = LineOfSightTransform(
        origin[0:nchannels, 0],
        origin[0:nchannels, 1],
        origin[0:nchannels, 2],
        direction[0:nchannels, 0],
        direction[0:nchannels, 1],
        direction[0:nchannels, 2],
        name="xrcs",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        passes=1,
    )
    return los_transform


def interferometer_transform_example(nchannels):
    los_start = np.array([[0.8, 0, 0]]) * np.ones((nchannels, 3))
    los_start[:, 2] = np.linspace(0, -0.2, nchannels)
    los_end = np.array([[0.17, 0, 0]]) * np.ones((nchannels, 3))
    los_end[:, 2] = np.linspace(0, -0.2, nchannels)
    origin = los_start
    direction = los_end - los_start
    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name="smmh1",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        passes=2,
    )
    return los_transform


def ts_transform_example(nchannels):
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)
    transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        "ts",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
    )
    return transform
