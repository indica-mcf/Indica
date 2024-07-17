import numpy as np
from indica.converters import TransectCoordinates, LineOfSightTransform

def pi_transform_example(nchannels: int):
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

def smmh1_transform_example(nchannels):
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
