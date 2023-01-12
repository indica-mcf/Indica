import numpy as np
import xarray as xr

import indica.models.helike_spectroscopy as helike
from indica.models.plasma import example_run as example_plasma
from indica.converters.line_of_sight import LineOfSightTransform

# Setup LOS
def helike_LOS_example(nchannels=3):
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.43, -0.43, nchannels)
    los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start

    if nchannels > 1:
        los_transform = LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            name="diagnostic_name",
            machine_dimensions=plasma.machine_dimensions,
            passes=1,
        )
    elif nchannels == 1:
        los_transform = LineOfSightTransform(
        origin[0:1, 0],
        origin[0:1, 1],
        origin[0:1, 2],
        direction[0:1, 0],
        direction[0:1, 1],
        direction[0:1, 2],
        name="diagnostic_name",
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    else:
        raise ValueError(f"nchannels: {nchannels} not >= 1")
    return los_transform

plasma = example_plasma()
single_time_point = plasma.time_to_calculate[0]
multiple_time_point = plasma.time_to_calculate

multiple_channel_los_transform = helike_LOS_example(nchannels=3)
single_channel_los_transform = helike_LOS_example(nchannels=1)

single_channel_los_transform.set_equilibrium(plasma.equilibrium)
multiple_channel_los_transform.set_equilibrium(plasma.equilibrium)

def test_plasma_time_to_calculate_is_longer_than_one():
    assert plasma.time_to_calculate.__len__() > 1

def test_helike_runs_with_example_plasma_and_multiple_LOS():
    model = helike.Helike_spectroscopy("diagnostic_name", )
    model.set_plasma(plasma)
    model.set_los_transform(multiple_channel_los_transform)
    bckc = model(calc_spectra=False)
    assert bckc

def test_helike_runs_with_example_plasma_and_single_LOS_and_multiple_time_point():
    model = helike.Helike_spectroscopy("diagnostic_name", )
    model.set_plasma(plasma)
    plasma.time_to_calculate = multiple_time_point
    model.set_los_transform(single_channel_los_transform)
    bckc = model(calc_spectra=False)
    assert bckc

def test_helike_runs_with_example_plasma_and_single_LOS_and_single_time_point():
    model = helike.Helike_spectroscopy("diagnostic_name", )
    model.set_plasma(plasma)
    plasma.time_to_calculate = single_time_point
    model.set_los_transform(single_channel_los_transform)
    bckc = model(calc_spectra=False)
    assert bckc

print()