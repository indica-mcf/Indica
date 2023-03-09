import numpy as np

from indica.converters.line_of_sight import LineOfSightTransform
import indica.models.helike_spectroscopy as helike
from indica.models.plasma import example_run as example_plasma


# Setup LOS
def helike_LOS_example(nchannels=3):
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.43, -0.43, nchannels)
    los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start

    los_transform = LineOfSightTransform(
        origin[0:nchannels, 0],
        origin[0:nchannels, 1],
        origin[0:nchannels, 2],
        direction[0:nchannels, 0],
        direction[0:nchannels, 1],
        direction[0:nchannels, 2],
        name="diagnostic_name",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        passes=1,
    )
    return los_transform


class TestHelike:
    def setup_class(self):
        self.plasma = example_plasma(pulse=9229)
        self.single_time_point = self.plasma.time_to_calculate[1]
        self.multiple_time_point = self.plasma.time_to_calculate
        self.multiple_channel_los_transform = helike_LOS_example(nchannels=3)
        self.single_channel_los_transform = helike_LOS_example(nchannels=1)
        self.single_channel_los_transform.set_equilibrium(self.plasma.equilibrium)
        self.multiple_channel_los_transform.set_equilibrium(self.plasma.equilibrium)

    def test_plasma_time_to_calculate_is_longer_than_one(self):
        assert self.plasma.time_to_calculate.__len__() > 1

    def test_helike_runs_with_example_plasma_and_multiple_LOS(self):
        model = helike.Helike_spectroscopy(
            "diagnostic_name",
        )
        self.plasma.time_to_calculate = self.multiple_time_point
        model.set_plasma(self.plasma)
        model.set_los_transform(self.multiple_channel_los_transform)
        bckc = model(calc_spectra=False)
        assert bckc

    def test_helike_runs_with_example_plasma_and_single_LOS_and_multiple_time_point(
        self,
    ):
        model = helike.Helike_spectroscopy(
            "diagnostic_name",
        )
        model.set_plasma(self.plasma)
        self.plasma.time_to_calculate = self.multiple_time_point
        model.set_los_transform(self.single_channel_los_transform)
        bckc = model(calc_spectra=False)
        assert bckc

    def test_helike_runs_with_example_plasma_and_single_LOS_and_single_time_point(self):
        model = helike.Helike_spectroscopy(
            "diagnostic_name",
        )
        model.set_plasma(self.plasma)
        self.plasma.time_to_calculate = self.single_time_point
        model.set_los_transform(self.single_channel_los_transform)
        bckc = model(calc_spectra=False)
        assert bckc
