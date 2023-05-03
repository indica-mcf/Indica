import indica.models.helike_spectroscopy as helike
from indica.models.helike_spectroscopy import helike_LOS_example
from indica.models.plasma import example_run as example_plasma
import pytest

class TestHelike:
    def setup_class(self):
        self.plasma = example_plasma()  # using Phantom
        self.single_time_point = self.plasma.time_to_calculate[1]
        self.multiple_time_point = self.plasma.time_to_calculate
        self.multiple_channel_los_transform = helike_LOS_example(nchannels=3)
        self.single_channel_los_transform = helike_LOS_example(nchannels=1)
        self.single_channel_los_transform.set_equilibrium(self.plasma.equilibrium)
        self.multiple_channel_los_transform.set_equilibrium(self.plasma.equilibrium)

    def setup_method(self):
        self.model = helike.Helike_spectroscopy("diagnostic_name")
        self.model.set_plasma(self.plasma)

    def test_helike_moment_runs_with_multiple_LOS(self, ):
        self.model.set_los_transform(self.multiple_channel_los_transform)
        bckc = self.model(calc_spectra=False, moment_analysis=True)
        assert bckc

    def test_helike_moment_runs_with_single_LOS(self, ):
        self.model.set_los_transform(self.single_channel_los_transform)
        bckc = self.model(calc_spectra=False, moment_analysis=True)
        assert bckc

    def test_helike_spectra_with_multiple_LOS(self, ):
        self.model.set_los_transform(self.multiple_channel_los_transform)
        bckc = self.model(calc_spectra=True, moment_analysis=False)
        assert bckc

    def test_helike_spectra_with_single_LOS(self, ):
        self.model.set_los_transform(self.single_channel_los_transform)
        bckc = self.model(calc_spectra=True, moment_analysis=False)
        assert bckc
