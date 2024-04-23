from indica.defaults.read_write_defaults import load_default_objects
import indica.models.helike_spectroscopy as helike
from indica.models.helike_spectroscopy import helike_transform_example


class TestHelike:
    def setup_class(self):
        machine = "st40"
        equilibrium = load_default_objects(machine, "equilibrium")
        _plasma = load_default_objects(machine, "plasma")
        _plasma.set_equilibrium(equilibrium)

        self.plasma = _plasma
        self.single_time_point = self.plasma.time_to_calculate[1]
        self.multiple_time_point = self.plasma.time_to_calculate
        self.multiple_channel_los_transform = helike_transform_example(nchannels=3)
        self.single_channel_los_transform = helike_transform_example(nchannels=1)
        self.single_channel_los_transform.set_equilibrium(self.plasma.equilibrium)
        self.multiple_channel_los_transform.set_equilibrium(self.plasma.equilibrium)

    def setup_method(self):
        self.model = helike.HelikeSpectrometer("diagnostic_name")
        self.model.set_plasma(self.plasma)

    def test_helike_moment_runs_with_multiple_LOS(
        self,
    ):
        self.model.set_los_transform(self.multiple_channel_los_transform)
        bckc = self.model(calc_spectra=False, moment_analysis=True)
        assert bckc

    def test_helike_moment_runs_with_single_LOS(
        self,
    ):
        self.model.set_los_transform(self.single_channel_los_transform)
        bckc = self.model(calc_spectra=False, moment_analysis=True)
        assert bckc

    def test_helike_spectra_with_multiple_LOS(
        self,
    ):
        self.model.set_los_transform(self.multiple_channel_los_transform)
        bckc = self.model(calc_spectra=True, moment_analysis=False)
        assert bckc

    def test_helike_spectra_with_single_LOS(
        self,
    ):
        self.model.set_los_transform(self.single_channel_los_transform)
        bckc = self.model(calc_spectra=True, moment_analysis=False)
        assert bckc
