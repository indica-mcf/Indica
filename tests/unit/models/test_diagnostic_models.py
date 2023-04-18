from typing import Callable
from typing import Dict

import numpy as np
import pytest

from indica.models.bolometer_camera import example_run as bolo
from indica.models.charge_exchange import example_run as cxrs
from indica.models.diode_filters import example_run as diodes
from indica.models.equilibrium_reconstruction import example_run as equil_recon
from indica.models.helike_spectroscopy import example_run as helike
from indica.models.interferometry import example_run as interf
from indica.models.plasma import example_run as example_plasma
from indica.models.thomson_scattering import example_run as ts


EXAMPLES: Dict[str, Callable] = {
    "bolometer_camera": bolo,
    "diode_filters": diodes,
    "interferometry": interf,
    "helike_spectroscopy": helike,
    "thomson_scattering": ts,
    "charge_exchange": cxrs,
    "equilibrium_reconstruction": equil_recon,
}


class TestModels:
    """Test that model calls run without error"""

    def setup_class(self):
        self.plasma = example_plasma(calc_power_loss=True)
        nt = np.size(self.plasma.t)
        tstart = self.plasma.tstart
        tend = self.plasma.tend
        dt = self.plasma.dt
        it = int(nt / 2.0)
        self.time_single_pass = float(self.plasma.t[it].values)
        self.time_single_fail = float(np.max(self.plasma.equilibrium.rho.t) + 1.0)
        self.time_interp = np.linspace(tstart + dt, tend - dt, num=int(nt / 3))

    def _test_timepoint_pass(self, model_name: str, **kwargs):
        """Test that model can be called for single time-point"""
        _, model, bckc = EXAMPLES[model_name](plasma=self.plasma, **kwargs)
        model(t=self.time_single_pass)

    def _test_timepoint_fail(self, model_name: str, **kwargs):
        """Test that model can be called for single time-point"""
        _, model, bckc = EXAMPLES[model_name](plasma=self.plasma, **kwargs)
        with pytest.raises(Exception):
            model(t=self.time_single_fail)

    def _test_time_interpolation(self, model_name: str, **kwargs):
        """Test that model correctly interpolates data on new axis"""
        _, model, bckc = EXAMPLES[model_name](plasma=self.plasma, **kwargs)
        bckc = model(t=self.time_interp, **kwargs)

        for quantity, value in bckc.items():
            if "t" in value.dims:
                assert np.array_equal(value.t.values, self.time_interp)

    def test_cxrs_timepoint_fail(self):
        self._test_timepoint_fail("charge_exchange")

    def test_cxrs_timepoint_pass(self):
        self._test_timepoint_pass("charge_exchange")

    def test_cxrs_interpolation(self):
        self._test_time_interpolation("charge_exchange")

    def test_ts_timepoint_fail(self):
        self._test_timepoint_fail("thomson_scattering")

    def test_ts_timepoint_pass(self):
        self._test_timepoint_pass("thomson_scattering")

    def test_ts_interpolation(self):
        self._test_time_interpolation("thomson_scattering")

    def test_bolo_timepoint_fail(self):
        self._test_timepoint_fail("bolometer_camera")

    def test_bolo_timepoint_pass(self):
        self._test_timepoint_pass("bolometer_camera")

    def test_bolo_interpolation(self):
        self._test_time_interpolation("bolometer_camera")

    def test_interf_timepoint_fail(self):
        self._test_timepoint_fail("interferometry")

    def test_interf_timepoint_pass(self):
        self._test_timepoint_pass("interferometry")

    def test_interf_interpolation(self):
        self._test_time_interpolation("interferometry")

    def test_diodes_timepoint_fail(self):
        self._test_timepoint_fail("diode_filters")

    def test_diodes_timepoint_pass(self):
        self._test_timepoint_pass("diode_filters")

    def test_diodes_interpolation(self):
        self._test_time_interpolation("diode_filters")

    def test_helike_timepoint_fail(self):
        self._test_timepoint_fail("helike_spectroscopy")

    def test_helike_timepoint_pass(self):
        self._test_timepoint_pass("helike_spectroscopy")

    def test_helike_interpolation(self):
        self._test_time_interpolation("helike_spectroscopy")

    def test_equil_recon_timepoint_fail(self):
        self._test_timepoint_fail("equilibrium_reconstruction")

    def test_equil_recon_timepoint_pass(self):
        self._test_timepoint_pass("equilibrium_reconstruction")

    def test_equil_recon_interpolation(self):
        self._test_time_interpolation("equilibrium_reconstruction")
