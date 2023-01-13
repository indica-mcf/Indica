import numpy as np

import indica.models.bolometer_camera as bolo
import indica.models.charge_exchange as cxrs
import indica.models.diode_filters as diodes
import indica.models.helike_spectroscopy as helike
import indica.models.interferometry as interf
from indica.models.plasma import example_run as example_plasma
import indica.models.thomson_scattering as ts

MODELS = {
    "bolometer_camera": bolo,
    "diode_filters": diodes,
    "interferometry": interf,
    "helike_spectroscopy": helike,
    "thomson_scattering": ts,
    "charge_exchange": cxrs,
}
PLASMA = example_plasma(pulse=None, tstart=0, tend=0.1, dt=0.02)
NT = np.size(PLASMA.t)
TSTART = PLASMA.tstart
TEND = PLASMA.tend
DT = PLASMA.dt
IT = int(NT / 2.0)
TIME_SINGLE_PASS = float(PLASMA.t[IT].values)
TIME_SINGLE_FAIL = float(np.max(PLASMA.equilibrium.rho.t) + 1.0)
TIME_INTERP = np.linspace(TSTART + DT, TEND - DT, num=int(NT / 3))


def _test_timepoint_pass(model_name: str, **kwargs):
    """Test that model can be called for single time-point"""
    model = MODELS[model_name]
    _, model, bckc = model.example_run(plasma=PLASMA, **kwargs)
    model(t=TIME_SINGLE_PASS)


def _test_timepoint_fail(model_name: str, **kwargs):
    """Test that model can be called for single time-point
    TODO: use pytes/unittest assertions to catch ValueError"""
    model = MODELS[model_name]
    _, model, bckc = model.example_run(plasma=PLASMA, **kwargs)
    try:
        model(t=TIME_SINGLE_FAIL)
    except ValueError:
        return


def _test_time_interpolation(model_name: str, **kwargs):
    """Test that model correctly interpolates data on new axis"""
    model = MODELS[model_name]
    _, model, bckc = model.example_run(plasma=PLASMA, **kwargs)
    bckc = model(t=TIME_INTERP, **kwargs)

    for quantity, value in bckc.items():
        if "t" in value.dims:
            assert np.array_equal(value.t.values, TIME_INTERP)


def test_cxrs_timepoint_fail():
    _test_timepoint_fail("charge_exchange")


def test_cxrs_timepoint_pass():
    _test_timepoint_pass("charge_exchange")


def test_cxrs_interpolation():
    _test_time_interpolation("charge_exchange")


def test_ts_timepoint_fail():
    _test_timepoint_fail("thomson_scattering")


def test_ts_timepoint_pass():
    _test_timepoint_pass("thomson_scattering")


def test_ts_interpolation():
    _test_time_interpolation("thomson_scattering")


def test_bolo_timepoint_fail():
    _test_timepoint_fail("bolometer_camera")


def test_bolo_timepoint_pass():
    _test_timepoint_pass("bolometer_camera")


def test_bolo_interpolation():
    _test_time_interpolation("bolometer_camera")


def test_interf_timepoint_fail():
    _test_timepoint_fail("interferometry")


def test_interf_timepoint_pass():
    _test_timepoint_pass("interferometry")


def test_interf_interpolation():
    _test_time_interpolation("interferometry")


def test_diodes_timepoint_fail():
    _test_timepoint_fail("diode_filters")


def test_diodes_timepoint_pass():
    _test_timepoint_pass("diode_filters")


def test_diodes_interpolation():
    _test_time_interpolation("diode_filters")


def test_helike_timepoint_fail():
    _test_timepoint_fail("helike_spectroscopy")


def test_helike_timepoint_pass():
    _test_timepoint_pass("helike_spectroscopy")


def test_helike_interpolation():
    _test_time_interpolation("helike_spectroscopy")


def test_helike_full_timepoint_fail():
    _test_timepoint_fail("helike_spectroscopy", calc_spectra=True)


def test_helike_full_timepoint_pass():
    _test_timepoint_pass("helike_spectroscopy", calc_spectra=True)


def test_helike_full_interpolation():
    _test_time_interpolation("helike_spectroscopy", calc_spectra=True)
