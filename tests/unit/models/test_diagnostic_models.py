import numpy as np

import indica.models.bolometer_camera as bolo
import indica.models.charge_exchange as cxrs
import indica.models.diode_filters as diodes
import indica.models.helike_spectroscopy as helike
import indica.models.interferometry as interf
from indica.models.plasma import example_run as example_plasma
import indica.models.thomson_scattering as ts

MODELS =  {"bolometer_camera":bolo,
           "diode_filters":diodes,
           "interferometry":interf,
           "helike_spectroscopy":helike,
           "thomson_scattering":ts,
           "charge_exchange":cxrs}
PLASMA = example_plasma()
NT = np.size(PLASMA.t)
TSTART = PLASMA.tstart
TEND = PLASMA.tend
DT = PLASMA.dt
TIME_SINGLE = PLASMA.t[int(NT/2.)] + DT/3.
TIME_INTERP = np.linspace(TSTART + DT, TEND - DT, num=int(NT/3))

def _test_examples(model_name:str, **kwargs):
    """ Test that example workflow runs without errors """
    model = MODELS[model_name]
    model.example_run(PLASMA, **kwargs)

def _test_timepoint(model_name:str, **kwargs):
    """ Test that model can be called for single time-point """
    model = MODELS[model_name]
    _, model, bckc = model.example_run(plasma=PLASMA, **kwargs)
    model(t=TIME_SINGLE)

def _test_time_interpolation(model_name:str, **kwargs):
    """ Test that model correctly interpolates data on new axis """
    model = MODELS[model_name]
    _, model, bckc = model.example_run(plasma=PLASMA, **kwargs)
    bckc = model(t=TIME_INTERP, **kwargs)

    for quantity, value in bckc.items():
        if "t" in value.dims:
            assert np.array_equal(value.t.values, TIME_INTERP)

def test_cxrs_example():
    _test_examples("charge_exchange")

def test_cxrs_timepoint():
    _test_timepoint("charge_exchange")

def test_cxrs_interpolation():
    _test_time_interpolation("charge_exchange")

def test_ts_example():
    _test_examples("thomson_scattering")

def test_ts_timepoint():
    _test_timepoint("thomson_scattering")

def test_ts_interpolation():
    _test_time_interpolation("thomson_scattering")

def test_bolo_example():
    _test_examples("bolometer_camera")

def test_bolo_timepoint():
    _test_timepoint("bolometer_camera")

def test_bolo_interpolation():
    _test_time_interpolation("bolometer_camera")

def test_interf_example():
    _test_examples("interferometry")

def test_interf_timepoint():
    _test_timepoint("interferometry")

def test_interf_interpolation():
    _test_time_interpolation("interferometry")

def test_diodes_example():
    _test_examples("diode_filters")

def test_diodes_timepoint():
    _test_timepoint("diode_filters")

def test_diodes_interpolation():
    _test_time_interpolation("diode_filters")

def test_helike_example():
    _test_examples("helike_spectroscopy")

def test_helike_timepoint():
    _test_timepoint("helike_spectroscopy")

def test_helike_interpolation():
    _test_time_interpolation("helike_spectroscopy")

def test_helike_full_example():
    _test_examples("helike_spectroscopy", calc_spectra=True)
#
def test_helike_full_timepoint():
    _test_timepoint("helike_spectroscopy", calc_spectra=True)

def test_helike_full_interpolation():
    _test_time_interpolation("helike_spectroscopy", calc_spectra=True)
