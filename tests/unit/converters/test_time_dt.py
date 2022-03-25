from copy import deepcopy

import numpy as np
from pytest import approx
from xarray import DataArray

from indica.converters.time import convert_in_time_dt


class Test_time:
    """Provides unit tests for the time converter"""

    nt = 50
    values = np.sin(np.linspace(0, np.pi * 3, nt)) + np.random.random(nt) - 0.5
    time = np.linspace(0, 0.1, nt)
    data = DataArray(values, coords=[("t", time)])
    error = deepcopy(data)
    error.values = np.sqrt(np.abs(values))
    data.attrs = {"error": error}

    dt_data = (data.t[1] - data.t[0]).values

    def test_identity(self):
        """Checks interpolation works as expected and returned data is withing limits"""
        dt = self.dt_data * 1.0

        tstart = (self.data.t[0] + 5 * self.dt_data).values
        tend = (self.data.t[-1] - 10 * self.dt_data).values

        try:
            _data = convert_in_time_dt(tstart, tend, dt, self.data)
        except Exception as e:
            raise e

        for t in _data.t:
            delta = np.min(np.abs(t - self.data.t))
            try:
                assert approx(delta) == 0
            except AssertionError as e:
                print("Original and new time axis aren't identical")
                raise e

            delta = np.min(
                np.abs(_data.sel(t=t) - self.data.sel(t=t, method="nearest"))
            )
            try:
                assert approx(delta) == 0
            except AssertionError as e:
                print("Original and new data aren't identical")
                raise e

        for t in _data.error.t:
            delta = np.min(np.abs(t - self.data.error.t))
            try:
                assert approx(delta) == 0
            except AssertionError as e:
                print("Original and new time axis of error aren't identical")
                raise e

            delta = np.min(
                np.abs(
                    _data.error.sel(t=t) - self.data.error.sel(t=t, method="nearest")
                )
            )
            try:
                assert approx(delta) == 0
            except AssertionError as e:
                print("Original and new error aren't identical")
                raise e

    def test_binning(self):
        """Checks interpolation works as expected and returned data is withing limits"""
        dt = self.dt_data * 3.0

        tstart = (self.data.t[0] + 5 * self.dt_data).values
        tend = (self.data.t[-1] - 10 * self.dt_data).values

        try:
            _data = convert_in_time_dt(tstart, tend, dt, self.data)
        except Exception as e:
            raise e

        _dt = (_data.t[1] - _data.t[0]).values

        assert np.all(_data.t <= self.data.t.max())
        assert np.all(_data.t >= self.data.t.min())
        assert _dt == approx(dt)

        _dt = (_data.error.t[1] - _data.error.t[0]).values

        assert np.all(_data.error.t <= self.data.error.t.max())
        assert np.all(_data.error.t >= self.data.error.t.min())
        assert _dt == approx(dt)

    def test_interpolation(self):
        """Checks interpolation works as expected and returned data is withing limits"""
        dt = self.dt_data / 3.0

        tstart = (self.data.t[0] + 5 * self.dt_data).values
        tend = (self.data.t[-1] - 10 * self.dt_data).values

        try:
            _data = convert_in_time_dt(tstart, tend, dt, self.data)
        except Exception as e:
            raise e

        _dt = (_data.t[1] - _data.t[0]).values

        assert np.all(_data.t <= self.data.t.max())
        assert np.all(_data.t >= self.data.t.min())
        assert _dt == approx(dt)

        _dt = (_data.error.t[1] - _data.error.t[0]).values

        assert np.all(_data.error.t <= self.data.error.t.max())
        assert np.all(_data.error.t >= self.data.error.t.min())
        assert _dt == approx(dt)

    def test_wrong_start_time(self):
        """Checks interpolation works as expected and returned data is withing limits"""
        dt = self.dt_data

        tstart = (self.data.t[0] - 5 * self.dt_data).values
        tend = (self.data.t[-1] - 10 * self.dt_data).values

        try:
            _ = convert_in_time_dt(tstart, tend, dt, self.data)
        except ValueError as e:
            assert e

    def test_wrong_end_time(self):
        """Checks interpolation works as expected and returned data is withing limits"""
        dt = self.dt_data

        tstart = (self.data.t[0] + 5 * self.dt_data).values
        tend = (self.data.t[-1] + 10 * self.dt_data).values

        try:
            _ = convert_in_time_dt(tstart, tend, dt, self.data)
        except ValueError as e:
            assert e
