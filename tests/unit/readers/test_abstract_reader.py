"""Test methods for the abstracreader class"""

from numbers import Number
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set

import numpy as np
import pytest
from xarray import DataArray

from indica import session
from indica.numpy_typing import RevisionLike
from indica.readers import DataReader
from indica.readers.abstractreader import DataSelector
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


# TODO these values should come from the machine dimensions variable of the reader

_MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
_INSTRUMENT_METHODS = {
    "thomson_scattering": "get_thomson_scattering",
    "equilibrium": "get_equilibrium",
    "cyclotron_emissions": "get_cyclotron_emissions",
    "charge_exchange": "get_charge_exchange",
    "bremsstrahlung_spectroscopy": "get_bremsstrahlung_spectroscopy",
    "radiation": "get_radiation",
    "helike_spectroscopy": "get_helike_spectroscopy",
    "interferometry": "get_interferometry",
    "filters": "get_filters",
}


def gen_array(_min: float, _max: float, shape: tuple, to_float=False):

    _values = list(np.linspace(_min, _max, shape[-1]))
    if len(shape) > 1:
        _values = [_values] * shape[-2]
    if len(shape) > 2:
        _values = [_values] * shape[-3]

    values = np.array(_values)

    if len(shape) == 1 and shape[0] == 1 and to_float:
        values = values[0]

    return values


def selector(
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number],
    unselected_channels: Iterable[Number] = [],
):
    return bad_channels


class TestReader(DataReader):
    """Class to read fake data"""

    MACHINE_DIMS = _MACHINE_DIMS
    INSTRUMENT_METHODS = _INSTRUMENT_METHODS

    def __init__(
        self,
        tstart: float,
        tend: float,
        max_freq: float = 1e6,
        selector: DataSelector = selector,
        sess: session.Session = session.global_session,
        empty=False,
        equil_unique=True,
    ):
        """
        Test version of DataReader

        Parameters
        ----------
        tstart
            ...as in abstractreader...
        tend
            ...as in abstractreader...
        max_freq
            ...as in abstractreader...
        selector
            Fake version of selector
        sess
            ...as in abstractreader...
        empty
            Set to True to return empty dictionary
        equil_unique
            Set to False to return equilibrium time axis with non-unique elements
        """
        self._reader_cache_id = ""
        super().__init__(
            tstart,
            tend,
            max_freq,
            sess,
            selector,
        )
        self._client = ""

        self.empty = empty
        self.equil_unique = equil_unique

        self.Rmin, self.Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        self.zmin, self.zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        self.length = 10
        self.dt = 0.01
        self.nwavelength = 256
        self.wavelength_start = 0.3
        self.wavelength_end = 0.3

        self.tstart = tstart
        self.tend = tend
        self.times = np.arange(self.tstart, self.tend, self.dt)
        self.nt = len(self.times)
        self.nrho = 35
        self.nr = 20
        self.nz = 40

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": self.length,
            "machine_dims": self.MACHINE_DIMS,
        }
        results["times"] = self.times
        results["texp"] = np.full_like(self.times, self.nt)
        results["element"] = "element"
        results["revision"] = revision
        results["R"] = gen_array(self.Rmin, self.Rmax, (results["length"],))
        results["z"] = gen_array(self.zmin, self.zmax, (results["length"],))
        results["ti"] = gen_array(10, 10.0e3, (self.nt, results["length"]))
        results["ti_error"] = np.sqrt(results["ti"])
        results["angf"] = gen_array(1.0e2, 1.0e6, (self.nt, results["length"]))
        results["angf_error"] = np.sqrt(results["angf"])
        results["conc"] = gen_array(1.0e-6, 1.0e-1, (self.nt, results["length"]))
        results["conc_error"] = np.sqrt(results["conc"])
        results["bad_channels"] = []

        for quantity in quantities:
            results[f"{quantity}_records"] = [
                f"{quantity}_R_path",
                f"{quantity}_z_path",
                f"{quantity}_element_path",
                f"{quantity}_time_path",
                f"{quantity}_value_path",
                f"{quantity}_error_path",
            ]

        return results

    def _get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": self.length,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["times"] = self.times
        results["revision"] = revision
        results["z"] = gen_array(self.zmin, self.zmax, (results["length"],))
        results["R"] = gen_array(self.Rmin, self.Rmax, (results["length"],))
        results["te"] = gen_array(10.0, 10.0e3, (self.nt, results["length"]))
        results["ne"] = gen_array(1.0e16, 1.0e21, (self.nt, results["length"]))
        results["te_error"] = np.sqrt(results["te"])
        results["ne_error"] = np.sqrt(results["ne"])
        results["bad_channels"] = []

        for quantity in quantities:
            results[f"{quantity}_records"] = [
                f"{quantity}_Rz_path",
                f"{quantity}_value_path",
                f"{quantity}_error_path",
            ]

        return results

    def _get_equilibrium(
        self,
        uid: str,
        calculation: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {}

        results["times"] = self.times
        if not self.equil_unique:
            results["times"][-1] = results["times"][-2]

        results["element"] = "element"

        results["rgeo"] = gen_array(self.Rmin, self.Rmax, (self.nt,))
        results["rmag"] = gen_array(self.Rmin, self.Rmax, (self.nt,))
        results["zmag"] = gen_array(self.zmin, self.zmax, (self.nt,))
        results["ipla"] = gen_array(1.0e4, 1.0e6, (self.nt,))
        results["wp"] = gen_array(1.0e2, 1.0e5, (self.nt,))
        results["df"] = gen_array(0, 1, (self.nt,))
        results["faxs"] = gen_array(1.0e-6, 0.1, (self.nt,))
        results["fbnd"] = gen_array(-1, 1, (self.nt,))

        results["psin"] = gen_array(0, 1, (self.nrho,))
        results["psi_r"] = gen_array(self.Rmin, self.Rmax, (self.nr,))
        results["psi_z"] = gen_array(self.zmin, self.zmax, (self.nz,))

        results["f"] = gen_array(self.Rmin, self.Rmax, (self.nt, self.nrho))
        results["ftor"] = gen_array(1.0e-4, 1.0e-2, (self.nt, self.nrho))
        results["vjac"] = gen_array(1.0e-3, 2, (self.nt, self.nrho))
        results["rmji"] = gen_array(self.Rmin, self.Rmax, (self.nt, self.nrho))
        results["rmjo"] = gen_array(self.Rmin, self.Rmax, (self.nt, self.nrho))
        results["rbnd"] = gen_array(self.Rmin, self.Rmax, (self.nt, self.nrho))
        results["zbnd"] = gen_array(self.zmin, self.zmax, (self.nt, self.nrho))

        results["psi"] = gen_array(-1, 1, (self.nt, self.nz, self.nr))

        results["revision"] = revision

        for quantity in quantities:
            results[f"{quantity}_records"] = [f"{quantity}_records"]
        results["psi_records"] = ["value_records", "R_records", "z_records"]
        return results

    def _get_cyclotron_emissions(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": self.length,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["times"] = self.times

        results["revision"] = revision
        results["z"] = gen_array(self.zmin, self.zmax, (1,), to_float=True)

        results["te"] = gen_array(10, 10.0e3, (self.nt, results["length"]))
        results["te_error"] = np.sqrt(results["te"])
        results["Btot"] = gen_array(0.1, 5, (results["length"],))
        results["bad_channels"] = []

        results["te_records"] = ["info_path", "data_path"]

        return results

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = revision
        results["times"] = self.times

        results["v_times"] = self.times
        results["v"] = gen_array(0, 1.0e6, (self.nt, self.length))
        results["v_error"] = np.sqrt(results["v"])
        results["v_xstart"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["v_ystart"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["v_zstart"] = gen_array(self.zmin, self.zmax, (self.length,))
        results["v_xstop"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["v_ystop"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["v_zstop"] = gen_array(self.zmin, self.zmax, (self.length,))

        results["h_times"] = self.times
        results["h"] = gen_array(0, 1.0e6, (self.nt, self.length))
        results["h_error"] = np.sqrt(results["h"])
        results["h_xstart"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["h_ystart"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["h_zstart"] = gen_array(self.zmin, self.zmax, (self.length,))
        results["h_xstop"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["h_ystop"] = gen_array(-self.Rmax, self.Rmax, (self.length,))
        results["h_zstop"] = gen_array(self.zmin, self.zmax, (self.length,))

        results["length"]["v"] = int(self.length)
        results["length"]["h"] = int(self.length)

        for quantity in quantities:
            results[f"{quantity}_records"] = [f"{quantity}_records"] * self.length

        return results

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = revision
        results["times"] = self.times

        length = int(1)
        signals = ["zefv", "zefh"]
        for k in signals:
            results[k] = gen_array(0, 1.0e6, (self.nt,))
            results[f"{k}_error"] = np.sqrt(results[k])
            results[f"{k}_xstart"] = gen_array(self.Rmin, self.Rmax, (1,))
            results[f"{k}_ystart"] = gen_array(self.Rmin, self.Rmax, (1,))
            results[f"{k}_zstart"] = gen_array(self.zmin, self.zmax, (1,))
            results[f"{k}_xstop"] = gen_array(self.Rmin, self.Rmax, (1,))
            results[f"{k}_ystop"] = gen_array(self.Rmin, self.Rmax, (1,))
            results[f"{k}_zstop"] = gen_array(self.zmin, self.zmax, (1,))
            results["length"][k] = int(length)

        for quantity in quantities:
            results[f"{quantity}_records"] = [
                f"{quantity}_path_records",
                f"{quantity}_los_records",
            ]

        return results

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = revision
        results["times"] = self.times

        results["wavelength"] = gen_array(
            self.wavelength_start, self.wavelength_end, (self.nwavelength,)
        )

        results["xstart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstart"] = gen_array(self.zmin, self.zmax, (1,))
        results["xstop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstop"] = gen_array(self.zmin, self.zmax, (1,))
        for quantity in quantities:
            if quantity == "spectra":
                results[quantity] = gen_array(0, 1.0e6, (self.nt, self.nwavelength))
            else:
                results[quantity] = gen_array(0, 1.0e4, (self.nt,))
            results[f"{quantity}_error"] = np.sqrt(results[quantity])

            results[f"{quantity}_records"] = [
                f"{quantity}_path_records",
            ]

        return results

    def _get_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = revision
        results["times"] = self.times

        results["xstart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstart"] = gen_array(self.zmin, self.zmax, (1,))
        results["xstop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstop"] = gen_array(self.zmin, self.zmax, (1,))
        for quantity in quantities:
            results[quantity] = gen_array(0, 1.0e6, (self.nt,))
            results[f"{quantity}_error"] = np.sqrt(results[quantity])

            results[f"{quantity}_records"] = [
                f"{quantity}_path_records",
            ]
            results[f"{quantity}_error_records"] = [
                f"{quantity}_error_path_records",
            ]

        return results

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if self.empty:
            return {}

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = revision
        results["times"] = self.times

        results["xstart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystart"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstart"] = gen_array(self.zmin, self.zmax, (1,))
        results["xstop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["ystop"] = gen_array(self.Rmin, self.Rmax, (1,))
        results["zstop"] = gen_array(self.zmin, self.zmax, (1,))
        for quantity in quantities:
            results[quantity] = gen_array(0, 1.0e6, (self.nt,))
            results[f"{quantity}_error"] = np.sqrt(results[quantity])

            results[f"{quantity}_records"] = [
                f"{quantity}_path_records",
            ]
            results[f"{quantity}_error_records"] = [
                f"{quantity}_error_path_records",
            ]

        return results

    def _get_bad_channels(
        self, uid: str, instrument: str, quantity: str
    ) -> List[Number]:
        return []

    def _set_times_item(
        self,
        results: Dict[str, Any],
        times: np.ndarray,
    ):
        if "times" not in results:
            times = times

    def close(self):
        del self._client

    def requires_authentication(self):
        return True


class UnimplementedMethodReader(DataReader):
    """Class to test calls to unimplemented get methods"""

    MACHINE_DIMS = _MACHINE_DIMS
    INSTRUMENT_METHODS = _INSTRUMENT_METHODS

    def __init__(
        self,
        tstart: float,
        tend: float,
        max_freq: float = 1e6,
        selector: DataSelector = selector,
        sess: session.Session = session.global_session,
    ):
        """
        Test version of DataReader

        Parameters
        ----------
        tstart
            ...as in abstractreader...
        tend
            ...as in abstractreader...
        max_freq
            ...as in abstractreader...
        selector
            Fake version of selector
        sess
            ...as in abstractreader...
        empty
            Set to True to return empty dictionary
        equil_unique
            Set to False to return equilibrium time axis with non-unique elements
        """
        super().__init__(
            tstart,
            tend,
            max_freq,
            sess,
            selector,
        )

    def close(self):
        del self._client

    def requires_authentication(self):
        return True


def _test_get_methods(
    instrument: str,
    tstart: float = 0.0,
    tend: float = 1.0,
):
    """
    Generalised test for all get methods of the abstractreader
    """

    _get_method = f"_{_INSTRUMENT_METHODS[instrument]}"

    reader = TestReader(
        tstart,
        tend,
    )

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    _results = getattr(reader, _get_method)("", instrument, 0, quantities)

    results = reader.get("", instrument, 0, quantities)

    for q, actual, expected in [(q, results[q], _results[q]) for q in quantities]:
        try:
            assert np.all(actual.values == expected)
        except AssertionError:
            return actual, expected


def _test_catch_unimplemented_reader(
    instrument: str,
    tstart: float = 0.0,
    tend: float = 1.0,
):
    """
    Test catch for unimplemented get methods in abstract reader
    """

    reader = UnimplementedMethodReader(
        tstart,
        tend,
    )

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    with pytest.raises(NotImplementedError):
        reader.get("", instrument, 0, quantities)


def _test_empty(
    instrument,
    tstart=0.0,
    tend=1.0,
):
    """
    Test for no data returned by reader _get_methods
    """

    reader = TestReader(
        tstart,
        tend,
        empty=True,
    )

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    results = reader.get("", instrument, 0, quantities)

    assert np.all(len(results) == 0)


def _test_invalid_quantity(
    instrument: str,
    tstart=0.0,
    tend=1.0,
):
    """
    Test raising ValueError when fetching unavailable quantity
    """

    reader = TestReader(tstart, tend)

    with pytest.raises(ValueError):
        reader.get("", instrument, 0, {"invalid"})


def _test_caching(instrument: str, tstart: float = 0.0, tend: float = 1.0):
    """
    Generalised test for all get methods of the abstractreader
    """

    _get_method = f"_{_INSTRUMENT_METHODS[instrument]}"

    reader = TestReader(
        tstart,
        tend,
    )

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    _results = getattr(reader, _get_method)("", instrument, 0, quantities)

    results = reader.get("", instrument, 0, quantities)

    for q, actual, expected in [(q, results[q], _results[q]) for q in quantities]:
        try:
            assert np.all(actual.values == expected)
        except AssertionError:
            return actual, expected


def _test_downsample_ratio(instrument: str, tstart: float = 0.0, tend: float = 1.0):
    _get_method = f"_{_INSTRUMENT_METHODS[instrument]}"

    reader = TestReader(tstart, tend)
    reader._max_freq = (1 / reader.dt) / 10

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    _results = getattr(reader, _get_method)("", instrument, 0, quantities)

    results = reader.get("", instrument, 0, quantities)

    for q, actual in [(q, results[q]) for q in quantities]:
        frequency = 1 / (actual.coords["t"][1:].values - actual.coords["t"][:-1].values)
        assert np.all(frequency - reader._max_freq < 1e-6)  # reader._max_freq / 1000)
        assert actual.coords["t"].size < _results["times"].size


def test_non_unique_times(
    instrument: str = "equilibrium", tstart: float = 0.0, tend: float = 1.0
):
    _get_method = f"_{_INSTRUMENT_METHODS[instrument]}"

    reader = TestReader(
        tstart,
        tend,
        equil_unique=False,
    )

    quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

    _results = getattr(reader, _get_method)("", instrument, 0, quantities)

    results = reader.get("", instrument, 0, quantities)

    for q, actual, expected in [(q, results[q], _results[q]) for q in quantities]:
        if "t" in results[q].dims:
            assert np.all(results[q].t.values == np.unique(_results["times"]))


def test_thomson_scattering():
    instrument = "thomson_scattering"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_equilibrium():
    instrument = "equilibrium"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_cyclotron_emissions():
    instrument = "cyclotron_emissions"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_charge_exchange():
    instrument = "charge_exchange"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_bremsstrahlung_spectroscopy():
    instrument = "bremsstrahlung_spectroscopy"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_radiation():
    instrument = "radiation"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_helike_spectroscopy():
    instrument = "helike_spectroscopy"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_interferometry():
    instrument = "interferometry"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_filters():
    instrument = "filters"
    _test_get_methods(instrument)
    _test_catch_unimplemented_reader(instrument)


def test_empty():
    for instrument in _INSTRUMENT_METHODS.keys():
        print(instrument)
        _test_empty(instrument)


def test_invalid_quantity():
    for instrument in _INSTRUMENT_METHODS.keys():
        print(instrument)
        _test_invalid_quantity(instrument)


def test_downsample_ratio():
    for instrument in _INSTRUMENT_METHODS.keys():
        print(instrument)
        _test_downsample_ratio(instrument)
