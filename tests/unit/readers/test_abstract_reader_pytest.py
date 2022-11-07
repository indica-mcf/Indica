"""Test methods present on the base class DataReader."""

from numbers import Number
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from xarray import DataArray

from indica import session
from indica.numpy_typing import RevisionLike
from indica.readers import DataReader
from indica.readers.abstractreader import DataSelector
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


# TODO these values should come from the machine dimensions variable of the reader

TSTART = 0
TEND = 10


def selector(
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number],
    unselected_channels: Iterable[Number] = [],
):
    return bad_channels


class Reader(DataReader):
    """Class to read fake data"""

    MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
    INSTRUMENT_METHODS = {
        "thomson_scattering": "get_thomson_scattering",
        "equilibrium": "get_equilibrium",
        "cyclotron_emissions": "get_cyclotron_emissions",
        "charge_exchange": "get_charge_exchange",
        "bremsstrahlung_spectroscopy": "get_bremsstrahlung_spectroscopy",
        "radiation": "get_radiation",
        "helike_spectroscopy": "get_helike_spectroscopy",
        "interferometry": "get_interferometry",
        "filters": "get_diode_filters",
    }

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "",
        default_error: float = 0.05,
        max_freq: float = 1e6,
        selector: DataSelector = selector,
        session: session.Session = session.global_session,
    ):
        self._reader_cache_id = ""
        self.NAMESPACE: Tuple[str, str] = ("", server)
        super().__init__(
            tstart,
            tend,
            max_freq,
            session,
            selector,
            pulse=pulse,
            server=server,
            default_error=default_error,
        )
        self.pulse = pulse

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        results["texp"] = np.full_like(times, dt)
        nt = times.shape[0]
        results["element"] = "element"
        results["revision"] = np.random.randint(0, 10)
        results["R"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["z"] = np.random.uniform(zmin, zmax, (results["length"],))
        results["ti"] = np.random.uniform(10, 10.0e3, (nt, results["length"]))
        results["ti_error"] = np.sqrt(results["ti"])
        results["angf"] = np.random.uniform(1.0e2, 1.0e6, (nt, results["length"]))
        results["angf_error"] = np.sqrt(results["angf"])
        results["conc"] = np.random.uniform(1.0e-6, 1.0e-1, (nt, results["length"]))
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

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }

        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        nt = times.shape[0]
        results["times"] = times
        results["revision"] = np.random.randint(0, 10)
        results["z"] = np.random.uniform(zmin, zmax, (results["length"],))
        results["R"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["te"] = np.random.uniform(10, 10.0e3, (nt, results["length"]))
        results["ne"] = np.random.uniform(1.0e16, 1.0e21, (nt, results["length"]))
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

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {}

        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]
        nrho = np.random.randint(20, 40)

        results["element"] = "element"

        results["R"] = np.random.uniform(Rmin, Rmax, (nrho,))
        results["z"] = np.random.uniform(zmin, zmax, (nrho,))

        results["rgeo"] = np.random.uniform(Rmin, Rmax, (nt,))
        results["rmag"] = np.random.uniform(Rmin, Rmax, (nt,))
        results["zmag"] = np.random.uniform(zmin, zmax, (nt,))
        results["ipla"] = np.random.uniform(1.0e4, 1.0e6, (nt,))
        results["wp"] = np.random.uniform(1.0e3, 1.0e5, (nt,))
        results["df"] = np.random.uniform(0, 1, (nt,))
        results["faxs"] = np.random.uniform(1.0e-6, 0.1, (nt,))
        results["fbnd"] = np.random.uniform(-1, 1, (nt,))

        results["psin"] = np.random.uniform(0, 1, (nrho,))
        results["psi_r"] = np.random.uniform(Rmin, Rmax, (nrho,))
        results["psi_z"] = np.random.uniform(zmin, zmax, (nrho,))

        results["f"] = np.random.uniform(1.0e-6, 0.1, (nt, nrho))
        results["ftor"] = np.random.uniform(1.0e-4, 1.0e-2, (nt, nrho))
        results["vjac"] = np.random.uniform(1.0e-3, 2.0, (nt, nrho))
        results["rmji"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
        results["rmjo"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
        results["rbnd"] = np.random.uniform(Rmin, Rmax, (nt, nrho))
        results["zbnd"] = np.random.uniform(zmin, zmax, (nt, nrho))
        results["psi"] = np.random.uniform(-1, 1, (nt, nrho, nrho))

        results["revision"] = np.random.randint(0, 10)

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

        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }

        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["revision"] = np.random.randint(0, 10)
        results["z"] = np.random.uniform(zmin, zmax)

        results["te"] = np.random.uniform(10, 10.0e3, (nt, results["length"]))
        results["te_error"] = np.sqrt(results["te"])
        results["Btot"] = np.random.uniform(0.1, 5, (results["length"],))
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

        _, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        length = np.random.randint(4, 20)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["v_times"] = times
        results["v"] = np.random.uniform(0, 1.0e6, (nt, length))
        results["v_error"] = np.sqrt(results["v"])
        results["v_xstart"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["v_ystart"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["v_zstart"] = np.random.uniform(zmin, zmax, (length,))
        results["v_xstop"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["v_ystop"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["v_zstop"] = np.random.uniform(zmin, zmax, (length,))

        results["h_times"] = times
        results["h"] = np.random.uniform(0, 1.0e6, (nt, length))
        results["h_error"] = np.sqrt(results["h"])
        results["h_xstart"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["h_ystart"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["h_zstart"] = np.random.uniform(zmin, zmax, (length,))
        results["h_xstop"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["h_ystop"] = np.random.uniform(-Rmax, Rmax, (length,))
        results["h_zstop"] = np.random.uniform(zmin, zmax, (length,))

        results["length"]["v"] = int(length)
        results["length"]["h"] = int(length)

        for quantity in quantities:
            results[f"{quantity}_records"] = [f"{quantity}_records"] * length

        return results

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        length = int(1)
        signals = ["zefv", "zefh"]
        for k in signals:
            results[k] = np.random.uniform(0, 1.0e6, (nt,))
            results[f"{k}_error"] = np.sqrt(results[k])
            results[f"{k}_xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
            results[f"{k}_ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
            results[f"{k}_zstart"] = np.array([np.random.uniform(zmin, zmax)])
            results[f"{k}_xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
            results[f"{k}_ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
            results[f"{k}_zstop"] = np.array([np.random.uniform(zmin, zmax)])
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

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        nwavelength = np.random.randint(256, 1024)
        wavelength_start, wavelength_end = 3.8, 4.0

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["wavelength"] = np.linspace(
            wavelength_start, wavelength_end, nwavelength
        )

        results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
        results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
        for quantity in quantities:
            if quantity == "spectra":
                results[quantity] = np.random.uniform(0, 1.0e6, (nt, nwavelength))
            else:
                results[quantity] = np.random.uniform(0, 1.0e4, (nt,))
            results[f"{quantity}_error"] = np.sqrt(results[quantity])

            results[f"{quantity}_records"] = [
                f"{quantity}_path_records",
            ]

        return results

    def _get_diode_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
        results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
        for quantity in quantities:
            results[quantity] = np.random.uniform(0, 1.0e6, (nt,))
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

        Rmin, Rmax = self.MACHINE_DIMS[0][0], self.MACHINE_DIMS[0][1]
        zmin, zmax = self.MACHINE_DIMS[1][0], self.MACHINE_DIMS[1][1]

        results: Dict[str, Any] = {
            "length": 1,
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["xstart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystart"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstart"] = np.array([np.random.uniform(zmin, zmax)])
        results["xstop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["ystop"] = np.array([np.random.uniform(Rmin, Rmax)])
        results["zstop"] = np.array([np.random.uniform(zmin, zmax)])
        for quantity in quantities:
            results[quantity] = np.random.uniform(0, 1.0e6, (nt,))
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


def _test_get_methods(
    instrument="ts",
    nsamples=10,
):
    """
    Generalised test for all get methods of the abstractreader
    """

    for i in range(nsamples):
        reader = Reader(
            1,
            TSTART,
            TEND,
        )

        quantities = set(AVAILABLE_QUANTITIES[reader.INSTRUMENT_METHODS[instrument]])

        results = reader.get("", instrument, 0, quantities)

        # Check whether data is as expected
        for q, actual, expected in [(q, results[q], results[q]) for q in quantities]:
            assert np.all(actual.values == expected)


def test_get_thomson_scattering():
    _test_get_methods(instrument="thomson_scattering", nsamples=10)


def test_get_charge_exchange():
    _test_get_methods(instrument="charge_exchange", nsamples=10)


def test_get_cyclotron_emissions():
    _test_get_methods(instrument="cyclotron_emissions", nsamples=10)


def test_get_equilibrium():
    _test_get_methods(instrument="equilibrium", nsamples=10)


def test_get_radiation():
    _test_get_methods(instrument="radiation", nsamples=10)


def test_get_bremsstrahlung_spectroscopy():
    _test_get_methods(
        instrument="bremsstrahlung_spectroscopy",
        nsamples=10,
    )


def test_get_helike_spectroscopy():
    _test_get_methods(
        instrument="helike_spectroscopy",
        nsamples=10,
    )


def test_get_diode_filters():
    _test_get_methods(
        instrument="filters",
        nsamples=10,
    )


def test_get_interferometry():
    _test_get_methods(
        instrument="interferometry",
        nsamples=10,
    )
