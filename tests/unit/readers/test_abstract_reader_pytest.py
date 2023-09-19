"""Test methods present on the base class DataReader."""

from copy import deepcopy
from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np

from indica import session
from indica.numpy_typing import RevisionLike
from indica.readers import DataReader
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


# TODO these values should come from the machine dimensions variable of the reader

TSTART = 0
TEND = 10


class Reader(DataReader):
    """Class to read fake data"""

    MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))
    INSTRUMENT_METHODS = {
        "thomson_scattering": "get_thomson_scattering",
        "equilibrium": "get_equilibrium",
        "cyclotron_emissions": "get_cyclotron_emissions",
        "charge_exchange": "get_charge_exchange",
        "spectrometer": "get_spectrometer",
        "bremsstrahlung_spectroscopy": "get_bremsstrahlung_spectroscopy",
        "radiation": "get_radiation",
        "helike_spectroscopy": "get_helike_spectroscopy",
        "interferometry": "get_interferometry",
        "diode_filters": "get_diode_filters",
    }

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "",
        default_error: float = 0.05,
        max_freq: float = 1e6,
        session: session.Session = session.global_session,
    ):
        self._reader_cache_id = ""
        self.NAMESPACE: Tuple[str, str] = ("", server)
        super().__init__(
            tstart,
            tend,
            max_freq,
            session,
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
        wavelength = np.arange(520, 530, 0.1)
        nt = times.shape[0]
        results["times"] = times
        results["wavelength"] = wavelength
        results["spectra"] = np.random.uniform(
            10, 10.0e3, (nt, results["length"], wavelength.size)
        )
        results["fit"] = deepcopy(results["spectra"])
        results["texp"] = np.full_like(times, dt)

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])

        results["element"] = "element"
        results["revision"] = np.random.randint(0, 10)
        results["x"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["y"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["z"] = np.random.uniform(zmin, zmax, (results["length"],))
        results["R"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["ti"] = np.random.uniform(10, 10.0e3, (nt, results["length"]))
        results["vtor"] = np.random.uniform(1.0e2, 1.0e6, (nt, results["length"]))
        results["angf"] = np.random.uniform(1.0e2, 1.0e6, (nt, results["length"]))
        results["conc"] = np.random.uniform(1.0e-6, 1.0e-1, (nt, results["length"]))
        results["bad_channels"] = []

        for quantity in quantities:
            results[f"{quantity}_records"] = [
                f"{quantity}_R_path",
                f"{quantity}_z_path",
                f"{quantity}_element_path",
                f"{quantity}_time_path",
                f"{quantity}_ti_path",
                f"{quantity}_angf_path",
                f"{quantity}_conc_path",
            ]

        return results

    def _get_spectrometer(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        wavelength = np.arange(520, 530, 0.1)
        nt = times.shape[0]
        results["times"] = times
        results["wavelength"] = wavelength
        results["spectra"] = np.random.uniform(
            10, 10.0e3, (nt, results["length"], wavelength.size)
        )

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])

        results["revision"] = np.random.randint(0, 10)

        for quantity in quantities:
            results[f"{quantity}_records"] = [
                f"{quantity}_time_path",
                f"{quantity}_spectra_path",
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

        results["x"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["y"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["z"] = np.random.uniform(zmin, zmax, (results["length"],))
        results["R"] = np.random.uniform(Rmin, Rmax, (results["length"],))
        results["chi2"] = np.random.uniform(0, 2.0, (nt, results["length"]))
        results["te"] = np.random.uniform(10, 10.0e3, (nt, results["length"]))
        results["ne"] = np.random.uniform(1.0e16, 1.0e21, (nt, results["length"]))
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
        results["ajac"] = np.random.uniform(1.0e-3, 2.0, (nt, nrho))
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

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]
        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])

        results["times"] = times
        results["brightness"] = np.random.uniform(0, 1.0e6, (nt, results["length"]))

        results["brightness_records"] = ["brightness_records"] * results["length"]

        return results

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])

        quantity = "zeff"
        results[quantity] = np.random.uniform(0, 1.0e6, (nt, results["length"]))

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

        nwavelength = np.random.randint(256, 1024)
        wavelength_start, wavelength_end = 3.8, 4.0

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }
        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])

        results["wavelength"] = np.linspace(
            wavelength_start, wavelength_end, nwavelength
        )

        for quantity in quantities:
            if quantity == "spectra":
                results[quantity] = np.random.uniform(
                    0, 1.0e6, (nt, results["length"], nwavelength)
                )
            else:
                results[quantity] = np.random.uniform(0, 1.0e4, (nt, results["length"]))

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

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["labels"] = np.array(["label"] * results["length"])

        for quantity in quantities:
            results[quantity] = np.random.uniform(0, 1.0e6, (nt, results["length"]))

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

        results: Dict[str, Any] = {
            "length": np.random.randint(4, 20),
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = np.random.randint(0, 10)
        dt = np.random.uniform(0.001, 1.0)
        times = np.arange(TSTART, TEND, dt)
        results["times"] = times
        nt = times.shape[0]

        results["location"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        results["direction"] = np.array([[1.0, 2.0, 3.0]] * results["length"])
        for quantity in quantities:
            results[quantity] = np.random.uniform(0, 1.0e6, (nt, results["length"]))

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
    nsamples=1,
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


def test_get_spectrometer():
    _test_get_methods(instrument="spectrometer", nsamples=10)


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
        instrument="diode_filters",
        nsamples=10,
    )


def test_get_interferometry():
    _test_get_methods(
        instrument="interferometry",
        nsamples=10,
    )
