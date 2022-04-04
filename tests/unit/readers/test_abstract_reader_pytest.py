"""Test methods present on the base class DataReader."""

from copy import deepcopy
from numbers import Number
from typing import Collection
from typing import Set

import numpy as np
from xarray import DataArray

from indica.numpy_typing import RevisionLike
from indica.readers import PPFReader
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


# TODO these values should come from the machine dimensions variable of the reader


class Generate_data:
    def __init__(self):
        self.tstart = 0
        self.tend = 10

        self.length = np.random.randint(4, 20)
        self.dt = np.random.uniform(0.001, 1.0)
        self.times = np.arange(self.tstart, self.tend, self.dt)
        self.texp = np.full_like(self.times, self.dt)
        nt = len(self.times)

        self.element = "element"

        self.machine_dims = ((1.83, 3.9), (-1.75, 2.0))
        self.revision = np.random.randint(0, 10)
        self.times = self.times
        self.texp = self.texp

        self.R = np.random.uniform(0.2, 3.0, (self.length,))
        self.z = np.random.uniform(-1, 1, (self.length,))
        self.z_ece = np.random.uniform(-1, 1)

        self.Te = np.random.uniform(10, 10.0e3, (nt, self.length))
        self.Te_error = np.sqrt(self.Te)
        self.Ti = deepcopy(self.Te)
        self.Ti_error = deepcopy(self.Te_error)
        self.Ne = np.random.uniform(1.0e16, 1.0e21, (nt, self.length))
        self.Ne_error = np.sqrt(self.Ne)
        self.angf = np.random.uniform(1.0e2, 1.0e6, (nt, self.length))
        self.angf_error = np.sqrt(self.angf)
        self.conc = np.random.uniform(1.0e-6, 1.0e-1, (nt, self.length))
        self.conc_error = np.sqrt(self.conc)

        self.Btot = np.random.uniform(0.1, 5, (self.length))
        self.bad_channels = []


def _select_channels(
    category: str,
    uid: str,
    instrument: str,
    quantity: str,
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number] = [],
):
    return []


def test_get_thomson_scattering(nsamples=10):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""

    def _get_thomson_scattering(
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        empty=False,
    ):
        """Strategy to produce a dictionary of DataArrays of the type that
        could be returned by a read operation.
        """

        if empty:
            return {}

        database_results: dict = {}
        database_results["times"] = data.times
        database_results["revision"] = data.revision
        database_results["length"] = data.length
        database_results["z"] = data.z
        database_results["R"] = data.R
        database_results["te"] = data.Te
        database_results["ne"] = data.Ne
        database_results["te_error"] = data.Te_error
        database_results["ne_error"] = data.Ne_error

        for quantity in quantities:
            database_results[f"{quantity}_records"] = [
                f"{quantity}_Rz_path",
                f"{quantity}_value_path",
                f"{quantity}_error_path",
            ]

        return database_results

    for i in range(nsamples):
        data = Generate_data()

        category = "get_thomson_scattering"
        uid = "jetppf"
        instruments = ["hrts", "lidr"]
        quantities = AVAILABLE_QUANTITIES[category]

        reader = PPFReader(0, data.tstart, data.tend + data.dt)
        reader._get_thomson_scattering = _get_thomson_scattering
        reader._select_channels = _select_channels

        for instrument in instruments:
            database_results = reader._get_thomson_scattering(
                uid, instrument, data.revision, quantities
            )
            results = reader.get_thomson_scattering(
                uid, instrument, data.revision, quantities
            )

            for q, actual, expected in [
                (q, results[q], database_results[q]) for q in quantities
            ]:
                assert np.all(actual.values == expected)


def test_get_charge_exchange(nsamples=10):
    """Test the get_charge_exchange method correctly combines and processes
    raw data."""

    def _get_charge_exchange(
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        empty=False,
    ):
        """Strategy to produce a dictionary of DataArrays of the type that
        could be returned by a read operation.
        """

        if empty:
            return {}

        database_results: dict = {}
        database_results["R"] = data.R
        database_results["z"] = data.z
        database_results["length"] = data.length
        database_results["element"] = data.element
        database_results["texp"] = data.texp
        database_results["times"] = data.times
        database_results["angf"] = data.angf
        database_results["angf_error"] = data.angf_error
        database_results["conc"] = data.conc
        database_results["conc_error"] = data.conc_error
        database_results["ti"] = data.Ti
        database_results["ti_error"] = data.Ti_error
        database_results["revision"] = data.revision

        for quantity in quantities:
            database_results[f"{quantity}_records"] = [
                f"{quantity}_R_path",
                f"{quantity}_z_path",
                f"{quantity}_element_path",
                f"{quantity}_time_path",
                f"{quantity}_value_path",
                f"{quantity}_error_path",
            ]

        return database_results

    for i in range(nsamples):
        data = Generate_data()

        category = "get_charge_exchange"
        uid = "jetppf"
        instruments = ["cxg6"]
        quantities = AVAILABLE_QUANTITIES[category]

        reader = PPFReader(0, data.tstart, data.tend + data.dt)
        reader._get_charge_exchange = _get_charge_exchange
        reader._select_channels = _select_channels

        for instrument in instruments:
            database_results = reader._get_charge_exchange(
                uid, instrument, data.revision, quantities
            )
            results = reader.get_charge_exchange(
                uid, instrument, data.revision, quantities
            )

            for q, actual, expected in [
                (q, results[q], database_results[q]) for q in quantities
            ]:
                assert np.all(actual.values == expected)


def test_cyclotron_emissions(nsamples=10):
    """Test the get_charge_exchange method correctly combines and processes
    raw data."""

    def _get_cyclotron_emissions(
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        empty=False,
    ):
        """Strategy to produce a dictionary of DataArrays of the type that
        could be returned by a read operation.
        """

        if empty:
            return {}

        database_results: dict = {}
        database_results["machine_dims"] = data.machine_dims
        database_results["z"] = data.z_ece
        database_results["length"] = data.length
        database_results["Btot"] = data.Btot
        database_results["bad_channels"] = data.bad_channels
        database_results["times"] = data.times
        database_results["te"] = data.Te
        database_results["te_error"] = data.Te_error
        database_results["revision"] = data.revision

        for quantity in quantities:
            database_results[f"{quantity}_records"] = []
            for i in range(data.length):
                database_results[f"{quantity}_records"].append(f"chan_{i}_path")

        return database_results

    for i in range(nsamples):
        data = Generate_data()

        category = "get_cyclotron_emissions"
        uid = "jetppf"
        instruments = ["kk3"]
        quantities = AVAILABLE_QUANTITIES[category]

        reader = PPFReader(0, data.tstart, data.tend + data.dt)
        reader._get_cyclotron_emissions = _get_cyclotron_emissions
        reader._select_channels = _select_channels

        for instrument in instruments:
            database_results = reader._get_cyclotron_emissions(
                uid, instrument, data.revision, quantities
            )
            results = reader.get_cyclotron_emissions(
                uid, instrument, data.revision, quantities
            )

            for q, actual, expected in [
                (q, results[q], database_results[q]) for q in quantities
            ]:
                assert np.all(actual.values == expected)
