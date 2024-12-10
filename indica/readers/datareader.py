"""Experimental design for reading data from disk/database."""

from abc import ABC
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
from xarray import DataArray

from indica import BaseIO
from indica.available_quantities import READER_QUANTITIES
from indica.configs.readers.machineconf import MachineConf
from indica.converters import CoordinateTransform
from indica.numpy_typing import RevisionLike
from indica.utilities import build_dataarrays


class DataReader(ABC):
    """Abstract base class to read data in from a database."""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf: MachineConf,
        reader_utils: BaseIO,
        verbose: bool = False,
        return_dataarrays: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        tstart
            Start of time range for which to get data.
        tend
            End of time range for which to get data.
        kwargs
            Any other arguments which should be recorded for the reader.
        """
        self.verbose = verbose
        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.reader_utils = reader_utils
        self.machine_conf = machine_conf()
        self.instrument_methods = self.machine_conf.INSTRUMENT_METHODS
        self.machine_dims = self.machine_conf.MACHINE_DIMS
        self.quantities_path = self.machine_conf.QUANTITIES_PATH
        self.verbose = verbose
        self.return_dataarrays = return_dataarrays
        self.kwargs = kwargs

    def get(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = 0,
        dl: float = 0.005,
        passes: int = 1,
        include_error: bool = True,
    ) -> Dict[str, DataArray]:
        """General method that reads data for a requested instrument."""
        if instrument not in self.instrument_methods.keys():
            raise ValueError(
                "{} does not support reading for instrument {}".format(
                    self.__class__.__name__, instrument
                )
            )
        # Read data from database
        _database_results = self._read_database(uid, instrument, revision)
        _database_results["dl"] = dl
        _database_results["passes"] = passes

        # Re-arrange data (machine-specific) and get instrument geometry transform
        method = self.instrument_methods[instrument]
        database_results, transform = getattr(self, f"_{method}")(_database_results)

        if self.return_dataarrays:
            quantities = READER_QUANTITIES[method]
            data_arrays = build_dataarrays(
                database_results,
                quantities,
                self.tstart,
                self.tend,
                transform,
                include_error,
            )
            return data_arrays
        else:
            return database_results

    def _read_database(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
    ) -> dict:
        """Read and return all raw database quantities and errors
        Exception handling is non-specific to guarantee generality across readers.

        Include in data dictionary also the UID, INSTRUMENT, MACHINE_DIMS and REVISION
        to guarantee data traceability across data-structures."""
        method = self.instrument_methods[instrument]
        quantities_paths = self.quantities_path[method]

        revision = self.reader_utils.get_revision(uid, instrument, revision)
        results: Dict[str, Any] = {
            "uid": uid,
            "instrument": instrument,
            "machine_dims": self.machine_dims,
            "revision": revision,
        }
        for _key, _path in quantities_paths.items():
            _path_err = _path + "_err"

            # Read quantity value
            try:
                q_val, q_path = self.reader_utils.get_signal(
                    uid,
                    instrument,
                    _path,
                    revision,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error reading {_path}: {e}")
                    raise e
                continue
            results[_key + "_records"] = q_path
            results[_key] = q_val

            # Read quantity error
            try:
                q_err, q_err_path = self.reader_utils.get_signal(
                    uid,
                    instrument,
                    _path_err,
                    revision,
                )
            except Exception as e:
                q_err = np.full_like(results[_key], 0.0)
                q_err_path = f"{e}"
            results[_key + "_error"] = q_err
            results[_key + "_error" + "_records"] = q_err_path

        return results

    # Machine-specific instrument methods that must be implemented in the child reader
    # to refactor database data structures and assign a geometry transform
    def _get_thomson_scattering(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_profile_fits(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_charge_exchange(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_spectrometer(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_equilibrium(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_radiation(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_helike_spectroscopy(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_diode_filters(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_interferometry(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_zeff(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError
