"""Experimental design for reading data from disk/database."""

from abc import ABC
from typing import Any
from typing import Dict

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.abstractio import BaseIO
from indica.configs.readers.st40conf import MachineConf
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.numpy_typing import LabeledArray
from indica.numpy_typing import RevisionLike
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import format_dataarray


class DataReader(ABC):
    """Abstract base class to read data in from a database."""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf: MachineConf,
        reader_utils: BaseIO,
        default_error: float = 0.05,
        verbose: bool = False,
        return_dataarrays: bool = True,
        **kwargs: Any,
    ):
        """This should be called by constructors on subtypes.

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
        self.default_error = default_error
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
        data = self._read_database(uid, instrument, revision)

        # Re-arrange data (machine-specific)
        method = self.instrument_methods[instrument]
        data = getattr(self, f"_{method}")(data)

        if self.return_dataarrays:
            # Instatiate transforms
            if "location" in data and "direction" in data:
                transform = LineOfSightTransform(
                    data["location"][:, 0],
                    data["location"][:, 1],
                    data["location"][:, 2],
                    data["direction"][:, 0],
                    data["direction"][:, 1],
                    data["direction"][:, 2],
                    machine_dimensions=data["machine_dims"],
                    dl=dl,
                    passes=passes,
                )
            elif "x" in data and "y" in data and "z" in data:
                transform = TransectCoordinates(
                    data["x"],
                    data["y"],
                    data["z"],
                    machine_dimensions=data["machine_dims"],
                )
            else:
                transform: CoordinateTransform = TrivialTransform

            # Build data-arrays
            data = self._build_dataarrays(data, transform, include_error=include_error)

        return data

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

    def _build_dataarrays(
        self,
        data: Dict[str, LabeledArray],
        transform=None,
        include_error: bool = True,
    ) -> Dict[str, DataArray]:
        """Organizes database data in DataArray format with
        coordinates, long_name and units"""
        data_arrays: dict = {}
        uid = data["uid"]
        instrument = data["instrument"]
        revision = data["revision"]
        datatype_dims = AVAILABLE_QUANTITIES[self.instrument_methods[instrument]]
        for quantity in datatype_dims.keys():
            if quantity not in data.keys():
                continue

            # Build coordinate dictionary
            datatype, dims = datatype_dims[quantity]
            if self.verbose:
                print(f"  {quantity} - {datatype}")

            coords: dict = {}
            for dim in dims:
                coords[dim] = data[dim]

            # Build DataArray
            _data = format_dataarray(data[quantity], datatype, coords)
            if "t" in _data.dims:
                _data = _data.sel(t=slice(self.tstart, self.tend))
            _data = _data.sortby(dims)

            # Build error DataArray and assign as coordinate
            if include_error and len(dims) != 0:
                _error = xr.zeros_like(_data)
                if quantity + "_error" in data:
                    _error = format_dataarray(
                        data[quantity + "_error"], datatype, coords
                    )
                    if "t" in _error.dims:
                        _error = _error.sel(t=slice(self.tstart, self.tend))
                _data = _data.assign_coords(error=(_data.dims, _error.data))

            # Check that times are unique
            if "t" in _data:
                t_unique, ind_unique = np.unique(_data["t"], return_index=True)
                if len(_data["t"]) != len(t_unique):
                    _data = _data.isel(t=ind_unique)

            # Add attributes and assign to dictionary
            _data.attrs["transform"] = transform
            _data.attrs["uid"] = uid
            _data.attrs["revision"] = revision
            data_arrays[quantity] = _data
        return data_arrays

    def _get_thomson_scattering(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_profile_fits(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_charge_exchange(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_spectrometer(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_equilibrium(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_radiation(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_helike_spectroscopy(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_diode_filters(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_interferometry(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def _get_zeff(
        self,
        data: dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError
