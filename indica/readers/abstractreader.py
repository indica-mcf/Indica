"""Experimental design for reading data from disk/database."""

from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple
from abc import ABC

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.configs.readers.st40conf import MachineConf
from indica.numpy_typing import OnlyArray
from indica.numpy_typing import RevisionLike, LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import format_dataarray, get_function_name


class DataReader(ABC):
    """Abstract base class to read data in from a database."""
    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        conf: MachineConf,
        utils: MachineConf,  
        default_error: float = 0.05,
        verbose:bool = False,
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
        self.utils = utils
        self.conf = conf()
        self.instrument_methods = self.conf.INSTRUMENT_METHODS
        self.machine_dims = self.conf.MACHINE_DIMS
        self.quantities_path = self.conf.QUANTITIES_PATH
        self.default_error = default_error
        self.verbose = verbose
        self.kwargs = kwargs

    def get(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = 0,
        include_error:bool=True,
    ) -> Dict[str, DataArray]:
        """General method that reads data for a requested instrument."""
        if instrument not in self.instrument_methods.keys():
            raise ValueError(
                "{} does not support reading for instrument {}".format(
                    self.__class__.__name__, instrument
                )
            )
        # Read data from database
        database_results = self._read_database(uid, instrument, revision)
        
        # Rearrange data and generate coordinate transform
        method = self.instrument_methods[instrument]
        database_results, transform = getattr(self, method)(database_results)
        
        # Build Indica-native DataArray structures
        data = self._build_dataarrays(database_results, transform, include_error=include_error)

        return data

    def _read_database(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
    ) -> dict:
        """ Read and return all raw database quantities and errors
        Exception handling is non-specific to guarantee generality across readers.

        Include in data dictionary also the UID, INSTRUMENT, MACHINE_DIMS and REVISION
        to guarantee data traceability across data-structures. """
        method = self.instrument_methods[instrument]
        quantities_paths = self.quantities_path[method]
        
        revision = self.utils.get_revision(uid, instrument, revision)
        results: Dict[str, Any] = {
            "uid":uid,
            "instrument":instrument,
            "machine_dims": self.machine_dims,
            "revision": revision,
        }
        for _key, _path in quantities_paths.items():
            # Read quantity value
            try:
                q_val, q_path = self.utils.get_signal(
                    uid,
                    instrument,
                    _path,
                    revision,
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error reading {_path}: {e}")                
                continue
            results[_key + "_records"] = q_path
            results[_key] = q_val

            # Read quantity error
            try:
                q_err, q_err_path = self.utils.get_signal(
                    uid,
                    instrument,
                    _path + "_err",
                    revision,
                )
            except Exception as e:
                q_err = np.full_like(results[_key], 0.0)
                q_err_path = ""
            results[_key + "_error"] = q_err
            results[_key + "_error" + "_records"] = q_err_path

        return results

    def _build_dataarrays(
        self,
        database_results: Dict[str, LabeledArray],
        transform=None,
        include_error: bool = True,
    ) -> Dict[str, DataArray]:
        """Organizes database data in DataArray format with coordinates, long_name and units"""
        data:dict = {}
        uid = database_results["uid"]
        instrument = database_results["instrument"]
        revision = database_results["revision"]
        datatype_dims = AVAILABLE_QUANTITIES[self.instrument_methods[instrument]]        
        for quantity in datatype_dims.keys():
            if quantity not in database_results.keys():
                continue

            # Build coordinate dictionary
            datatype, dims = datatype_dims[quantity]
            coords = {}
            for dim in dims:
                coords[dim] = database_results[dim]

            # Build DataArray
            _data = format_dataarray(database_results[quantity], datatype, coords)
            if "t" in data.dims:
                _data = _data.sel(t=slice(self._tstart, self._tend))
            _data = _data.sortby(dims)

            # Build error DataArray and assign as coordinate
            if include_error and len(dims) != 0:
                _error = xr.zeros_like(_data)
                if quantity + "_error" in database_results:
                    _error = format_dataarray(
                        database_results[quantity + "_error"], datatype, coords
                    )
                    if "t" in _error.dims:
                        _error = _error.sel(t=slice(self._tstart, self._tend))
                _data = _data.assign_coords(error=(_data.dims, _error.data))

            # Check that times are unique
            if "t" in database_results:
                t_unique, ind_unique = np.unique(database_results["t"], return_index=True)
                if len(database_results["t"]) != len(t_unique):
                    _data = _data.isel(t=ind_unique)

            # Add attributes
            _data.attrs["transform"] = transform
            _data.attrs["uid"] = uid
            _data.attrs["revision"] = revision

            data[quantity] = _data

        return data

    def get_thomson_scattering(
        self,
        database_results:dict,
    ) -> Dict[str, DataArray]:
        database_results = self._get_thomson_scattering(database_results)
        transform = TransectCoordinates(
            database_results["x"],
            database_results["y"],
            database_results["z"],
            machine_dimensions=self.conf["machine_dims"],
        )
        return database_results, transform

    def _get_thomson_scattering(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_profile_fits(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results = self._get_profile_fits(database_results)
        transform = None
        return database_results, transform

    def _get_profile_fits(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_charge_exchange(
        self,
        database_results:dict,
    ) -> Dict[str, DataArray]:
        database_results = self._get_charge_exchange(database_results)
        transform = TransectCoordinates(
            database_results["x"],
            database_results["y"],
            database_results["z"],
            machine_dimensions=database_results["machine_dims"],
        )
        return database_results, transform

    def _get_charge_exchange(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_spectrometer(
        self,
        database_results:dict,
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        database_results = self._get_charge_exchange(database_results)
        transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            database_results["machine_dims"],
            dl,
            passes,
        )
        return database_results, transform

    def _get_spectrometer(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_equilibrium(
        self,
        database_results:dict,
    ) -> Dict[str, DataArray]:
        database_results = self._get_equilibrium(database_results)

        # Reorganise coordinates
        database_results["rho_poloidal"] = np.sqrt(database_results["psin"])
        if "rbnd" in database_results.keys():
            database_results["boundary_index"] = np.arange(
                np.size(database_results["rbnd"][0, :])
            )
        return database_results

    def _get_equilibrium(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_radiation(
        self,
        database_results:dict,
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        database_results = self._get_radiation(database_results)
        transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            database_results["machine_dims"],
            dl,
            passes,
        )
        return database_results, transform

    def _get_radiation(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_helike_spectroscopy(
        self,
        database_results:dict,
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        database_results = self._get_radiation(database_results)
        transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            database_results["machine_dims"],
            dl,
            passes,
        )
        return database_results, transform

    def _get_helike_spectroscopy(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_diode_filters(
        self,
        database_results:dict,
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        database_results = self._get_diode_filters(database_results)
        transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            database_results["machine_dims"],
            dl,
            passes,
        )
        # data[quantity] = quant_data.assign_coords(
        #     label=("channel", database_results["labels"])
        # )
        return database_results, transform

    def _get_diode_filters(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError
    
    def get_interferometry(
        self,
        database_results:dict,
        dl: float = 0.005,
        passes: int = 2,
    ) -> Dict[str, DataArray]:
        database_results = self._get_diode_filters(database_results)
        transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            database_results["machine_dims"],
            dl,
            passes,
        )
        return database_results, transform

    def _get_interferometry(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def get_zeff(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        database_results = self._get_zeff(uid, instrument, revision, quantities)

        data = {}
        for quantity in quantities:
            if database_results.get(quantity) is None:
                continue
            _path: str = database_results[f"{quantity}_records"]
            print(_path)
            if "global" in _path.lower():
                dims = ["t"]
            elif "profiles" in _path.lower():
                dims = ["t", "rho_poloidal"]
            else:
                raise ValueError(f"Unknown quantity: {quantity}")

            data[quantity] = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=None,
            )
        return data

    def _get_zeff(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for ZEFF analysis from the database
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a '_get_zeff' " "method."
        )

def instatiate_line_of_sight(
    location: OnlyArray,
    direction: OnlyArray,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]],
    dl: float,
    passes: int,
) -> LineOfSightTransform:
    return LineOfSightTransform(
        location[:, 0],
        location[:, 1],
        location[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        machine_dimensions=machine_dimensions,
        dl=dl,
        passes=passes,
    )

