"""Refactoring of data read from the database to build DataArrays"""

from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np

from indica.configs.readers import ST40Conf
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.readers.datareader import DataReader
from indica.readers.mdsutils import MDSUtils


class ST40Reader(DataReader):
    """Class to read ST40 MDS+ data using MDSplus."""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf=ST40Conf,
        reader_utils=MDSUtils,
        server: str = "smaug",
        tree: str = "ST40",
        verbose: bool = False,
        default_error: float = 0.05,
        **kwargs: Any,
    ):

        if tstart < 0:
            tstart = 0

        super().__init__(
            pulse,
            tstart,
            tend,
            machine_conf=machine_conf,
            reader_utils=reader_utils,
            server=server,
            verbose=verbose,
            **kwargs,
        )
        self.default_error = (default_error,)
        self.reader_utils = self.reader_utils(pulse, server, tree)

    def _get_thomson_scattering(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        R = database_results["R"]
        database_results["channel"] = np.arange(len(R))
        database_results["z"] = R * 0.0
        database_results["x"] = R
        database_results["y"] = R * 0.0
        transform = assign_transect_transform(database_results)
        return database_results, transform

    def _get_profile_fits(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        # TODO: R_data and zpos issues need to be fixed in the database
        R_data = database_results["R_data"]
        if len(np.shape(R_data)) > 1:
            database_results["R_data"] = R_data[0, :]
        if "zpos" not in database_results:
            database_results["z"] = np.full_like(database_results["R"], 0)
        database_results["channel"] = np.arange(len(database_results["R_data"]))
        transform = assign_trivial_transform()
        return database_results, transform

    def _get_charge_exchange(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        database_results["channel"] = np.arange(len(database_results["x"]))
        database_results["element"] = ""
        if "wavelength" in database_results.keys():
            if len(np.shape(database_results["wavelength"])) > 1:
                database_results["wavelength"] = database_results["wavelength"][0, :]
            database_results["pixel"] = np.arange(len(database_results["wavelength"]))

        transform = assign_transect_transform(database_results)
        return database_results, transform

    def _get_spectrometer(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        # Sort channel indexing either hardcore or
        # selecting channels with finite data only
        spectra = database_results["spectra"]
        if database_results["instrument"] == "pi":
            has_data = np.arange(21, 28)
        else:
            has_data = np.where(np.isfinite(spectra[0, :, 0]) * (spectra[0, :, 0] > 0))[
                0
            ]
        database_results["spectra"] = database_results["spectra"][:, has_data, :]
        database_results["spectra_error"] = database_results["spectra_error"][
            :, has_data, :
        ]
        database_results["location"] = database_results["location"][has_data, :]
        database_results["direction"] = database_results["direction"][has_data, :]
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        if len(np.shape(database_results["wavelength"])) > 1:
            database_results["wavelength"] = database_results["wavelength"][0, :]

        rearrange_geometry(database_results["location"], database_results["direction"])

        transform = assign_lineofsight_transform(database_results)
        return database_results, transform

    def _get_equilibrium(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        # Add boundary index
        database_results["index"] = np.arange(np.size(database_results["rbnd"][0, :]))
        # Re-shape psi matrix
        database_results["psi"] = database_results["psi"].reshape(
            (
                len(database_results["t"]),
                len(database_results["z"]),
                len(database_results["R"]),
            )
        )
        transform = assign_trivial_transform()
        return database_results, transform

    def _get_radiation(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        transform = assign_lineofsight_transform(database_results)
        return database_results, transform

    def _get_radiation_inversion(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        transform = assign_trivial_transform()
        return database_results, transform

    def _get_helike_spectroscopy(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        transform = assign_lineofsight_transform(database_results)
        return database_results, transform

    def _get_diode_filters(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        _labels = database_results["label"]
        if type(_labels[0]) == np.bytes_:
            database_results["label"] = np.array(
                [label.decode("UTF-8") for label in _labels]
            )
        else:
            database_results["label"] = _labels
        rearrange_geometry(database_results["location"], database_results["direction"])
        transform = assign_lineofsight_transform(database_results)
        return database_results, transform

    def _get_interferometry(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        # if database_results["instrument"] == "smmh":
        #     location = (location + location_r) / 2.0
        #     direction = (direction + direction_r) / 2.0
        database_results["passes"] = 2
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        rearrange_geometry(database_results["location"], database_results["direction"])
        transform = assign_lineofsight_transform(database_results)
        return database_results, transform

    def _get_zeff(
        self,
        database_results: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        transform = assign_trivial_transform()
        return database_results, transform


def rearrange_geometry(location, direction):
    if len(np.shape(location)) == 1:
        location = np.array([location])
        direction = np.array([direction])


def assign_lineofsight_transform(database_results: Dict):
    transform = LineOfSightTransform(
        database_results["location"][:, 0],
        database_results["location"][:, 1],
        database_results["location"][:, 2],
        database_results["direction"][:, 0],
        database_results["direction"][:, 1],
        database_results["direction"][:, 2],
        machine_dimensions=database_results["machine_dims"],
        dl=database_results["dl"],
        passes=database_results["passes"],
    )
    return transform


def assign_transect_transform(database_results: Dict):
    transform = TransectCoordinates(
        database_results["x"],
        database_results["y"],
        database_results["z"],
        machine_dimensions=database_results["machine_dims"],
    )

    return transform


def assign_trivial_transform():
    transform = TrivialTransform()
    return transform
