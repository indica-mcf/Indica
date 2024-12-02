"""Provides implementation of :py:class:`readers.DataReader` for reading PPF data
produced by JET

"""

from typing import Any
from typing import Dict
from typing import Tuple

from indica.abstractio import BaseIO
from indica.configs.readers import JETConf
from indica.configs.readers import MachineConf
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.readers.datareader import DataReader
from indica.readers.salutils import SALUtils


class JETReader(DataReader):
    """Class to read JET PPF data using SAL"""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf: MachineConf = JETConf,
        reader_utils: BaseIO = SALUtils,
        server: str = "https://sal.jetdata.eu",
        verbose: bool = False,
        default_error: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(
            pulse,
            tstart,
            tend,
            machine_conf=machine_conf,
            reader_utils=reader_utils,
            server=server,
            verbose=verbose,
            default_error=default_error,
            **kwargs,
        )
        self.reader_utils = self.reader_utils(pulse, server)

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

    def _get_cyclotron_emissions(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError

    def _get_density_reflectometer(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        raise NotImplementedError


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
