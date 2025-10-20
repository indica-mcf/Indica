"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""


from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np

from indica.configs.readers import ST40Conf, TRANSPConf
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.converters import TransectCoordinates
from indica.converters import TrivialTransform
from indica.numpy_typing import RevisionLike
from indica.readers.datareader import DataReader
from indica.readers.mdsutils import MDSUtils


class TRANSPreader(DataReader):


    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf=TRANSPConf,
        reader_utils=MDSUtils,
        server: str = "smaug",
        tree: str = "TRANSP_TEST",
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




    #Right now this is essentially just get equlibrium
    def _get_transp(
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

    #Right now this is essentially j


    def __call__(
        self,
        instruments: list = None,
        revisions: Dict[str, RevisionLike] = None,
        debug: bool = False,
    ):

        if instruments is None:
            instruments = self.machine_conf.INSTRUMENT_METHODS.keys()
        if revisions is None:
            revisions = {instrument: 0 for instrument in instruments}
        for instr in instruments:
            if instr not in revisions.keys():
                revisions[instr] = 0

        self.data = {}
        for instrument in instruments:
            try:
                self.data[instrument] = self.get(
                    "",
                    instrument,
                    revisions[instrument],
                )
            except Exception as e:
                print(f"error reading: {instrument} \nException: {e}")
                if debug:
                    raise e
        return self.data


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
