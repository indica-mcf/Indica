"""Provides implementation of :py:class:`readers.DataReader` for reading UDA data
produced by MAST-U

"""

from typing import Any
from typing import Dict
from typing import Tuple

from xarray import DataArray

from indica import Equilibrium
from indica.abstractio import BaseIO
from indica.configs.readers.machineconf import MachineConf
from indica.configs.readers.mastuconf import MASTUConf
from indica.converters import CoordinateTransform
from indica.numpy_typing import RevisionLike
from indica.readers.datareader import DataReader
from indica.readers.jetutils import assign_trivial_transform
from indica.readers.udautils import UDAUtils


class MASTUReader(DataReader):
    """Class to read MAST-U data from UDA"""

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        machine_conf: MachineConf = MASTUConf,
        reader_utils: BaseIO = UDAUtils,
        server: str = "",
        verbose: bool = False,
        default_error: float = 0.05,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------

        pulse
            MAST-U shotnumber - should be greater than 40000
        """

        if pulse < 40000:
            raise ValueError(f"MAST-U pulse number must be >= 40,000, got {pulse}")

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
        self.reader_utils = self.reader_utils(pulse)

    def get(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = "LATEST",
        dl: float = 0.005,
        passes: int = 1,
        return_dataarrays: bool = True,
        debug: bool = False,
        equilibrium: Equilibrium = None,
    ) -> Dict[str, DataArray]:
        """Wrap parent class method but use "LATEST" as default revision"""
        return super().get(**locals())

    def _equilibrium(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        transform = assign_trivial_transform()
        return data, transform
