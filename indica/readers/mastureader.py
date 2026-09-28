"""Provides implementation of :py:class:`readers.DataReader` for reading UDA data
produced by MAST-U

"""

from typing import Any
from typing import Dict
from typing import Tuple
import warnings

import numpy as np
from xarray import DataArray

from indica import Equilibrium
from indica.abstractio import BaseIO
from indica.available_quantities import READER_QUANTITIES
from indica.configs.readers.machineconf import MachineConf
from indica.configs.readers.mastuconf import MASTUConf
from indica.converters import CoordinateTransform
from indica.numpy_typing import RevisionLike
from indica.readers.actcalib import ACTCalib
from indica.readers.actcalib import ACTCalibError
from indica.readers.actcalib import KERNEL_HALFPIX
from indica.readers.datareader import DataReader
from indica.readers.jetreader import assign_trivial_transform
from indica.readers.udautils import UDAUtils


class MASTUReaderError(Exception):
    """An error for MASTUReader"""


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
        d = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        return super().get(**d)

    def _equilibrium(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:
        data["index"] = data["rbnd_dimensions"][1]
        transform = assign_trivial_transform()
        return data, transform

    def _cx_spectrometer(
        self,
        data: dict,
    ) -> Tuple[Dict[str, Any], CoordinateTransform]:

        # Retreive vital info
        instrument = data["instrument"]
        revision = data["revision"]

        # Assign transform
        transform = assign_trivial_transform()

        # Assign pixel number
        if "wavelength" in data.keys():
            data["index"] = np.array(range(np.shape(data["wavelength"])[1]))
        else:
            raise MASTUReaderError(f"Could not read {instrument} wavelength from UDA")
        if "kernel_index" not in data.keys():
            data["kernel_index"] = np.array(range((2 * KERNEL_HALFPIX) + 1))

        # Deduce which (if any) quantities are missing from database results
        quantities = READER_QUANTITIES["cx_spectrometer"]
        absent_quantities = [x for x in quantities if x not in data.keys()]

        # Wrap up if nothing is missing
        if len(absent_quantities) == 0:
            return data, transform

        # Look in act_calib for missing items
        act_calib = ACTCalib(self.pulse, kernel_halfpix=KERNEL_HALFPIX)
        for aq in absent_quantities:
            data_path = self.machine_conf.QUANTITIES_PATH["cx_spectrometer"][aq]
            try:
                data[aq] = act_calib.get_vm_data(
                    construct_act_calib_path(data_path),
                    instrument,
                    revision,
                )
                warnings.warn(f"{aq} read from `act_calib` database")
            except ACTCalibError as e:
                e.add_note(f"Occured while reading {aq} ({data_path}) for {instrument}")
                raise

        return data, transform


def construct_act_calib_path(uda_path):
    """Convert a uda path into a act_calib path"""

    s = uda_path.split("/")

    # Capitalise view if necessary
    if s[0].lower() in ["ss", "bg"]:
        s[0] = s[0].upper()

    # Resolve aliases
    aliases = {"sensitivity": "spectral_filter"}
    s = [aliases[x.lower()] if x.lower() in aliases.keys() else x for x in s]

    # Final entry in camel caps
    s[-1] = "".join([u.capitalize() for u in s[1].lower().split("_")])

    # Rejoin string
    s = "/".join(s)

    return s
