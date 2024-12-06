# TODO: implement test to check availability of instruments methods and
#       quantities have corresponding DATATYPE
# TODO: why a class and not a simple JSON file or a dictionary?

from abc import ABC
from typing import Dict
from typing import Tuple


class MachineConf(ABC):
    """Machine configuration abstract class containing:
    MACHINE_DIMS = ((R_min, R_ma), (z_min, z_max))
    INSTRUMENT_METHODS = {"instrument_name": "reader_method"}
    QUANTITIES_PATH = {"reader_method": {"quantity":"database_path"}
    """

    def __init__(self):
        self.MACHINE_DIMS: Tuple[Tuple[float, float], Tuple[float, float]] = ()
        self.INSTRUMENT_METHODS: Dict[str, str] = {}
        self.QUANTITIES_PATH: Dict[str, Dict[str, str]] = {}
