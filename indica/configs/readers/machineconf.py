# TODO: implement test to check availability of instruments methods and
#       quantities have corresponding DATATYPE
# TODO: Make BaseConf class with all the details of what a
#  configuration class should contain?
# TODO: why a class and not a simple JSON file or a dictionary?

from abc import ABC
from typing import Dict
from typing import Tuple


class MachineConf(ABC):
    def __init__(self):
        self.MACHINE_DIMS: Tuple[Tuple[float, float], Tuple[float, float]] = ()
        self.INSTRUMENT_METHODS: Dict[str, str] = {}
        self.QUANTITIES_PATH: Dict[str, Dict[str, str]] = {}
