"""Module to provide objects for reading fusion data from the
filesystem or databases. These objects are all subclasses of
:py:class:`readers.DataReader`, with each subclass providing
functionality for a different format of data.

"""

from .abstractreader import DataReader
from .adas import ADASReader
from .adf11 import ADF11Reader
from .ppfreader import PPFReader
from .selectors import choose_on_plot
from .selectors import DataSelector

__all__ = [
    "ADASReader",
    "ADF11Reader",
    "DataReader",
    "PPFReader",
    "choose_on_plot",
    "DataSelector",
]
