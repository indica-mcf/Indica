"""Module to provide objects for reading fusion data from the
filesystem or databases. These objects are all subclasses of
:py:class:`readers.DataReader`, with each subclass providing
functionality for a different format of data.

"""

from .abstractreader import DataReader
from .adas import ADASReader
from .ppfreader import PPFReader
from .selectors import choose_on_plot
from .selectors import DataSelector
from .st40reader import ST40Reader

__all__ = [
    "ADASReader",
    "DataReader",
    "PPFReader",
    "ST40Reader",
    "choose_on_plot",
    "DataSelector",
]
