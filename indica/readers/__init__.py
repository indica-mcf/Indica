"""Module to provide objects for reading fusion data from the
filesystem or databases. These objects are all subclasses of
:py:class:`readers.DataReader`, with each subclass providing
functionality for a different format of data.

"""
from .adas import ADASReader
from .readerprocessor import ReaderProcessor
__all__ = ["ADASReader", "ReaderProcessor"]

try:
    from .ppfreader import PPFReader

    __all__ += ["PPFReader"]
except ImportError:
    pass

try:
    from .mdsutils import MDSUtils

    __all__ += ["MDSUtils"]
except ImportError as e:
    print(e)
    pass


try:
    from .st40reader import ST40Reader

    __all__ += ["ST40Reader"]
except ImportError as e:
    print(e)
    pass
