"""Module to provide objects for reading fusion data from the
filesystem or databases. These objects are all subclasses of
:py:class:`readers.DataReader`, with each subclass providing
functionality for a different format of data.

"""

from .abstractreader import DataReader
from .adas import ADASReader

__all__ = ["ADASReader", "DataReader"]

try:
    from .ppfreader import PPFReader

    __all__ += ["PPFReader"]
except ImportError:
    pass

try:
    from .st40conf import ST40Conf

    __all__ += ["ST40Conf"]
except ImportError as e:
    print(e)
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
