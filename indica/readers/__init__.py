"""Module to provide objects for reading fusion data from the
filesystem or databases. These objects are all subclasses of
:py:class:`readers.DataReader`, with each subclass providing
functionality for a different format of data.

"""

from .adas import ADASReader

__all__ = ["ADASReader"]

try:
    from .salutils import SALUtils

    __all__ += ["SALUtils"]
except ImportError:
    pass

try:
    from .jetreader import JETReader

    __all__ += ["JETReader"]
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
