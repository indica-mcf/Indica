"""Experimental design for reading data from disk/database.
"""

from abc import ABC, abstractmethod
from numbers import Number as Scalar
import socket
from typing import Any, ClassVar, Dict, Optional, Tuple, Union

from numpy import ndarray
from sal.client import SALClient, AuthenticationFailed
from sal.dataclass import Signal
from xarrays import DataArray


DataType = Tuple[str, str]
Number = Union[ndarray, Scalar]
OptNumber = Union[Number, None]


def trivial_remap(x1: OptNumber, x2: OptNumber, t: OptNumber) -> \
                                     Tuple[OptNumber, OptNumber, OptNumber]:
    """A trivial function for mapping between coordinate systems.

    This makes no change to the coordinates and should be used for
    data on the master coordinate system.

    Parameters
    ----------
    x1
        The first spatial coordinate (if there is one, otherwise ``None``)
    x2
        The second spatial coordinate (if there is one, otherwise ``None``)
    t
        The time coordinate (if there is one, otherwise ``None``)

    Returns
    -------
    x1 :
        New coordinate in first spatial dimension (if there is one, otherwise
        ``None``)
    x2 :
        New coordinate in second spatial dimension (if there is one, otherwise
        ``None``)
    t :
        New time coordinate( if there is one, otherwise ``None``)
    """
    return x1, x2, t


class DataReader(ABC):
    """Abstract base class to read data in from a database.

    This defines the interface used by all concrete objects which read
    data from the disc, a database, etc. It is a `context manager
    <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`_
    and can be used in a `with statement
    <https://docs.python.org/3/reference/compound_stmts.html#with>`_.

    Attributes
    ----------
    AVAILABLE_DATA: Dict[str, DataType]
        A mapping of the keys used to get each piece of data to the type of
        data associated with that key.
    """

    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {}

    def __enter__(self) -> 'DataReader':
        """Called at beginning of a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        """Close reader at end of context manager. Don't try to handle
        exceptions."""
        self.close()
        return False

    def get(self, key: str, revision: Optional[Any] = None) -> DataArray:
        """Reads data for the specified key.

        Reads data from a database or disk. This object will provide
        all necessary additional attributes to describe provenance and
        the coordinate system.

        This method requires :py:meth:`_get_data` to be overridden by
        a subclass. It will call that method, checking to ensure your
        data is listed as being available from this reader. It will
        also ensure the result contains necessary metadata such as
        provenance and conversions for the coordinate system.

        Parameters
        ----------
        key
            Identifies what data is to be read. Must be present in
            :py:attr:`AVAILABLE_DATA`.
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.

        Returns
        -------
        :
            The requested raw data, with appropriate metadata.
        """
        if key in self.AVAILABLE_DATA:
            raise ValueError("{} can not read data for key {}".format(
                self.__class__.__name__, repr(key)))
        # Check the data type is one that is registered globally
        result = self._get_data(key, revision)
        # Check the result has appropriate metadata
        return result

    def authenticate(name: str, password: str) -> bool:
        """Confirms user has permission to access data.

        This must be called before reading data from some sources. The default
        implementation does nothing. If the value of
        `py:meth:requires_authentication` is ``False`` then it does not need
        to be called.

        Parameters
        ----------
        name
            Username to authenticate against.
        password
            Password for that user.

        Returns
        -------
        :
            Indicates whether authentication was succesful.
        """
        return True

    @property
    @abstractmethod
    def requires_authentication(self) -> bool:
        """Indicates whether authentication is required to read data.

        Returns
        ----------
        :
            True of authenticationis needed, otherwise false.
        """
        raise NotImplementedError("{} does not implement a "
                                  "'requires_authentication' "
                                  "property.".format(self.__class__.__name__))

    @abstractmethod
    def _get_data(self, key: str, revision: Optional[Any] = None) -> DataArray:
        """An unsafe version of :py:meth:`get` which must be implemented by
        each subclass. It does not check that the key is valid."""
        raise NotImplementedError("{} does not implement a '_get_unsafe' "
                                  "method.".format(self.__class__.__name__))

    @abstractmethod
    def close(self) -> None:
        """Closes connection to whatever backend (file, database, server,
        etc.) from which data is being read."""
        raise NotImplementedError("{} does not implement a 'close' "
                                  "method.".format(self.__class__.__name__))


class PPFReader(DataReader):
    """Class to read JET PPF data using SAL.

    Currently the following types of data are supported.

    ==========  =============  ==============  =================
    Key         Data type      Data for        Instrument
    ==========  =============  ==============  =================
    efit_rmag   Major radius   Magnetic axis   EFIT equilibrium
    efit_zmag   Z position     Magnetic axis   EFIT equilibrium
    efit_rsep   Major radius   Separatrix      EFIT equilibrium
    efit_zsep   Z position     Separatrix      EFIT equilibrium
    ==========  =============  ==============  =================

    Note that there will need to be some refactoring to support other
    data types. However, **this is guaranteed not to affect the public
    interface**.

    Parameters
    ----------
    pulse
        The ID number for the pulse from which to get data.
    uid
        The UID for the particular data to be read.
    server
        The URL for the SAL server to read data from.

    Attributes
    ----------
    AVAILABLE_DATA: Dict[str, DataType]
        A mapping of the keys used to get each piece of data to the type of
        data associated with that key.
    pulse : int
        The ID number for the pulse from which to get data.
    uid : str
        The UID for the particular data to be read.

    """

    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {
        "efit_rmag": ("major_rad", "mag_axis"),
        "efit_zmag": ("z", "mag_axis"),
        "efit_rsep": ("major_rad", "separatrix_axis"),
        "efit_zsep": ("z", "separatrix_axis"),
        # "hrts_ne": ("number_density", "electrons"),
        # "hrts_te": ("temperature", "electrons"),
        # "lidr_ne": ("number_density", "electrons"),
        # "lidr_te": ("temperature", "electrons"),
    }

    def __init__(self, pulse: int, uid: str = "jetppf",
                 server: str = "https://sal.jet.uk"):
        self.pulse = pulse
        self.uid = uid
        self._client = SALClient(server)

    def _get_data(self, key: str, revision: int = 0) -> DataArray:
        """Reads and returns the data for the given key. Should only be called
        by :py:meth:`DataReader.get`."""
        path = "/pulse/{:i}/ppf/signal/{}/{}:{:i}"
        signal = self._client.get(path.format(self.pulse, self.uid,
                                              key.replace("_", "/"), revision))
        provenance = None  # TODO: construct provenance data
        meta = {"map_to_master": trivial_remap,
                "map_from_master": trivial_remap,
                "datatype": self.AVAILABLE_DATA[key],
                "provenance": provenance}
        return DataArray(signal.data, {'t': signal.dimensions[0].data}, "t",
                         key, meta)

    # def _get_hrts_ne(self, revision: int = 0) -> DataArray:
    #     key = "hrts_ne"
    #     signal = self._get_signal(key, revision)
    #     error = self._get_signal("hrts_dne", revision)
    #     provenance = None  # TODO: construct provenance data
    #     meta = {"datatype": self.AVAILABLE_DATA[key], "error": error.data, "provenance": provenance}
    #     return DataArray(signal.data, {"t": signal.dimensions[0].data,
    #                                    "R0": signal.dimensions[0].data},
    #                      ["t", "R0"], "hrts_ne", key, meta)

    def close(self):
        """Ends connection to the SAL server from which PPF data is being
        read."""
        del self.server

    @property
    def requires_authentication(self):
        # Perform the necessary logic to know whether authentication is needed.
        return not socket.gethostname().startswith("heimdall")

    def authenticate(self, name: str, password: str):
        """Log onto the JET/SAL system to access data.

        Parameters
        ----------
        name:
            Your username when logging onto Heimdall.
        password:
            SecureID passcode (pin followed by value displayed on token).

        Returns
        -------
        :
            Indicates whether authentication was succesful.
        """
        try:
            self._client.authenticate(name, password)
            return True
        except AuthenticationFailed:
            return False
