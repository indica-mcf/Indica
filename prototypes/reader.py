"""Experimental design for reading data from disk/database.
"""

from abc import ABC, abstractmethod
import datetime
from numbers import Number as Scalar
import socket
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import prov.model as prov
from sal.client import SALClient, AuthenticationFailed
from sal.dataclass import Signal
from xarrays import DataArray, Dataset

import session


DataType = Tuple[str, str]
Number = Union[np.ndarray, Scalar]
OptNumber = Union[Number, None]
Coordinate = Tuple[OptNumber, OptNumber, OptNumber]
Remapper = Callable[[OptNumber, OptNumber, OptNumber], Coordinate]


def trivial_factory(equilibrium: Dataset) -> Tuple[Remapper, Remapper]:
    """Construct functions to map between the coordinate system on data
    and the master coordinate system.

    This particular factory produces :py:func:`trivial_remap` mapping
    functions.

    Parameters
    ---------
    equilibrium
        The set of equilibrium data to use when calculating mapping.

    Returns
    -------
    map_to : coordinate mapping function
        Function mapping from the data coordinates to the master coordinates.

    map_from : coordinate mapping function
        Function mapping from the master coordinates to the data coordinates.

    """
    return (trivial_remap, trivial_remap)


def trivial_remap(x1: OptNumber, x2: OptNumber, t: OptNumber) -> Coordinate:
    """A trivial function for mapping between coordinate systems.

    This makes no change to the coordinates and should be used for
    data on the master coordinate system.

    Parameters
    ----------
    x1 : array_like or None
        The first spatial coordinate (if there is one, otherwise ``None``)
    x2 : array_like or None
        The second spatial coordinate (if there is one, otherwise ``None``)
    t : array_like or None
        The time coordinate (if there is one, otherwise ``None``)

    Returns
    -------
    x1 : ndarray or None
        New coordinate in first spatial dimension (if there is one, otherwise
        ``None``)
    x2 : ndarray or None
        New coordinate in second spatial dimension (if there is one, otherwise
        ``None``)
    t : ndarray or None
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
    NAMESPACE: Tuple[str, str]
        The abbreviation and full URL for the PROV namespace of the reader
        class.
    prov_id: str
        The hash used to identify this object in provenance documents.
    agent: prov.model.ProvAgent
        An agent representing this object in provenance documents.
        DataArray objects can be attributed to it.
    entity: prov.model.ProvEntity
        An entity representing this object in provenance documents. It is used
        to provide information on the object's own provenance.

    """

    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {}
    NAMESPACE: Tuple[str, str] = ("impurities", "https://ccfe.ukaea.uk")

    def __init__(self, sess: session.Session = session.global_session,
                 **kwargs: Dict[str, Any]):
        """Creates a provenance entity/agent for the reader object.

        """
        self.session = sess
        self.session.prov.add_namespace(self.NAMESPACE[0], self.NAMESPACE[1])
        # TODO: also include library version and, ideally, version of
        # relevent dependency in the hash
        self.prov_id = session.hash_vals(reader_type=self.__class__.__name__,
                                         **kwargs)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.actedOnBehalfOf(self.agent, self.session.agent)
        self.entity = self.session.prov.entity(self.prov_id, kwargs)
        self.session.prov.generation(self.entity, self.session.session,
                                     time=datetime.datetime.now())
        self.session.prov.attribution(self.entity, self.session.agent)

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
        assert "generate_mappers" in result.attrs
        assert "map_to_master" in result.attrs
        assert "map_from_master" in result.attrs
        assert "datatype" in result.attrs
        assert "provenance" in result.attrs
        return result

    def create_provenance(self, key: str, revision: Any,
                          data_objects: Iterable[str],
                          start_time: Optional[datetime.datetime] = None,
                          end_time: Optional[datetime.datetime] = None) -> \
            prov.ProvEntity:
        """Create a provenance entity for the given set of data. This should
        be attached as metadata.

        Note that this method just creates the provenance data
        appropriate for the arguments it has been provided with. It
        does not check that these arguments are actually valid and
        that the provenance corresponds to actually existing data.

        Parameters
        ----------
        key
            Identifies what data was read. Should be present in
            :py:attr:`AVAILABLE_DATA`.
        revision
            Object indicating which version of data should be used.
        data_objects
            Identifiers for the database entries or files which the data was 
            read from.
        start_time
            The time read-in began.
        end_time
            The time read-in was finished (defaults to present time).

        Returns
        -------
        :
            A provenance entity for the newly read-in data.
        """
        if not end_time:
            end_time = datetime.datetime.now()
        entity_id = session.hash_vals(creator=self.prov_id, key=key,
                                      revision=revision, date=end_time)
        attrs = {prov.PROV_TYPE: "DataArray",
                 prov.PROV_VALUE: ",".join(self.AVAILABLE_DATA[key])}
        activity_id = session.hash_vals(agent=self.prov_id, date=end_time)
        activity = self.session.prov.activity(activity_id, start_time,
                                              end_time,
                                              {prov.PROV_TYPE: "ReadData"})
        activity.wasAssociatedWith(self.session.agent)
        activity.wasAssociatedWith(self.agent)
        activity.wasInformedBy(self.session.session)
        entity = self.session.prov.entity(entity_id, attrs)
        entity.wasGeneratedBy(activity, end_time)
        entity.wasAttributedTo(self.session.agent)
        entity.wasAttributedTo(self.agent)
        for data in data_objects:
            # TODO: Find some way to avoid duplicate records
            data_entity = self.prov.entity(self.namespace[0]+data)
            entity.wasDerivedFrom(data_entity)
            activity.used(data_entity)
        return entity

    def authenticate(self, name: str, password: str) -> bool:
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

    ==========  ===============  ==============  =================
    Key         Data type        Data for        Instrument
    ==========  ===============  ==============  =================
    efit_rmag   Major radius     Magnetic axis   EFIT equilibrium
    efit_zmag   Z position       Magnetic axis   EFIT equilibrium
    efit_rsep   Major radius     Separatrix      EFIT equilibrium
    efit_zsep   Z position       Separatrix      EFIT equilibrium
    hrts_ne     Number density   Electrons       HRTS
    hrts_te     Temperature      Electrons       HRTS
    lidr_ne     Number density   Electrons       LIDR
    lidr_te     Temperature      Electrons       LIDR
    ==========  ===============  ==============  =================

    Note that there will need to be some refactoring to support other
    data types. However, **this is guaranteed not to affect the public
    interface**.

    Parameters
    ----------
    pulse : int
        The ID number for the pulse from which to get data.
    uid : str
        The UID for the particular data to be read.
    server : str
        The URL for the SAL server to read data from.

    Attributes
    ----------
    AVAILABLE_DATA: Dict[str, DataType]
        A mapping of the keys used to get each piece of data to the type of
        data associated with that key.
    NAMESPACE: Tuple[str, str]
        The abbreviation and full URL for the PROV namespace of the reader
        class.

    """

    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {
        "efit_rmag": ("major_rad", "mag_axis"),
        "efit_zmag": ("z", "mag_axis"),
        "efit_rsep": ("major_rad", "separatrix_axis"),
        "efit_zsep": ("z", "separatrix_axis"),
        "hrts_ne": ("number_density", "electrons"),
        "hrts_te": ("temperature", "electrons"),
        "lidr_ne": ("number_density", "electrons"),
        "lidr_te": ("temperature", "electrons"),
    }

    _HANDLER_METHODS: ClassVar[Dict[str, str]] = {
        "efit_rmag": "_handle_equilibrium_position",
        "efit_zmag": "_handle_equilibrium_position",
        "efit_rsep": "_handle_equilibrium_position",
        "efit_zsep": "_handle_equilibrium_position",
        "hrts_ne": "_handle_electron_data",
        "hrts_te": "_handle_electron_data",
        "lidr_ne": "_handle_electron_data",
        "lidr_te": "_handle_electron_data",
    }

    def __init__(self, pulse: int, uid: str = "jetppf",
                 server: str = "https://sal.jet.uk",
                 sess: session.Session = session.global_session):
        self.NAMESPACE: Tuple[str, str] = ("jet", server)
        super().__init__(sess, puls=pulse, uid=uid, server=server)
        self.pulse = pulse
        self.uid = uid
        self._client = SALClient(server)

    def _get_data(self, key: str, revision: int = 0) -> DataArray:
        """Reads and returns the data for the given key. Should only be called
        by :py:meth:`DataReader.get`."""
        return getattr(self, self._HANDLER_METHODS[key])(key, revision)

    def _get_signal(self, key: str, revision: int) -> Tuple[Signal, str]:
        """Gets the signal for the given DDA, at the given revision."""
        path = "/pulse/{:i}/ppf/signal/{}/{}:{:i}"
        # TODO: if revision == 0 update it with absolute revision
        # number in path before returning
        return (self._client.get(path.format(self.pulse, self.uid,
                                             key.replace("_", "/"), revision)),
                path)

    def _handle_equilibrium_position(self, key: str,
                                     revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for data relating to position
        of equilibrium."""
        start = datetime.datetime.now()
        signal, uid = self._get_signal(key, revision)
        meta = {"generate_mappers": trivial_factory,
                "map_to_master": None,
                "map_from_master": None,
                "datatype": self.AVAILABLE_DATA[key]}
        data = DataArray(signal.data, [("t", signal.dimensions[0].data)],
                         name=key, attrs=meta)
        data.attrs['provenance'] = self.create_provenance(key, revision,
                                                          [uid], start)
        return data

    def _handle_electron_data(self, key: str, revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for electron temperature or
        number density."""
        start = datetime.datetime.now()
        uids = []
        uid, signal = self._get_signal(key, revision)
        uids.append(uid)
        uid, error = self._get_signal(key[:-2] + "d" + key[-2:], revision)
        uids.append(uid)
        ticks = np.arange(signal.dimensions[1].length)

        r0 = signal.dimensions[1].data
        z, uid = self._get_signal(key[:-2] + "z", revision).data
        uids.append(uid)

        def map_factory(equilibrium: Dataset) -> Tuple[Remapper, Remapper]:
            # Implementation TBC, but use r0 and z.
            r0
            z
            return (None, None)

        meta = {"generate_mappers": map_factory,
                "map_to_master": None,
                "map_from_master": None,
                "datatype": self.AVAILABLE_DATA[key],
                "error": error.data}
        data = DataArray(signal.data, [("t", signal.dimensions[0].data),
                                       (key[:-2] + "coord", ticks)],
                         name=key, attrs=meta)
        data.attrs['provenance'] = self.create_provenance(key, revision, uids,
                                                          start)
        return data

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
