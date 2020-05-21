"""Experimental design for reading data from disk/database.
"""

from abc import ABC, abstractmethod
import datetime
from numbers import Number
import os
from typing import Any, ClassVar, Container, Dict, Iterable, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d
import prov.model as prov
from xarray import DataArray

import datatypes
import session
from .selector import choose_on_plot, DataSelector
from ..utilities import to_filename

# TODO: Place this in som global location?
CACHE_DIR = ".impurities"
MIN_TIME_BIN = 5
MAX_TIME_FACTOR = 10

class DataReader(ABC):
    """Abstract base class to read data in from a database.

    This defines the interface used by all concrete objects which read
    data from the disc, a database, etc. It is a `context manager
    <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`_
    and can be used in a `with statement
    <https://docs.python.org/3/reference/compound_stmts.html#with>`_.

    Attributes
    ----------
    AVAILABLE_DATA: Dict[str, datatypes.DataType]
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

    AVAILABLE_DATA: ClassVar[Dict[str, datatypes.DataType]] = {}
    NAMESPACE: Tuple[str, str] = ("impurities", "https://ccfe.ukaea.uk")

    def __init__(self, tstart: float, tend: float, interval: float,
                 sess: session.Session = session.global_session,
                 selector: DataSelector = choose_on_plot,
                 **kwargs: Dict[str, Any]):
        """Creates a provenance entity/agent for the reader object. Also
        checks valid datatypes have been specified for the available data.

        """
        self._tstart = tstart
        self._tend = tend
        self._interval = interval
        self._start_time = None
        self.session = sess
        self._selector = selector
        self.session.prov.add_namespace(self.NAMESPACE[0], self.NAMESPACE[1])
        # TODO: also include library version and, ideally, version of
        # relevent dependency in the hash
        self.prov_id = session.hash_vals(reader_type=self.__class__.__name__,
                                         **kwargs)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.actedOnBehalfOf(self.agent, self.session.agent)
        # TODO: Properly namespace the attributes on this entity.
        self.entity = self.session.prov.entity(self.prov_id,
                                               {"tstart": tstart, "tend": tend,
                                                "interval": interval} + kwargs)
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
        self._start_time = datetime.datetime.now()
        result = self._get_data(key, revision)
        # Check the result has appropriate metadata
        assert "generate_mappers" in result.attrs
        assert "map_to_master" in result.attrs
        assert "map_from_master" in result.attrs
        assert "datatype" in result.attrs
        dt = result.attrs["datatype"]
        expected_dt = self.AVAILABLE_DATA[key]
        assert dt[0] == expected_dt[0]
        if dt[0] not in datatypes.GENERAL_DATATYPES:
            warn("DataReader class {} returns unrecognised general "
                 "datatype '{}' for  key '{}'".format(
                     self.__class__.__name__, dt[0], key),
                 datatypes.DatatypeWarning)
        if expected_dt[1]:
            assert dt[1] == expected_dt[1]
        if dt[1] not in datatypes.SPECIFIC_DATATYPES:
            warn("DataReader class {} returns unrecognised specific "
                 "datatype '{}' for key '{}'".format(
                     self.__class__.__name__, dt[1], key),
                 datatypes.DatatypeWarning)
        assert "provenance" in result.attrs
        return result

    def create_provenance(self, key: str, revision: Any,
                          data_objects: Iterable[str],
                          ignored: Iterable[Number]) -> prov.ProvEntity:
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
        ignored
            A list of channels which were ignored/dropped from the data.

        Returns
        -------
        :
            A provenance entity for the newly read-in data.
        """
        end_time = datetime.datetime.now()
        entity_id = session.hash_vals(creator=self.prov_id, key=key,
                                      revision=revision, ignored=ignored,
                                      date=end_time)
        # TODO: properly namespace the data type and ignored channels
        attrs = {prov.PROV_TYPE: "DataArray",
                 prov.PROV_VALUE: ",".join(self.AVAILABLE_DATA[key]),
                 'ignored_channels': ",".join(ignored)}
        activity_id = session.hash_vals(agent=self.prov_id, date=end_time)
        activity = self.session.prov.activity(activity_id, self._start_time,
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

    @abstractmethod
    def _select_channels(self, cache_key: str, data: DataArray, channel_dim:
                         str, bad_channels: Container[Number]) \
            -> Iterable[Number]:
        """Allows the user to select which channels should be read and which
        should be discarded, using whichever method was specified when
        the reader was constructed.

        This method will check whether channels have previously been
        selected for this particular data and load them if so. The
        user will then be given a chance to modify this selection. The
        user's choices will then be cached for reuse later,
        overwriting any existing records which were loaded.

        Parameters
        ----------
        cache_key:
            Name of file from which to load a user's previous selection and to
            which to save the results of this selection.
        data:
            The data from which channels should be selected to discard.
        channel_dim:
            The name of the dimension used for storing separate channels. This
            will be used for the x-axis in the plot.
        bad_channels:
            A (possibly empty) list of channel labels which are known to be
            incorrectly calibrated, faulty, or otherwise untrustworty. These
            will be plotted in red, but must still be specifically selected by
            the user to be discared.

        Returns
        -------
        :
            A list of channel labels which the user has selected to be
            discarded.

        """
        cache_name = to_filename(cache_key)
        cache_file = os.path.expanduser(os.path.join("~", CACHE_DIR,
                                                     self.__class__.name,
                                                     cache_name))
        os.makedirs(os.path.dirname(cache_file), 0o755, exist_ok=True)
        if os.path.exists(cache_file):
            cached_vals = np.loadtxt(cache_file,
                                     dtype=data.coords[channel_dim].dtype)
        else:
            cached_vals = []
        ignored = self._selector(data, channel_dim, bad_channels, cached_vals)
        np.savetxt(cache_file, ignored)
        return ignored

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
        -------
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
