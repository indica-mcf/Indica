"""Experimental design for reading data from disk/database.
"""

from abc import ABC
from abc import abstractmethod
import datetime
from numbers import Number
import os
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import DataArray
from xarray import Dataset

from .selectors import choose_on_plot
from .selectors import DataSelector
from ..datatypes import ArrayType
from ..session import hash_vals
from ..session import Session
from ..utilities import get_slice_limits
from ..utilities import to_filename

# TODO: Place this in som global location?
CACHE_DIR = ".impurities"


class DataReader(ABC):
    """Abstract base class to read data in from a database.

    This defines the interface used by all concrete objects which read
    data from the disc, a database, etc. It is a `context manager
    <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`_
    and can be used in a `with statement
    <https://docs.python.org/3/reference/compound_stmts.html#with>`_.

    Attributes
    ----------
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

    DIAGNOSTIC_QUANTITIES: Dict[str, Dict[str, Dict[str, Dict[str, ArrayType]]]] = {}
    RECORD_TEMPLATE = "{}-{}-{}-{}"
    NAMESPACE: Tuple[str, str] = ("impurities", "https://ccfe.ukaea.uk")

    def __init__(
        self,
        tstart: float,
        tend: float,
        max_freq: float,
        sess: Session,
        selector: DataSelector = choose_on_plot,
        **kwargs: Any
    ):
        """Creates a provenance entity/agent for the reader object. Also
        checks valid datatypes have been specified for the available data.

        """
        self._tstart = tstart
        self._tend = tend
        self._max_freq = max_freq
        self._start_time = None
        self.session = sess
        self._selector = selector
        # TODO: This should be done once in the Session object
        self.session.prov.add_namespace(self.NAMESPACE[0], self.NAMESPACE[1])
        # TODO: also include library version and, ideally, version of
        # relevent dependency in the hash
        prov_attrs: Dict[str, Any] = dict(
            tstart=tstart, tend=tend, max_freq=max_freq, **kwargs
        )
        self.prov_id = hash_vals(reader_type=self.__class__.__name__, **prov_attrs)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.actedOnBehalfOf(self.agent, self.session.agent)
        # TODO: Properly namespace the attributes on this entity.
        self.entity = self.session.prov.entity(self.prov_id, prov_attrs)
        self.session.prov.generation(
            self.entity, self.session.session, time=datetime.datetime.now()
        )
        self.session.prov.attribution(self.entity, self.session.agent)

    def __enter__(self) -> "DataReader":
        """Called at beginning of a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        """Close reader at end of context manager. Don't try to handle
        exceptions."""
        self.close()
        return False

    def get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: int = 0,
        quantities: Set[str] = {"ne", "te"},
    ) -> Dataset:
        """Reads data based on Thomson Scattering.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            "ne" (electron number density) and "te" (electron temperature).

        Returns
        -------
        :
            A dataset containing the requested physical quantities.
        """
        available_quantities = self.DIAGNOSTIC_QUANTITIES["thomson_scattering"][uid][
            instrument
        ]
        database_results = self._get_thomson_scattering(
            uid, instrument, revision, quantities
        )
        ticks = np.arange(database_results["length"])
        diagnostic_coord = instrument + "_coord"
        coords = [("t", database_results["times"]), (diagnostic_coord, ticks)]
        data = {}
        # TODO: Assemble a CoordinateTransform object
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read thomson_scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            cachefile = self.RECORD_TEMPLATE.format(
                "thomson", instrument, uid, quantity
            )
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(database_results[quantity + "_error"], coords),
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                name=instrument + "_" + quantity,
                attrs=meta,
            )
            drop = self._select_channels(cachefile, quant_data, diagnostic_coord)
            quant_data.attrs["provenance"] = self.create_provenance(
                "thompson",
                uid,
                instrument,
                revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            data[quantity] = quant_data.drop_sel({diagnostic_coord: drop})
        return Dataset(data)

    def _get_thomson_scattering(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for Thomson scattering from the database. Data outside
        the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            "ne" (electron number density) and "te" (electron temperature).

        Returns
        -------
        A dictionary containing the following items:

        length : int
            Number of channels in data
        R : ndarray
            Major radius positions for each channel
        z : ndarray
            Vertical position of each channel
        times : ndarray
            The times at which measurements were taken
        ne : ndarray (optional)
            Electron number density (first axis is time, second channel)
        ne_error : ndarray (optional)
            Uncertainty in electron number density
        ne_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed for electron number density.
        te : ndarray (optional)
            Electron temperature (first axis is time, second channel)
        te_error : ndarray (optional)
            Uncertainty in electron temperature
        te_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed for electron temperature.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_thomson_scattering' "
            "method.".format(self.__class__.__name__)
        )

    def get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: int = 0,
        quantities: Set[str] = {"ne", "te"},
    ) -> Dataset:
        """Reads charge exchange data.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            "ne" (electron number density) and "te" (electron temperature).

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.

        """
        available_quantities = self.DIAGNOSTIC_QUANTITIES["charge_exchange"][uid][
            instrument
        ]
        database_results = self._get_charge_exchange(
            uid, instrument, revision, quantities
        )
        ticks = np.arange(database_results["length"])
        diagnostic_coord = instrument + "_coord"
        coords = [("t", database_results["times"]), (diagnostic_coord, ticks)]
        data = {}
        # TODO: Assemble a CoordinateTransform object
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read thomson_scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            cachefile = self.RECORD_TEMPLATE.format("cxrs", instrument, uid, quantity)
            meta = {
                "datatype": available_quantities[quantity],
                "element": database_results["element"],
                "error": DataArray(database_results[quantity + "_error"], coords),
                "exposure_time": available_quantities["texp"],
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                name=instrument + "_" + quantity,
                attrs=meta,
            )
            drop = self._select_channels(cachefile, quant_data, diagnostic_coord)
            quant_data.attrs["provenance"] = self.create_provenance(
                "cxrs",
                uid,
                instrument,
                revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            data[quantity] = quant_data.drop_sel({diagnostic_coord: drop})
        return Dataset(data)

    def _get_charge_exchange(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for charge exchange from the database. Data outside
        the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            "ne" (electron number density) and "te" (electron temperature).

        Returns
        -------
        A dictionary containing the following items:

        length : int
            Number of channels in data
        R : ndarray
            Major radius positions for each channel
        z : ndarray
            Vertical position of each channel
        element : str
            The element this ion data is for
        texp : ndarray
            Exposure times
        times : ndarray
            The times at which measurements were taken
        angf : ndarray (optional)
            Angular frequency of ion (first axis is time, second channel)
        angf_error : ndarray (optional)
            Uncertainty in angular frequency
        angf_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed for angular frequency.
        conc : ndarray (optional)
            Ion concentration (first axis is time, second channel)
        conc_error : ndarray (optional)
            Uncertainty in ion concentration.
        conc_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed for ion concentration.
        ti : ndarray (optional)
            Ion temperature (first axis is time, second channel)
        ti_error : ndarray (optional)
            Uncertainty in ion temperature.
        ti_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed for ion temperature.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_unsafe' "
            "method.".format(self.__class__.__name__)
        )

    def create_provenance(
        self,
        diagnostic: str,
        uid: str,
        instrument: str,
        revision: Optional[int],
        quantity: str,
        data_objects: Iterable[str],
        ignored: Iterable[Number],
    ) -> prov.ProvEntity:
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
        entity_id = hash_vals(
            creator=self.prov_id,
            diagnostic=diagnostic,
            uid=uid,
            instrument=instrument,
            revision=revision,
            quantity=quantity,
            ignored=ignored,
            date=end_time,
        )
        # TODO: properly namespace the data type and ignored channels
        attrs = {
            prov.PROV_TYPE: "DataArray",
            prov.PROV_VALUE: ",".join(
                self.DIAGNOSTIC_QUANTITIES["charge_exchange"][uid][instrument]
            ),
            "ignored_channels": str(ignored),
        }
        activity_id = hash_vals(agent=self.prov_id, date=end_time)
        activity = self.session.prov.activity(
            activity_id, self._start_time, end_time, {prov.PROV_TYPE: "ReadData"},
        )
        activity.wasAssociatedWith(self.session.agent)
        activity.wasAssociatedWith(self.agent)
        activity.wasInformedBy(self.session.session)
        entity = self.session.prov.entity(entity_id, attrs)
        entity.wasGeneratedBy(activity, end_time)
        entity.wasAttributedTo(self.session.agent)
        entity.wasAttributedTo(self.agent)
        for data in data_objects:
            # TODO: Find some way to avoid duplicate records
            data_entity = self.session.prov.entity(self.NAMESPACE[0] + ":" + data)
            entity.wasDerivedFrom(data_entity)
            activity.used(data_entity)
        return entity

    def _select_channels(
        self,
        cache_key: str,
        data: DataArray,
        channel_dim: str,
        bad_channels: Collection[Number] = [],
    ) -> Iterable[Number]:
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
        cache_file = os.path.expanduser(
            os.path.join("~", CACHE_DIR, self.__class__.__name__, cache_name)
        )
        os.makedirs(os.path.dirname(cache_file), 0o755, exist_ok=True)
        if os.path.exists(cache_file):
            cached_vals = np.loadtxt(cache_file, dtype=data.coords[channel_dim].dtype)
        else:
            cached_vals = []
        ignored = self._selector(data, channel_dim, bad_channels, cached_vals)
        np.savetxt(cache_file, ignored)
        return ignored

    def _set_times_item(
        self, results: Dict[str, Any], times: np.ndarray, nstart: int, nend: int,
    ) -> Tuple[int, int]:
        """Add the "times" data to the dictionary, if not already
        present. Also return the upper and lower limits required based
        on the start and end times desired.

        """
        if "times" not in results:
            nstart, nend = get_slice_limits(self._tstart, self._tend, times)
            results["times"] = times[nstart, nend].copy()
        return nstart, nend

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
            True if authenticationis needed, otherwise false.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'requires_authentication' "
            "property.".format(self.__class__.__name__)
        )

    @abstractmethod
    def close(self) -> None:
        """Closes connection to whatever backend (file, database, server,
        etc.) from which data is being read."""
        raise NotImplementedError(
            "{} does not implement a 'close' " "method.".format(self.__class__.__name__)
        )
