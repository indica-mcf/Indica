"""Experimental design for reading data from disk/database.
"""

import datetime
from numbers import Number
import os
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import DataArray
from xarray import Dataset

from .selectors import choose_on_plot
from .selectors import DataSelector
from ..abstractio import BaseIO
from ..datatypes import ArrayType
from ..session import hash_vals
from ..session import Session
from ..utilities import get_slice_limits
from ..utilities import to_filename

# TODO: Place this in some global location?
CACHE_DIR = ".impurities"


class DataReader(BaseIO):
    """Abstract base class to read data in from a database.

    This defines the interface used by all concrete objects which read
    data from the disc, a database, etc. It is a `context manager
    <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`_
    and can be used in a `with statement
    <https://docs.python.org/3/reference/compound_stmts.html#with>`_.

    Attributes
    ----------
    agent: prov.model.ProvAgent
        An agent representing this object in provenance documents.
        DataArray objects can be attributed to it.
    DIAGNOSTIC_QUANTITIES: Dict[str, Dict[str, Dict[str, Dict[str, ArrayType]]]]
        Hierarchical information on the quantities which are available for
        reading. These are indexed by (in order) diagnostic name, UID,
        instrument name, and quantity name. The values of the innermost
        dictionary describe the physical type of the data to be read.
    entity: prov.model.ProvEntity
        An entity representing this object in provenance documents. It is used
        to provide information on the object's own provenance.
    NAMESPACE: Classvar[Tuple[str, str]]
        The abbreviation and full URL for the PROV namespace of the database
        the class reads from.
    prov_id: str
        The hash used to identify this object in provenance documents.

    """

    DIAGNOSTIC_QUANTITIES: Dict[str, Dict[str, Dict[str, Dict[str, ArrayType]]]] = {}
    _RECORD_TEMPLATE = "{}-{}-{}-{}"
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
        checks valid datatypes have been specified for the available
        data. This should be called by constructors on subtypes.

        Parameters
        ----------
        tstart
            Start of time range for which to get data.
        tend
            End of time range for which to get data.
        max_freq
            Maximum frequency of data-sampling, abov which some osrt of
            averaging or compression may be performed.
        sess
            An object representing the session being run. Contains information
            such as provenance data.
        selector
            A callback which can be used to interactively determine the which
            channels of data can be dropped.
        kwargs
            Any other arguments which should be recorded in the PROV entity for
            the reader.

        """
        self._tstart = tstart
        self._tend = tend
        self._max_freq = max_freq
        self._start_time = None
        self.session = sess
        self._selector = selector
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

            cachefile = self._RECORD_TEMPLATE.format(
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
        return data

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
            A dataset containing the requested physical quantities.

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

            cachefile = self._RECORD_TEMPLATE.format("cxrs", instrument, uid, quantity)
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
        return data

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
            "{} does not implement a '_get_charge_exchange' "
            "method.".format(self.__class__.__name__)
        )

    def get_equilibrium(
        self,
        uid: str,
        calculation: str,
        revision: int = 0,
        quantities: Set[str] = {"ne", "te"},
    ) -> Dataset:
        """Reads equilibrium data.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        calculation
            Name of the code used to calculate this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            TODO!

        Returns
        -------
        :
            A dataset containing the requested physical quantities.

        """

    def _get_equilibrium(
        self, uid: str, calculation: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for equilibrium from the database. Data outside
        the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        calculation
            Name of the code used to calculate this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Options are
            TODO!

        Returns
        -------
        A dictionary containing the following items:

        TODO: Complete the list of quantitites

        """
        raise NotImplementedError(
            "{} does not implement a '_get_equilibrium' "
            "method.".format(self.__class__.__name__)
        )

    def get_cyclotron(self, uid: str, instrument: str, revision: int = 0,) -> Dataset:
        """Reads electron temperature measurements from cyclotron data.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.

        Returns
        -------
        :
            A dataset containing the electron temperature.

        """

    def _get_cyclotron(
        self, uid: str, calculation: str, revision: int = 0,
    ) -> Dict[str, Any]:
        """Gets raw data for cyclotron resonance from the database. Data
        outside the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.

        Returns
        -------
        A dictionary containing the following items:

        length : int
            Number of channels in data
        z : float
            Vertical position of line of sight
        times : ndarray
            The times at which measurements were taken
        te : ndarray
            Electron temperature (first axis is time, second channel)
        te_error : ndarray
            Uncertainty in electron temperature
        te_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for electron temperature.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_cyclotron' "
            "method.".format(self.__class__.__name__)
        )

    def get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: int = 0,
        quantities: Set[str] = {"V", "H"},
    ) -> Dataset:
        """Reads data on radiation flux.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument/DDA which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which cameras to read quantitie(s) from. Options are
            "H", "T", and "V". Not all cameras are available for all DDAs.

        Returns
        -------
        :
            A dataset containing the requested radiation values.
        """

    def _get_radiation(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for radiant fluxes from the database. Data outside
        the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the instrument/DDA which measured this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database.  Options are
            "H", "T", and "V". Not all cameras are available for all DDAs.

        Returns
        -------
        A dictionary containing the following items:

        length : Dict[str, int]
            Number of channels in data for each camera
        times : ndarray
            The times at which measurements were taken
        H : ndarray (optional)
            Brightness from camera H (first axis is time, second channel)
        H_error : ndarray (optional)
            Uncertainty in brightness for camera H.
        H_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed from camera H.
        H_Rstart : ndarray (optional)
            Major radius of start positions for lines of sight from camera H.
        H_Rstop : ndarray (optional)
            Major radius of stop positions for lines of sight from camera H.
        H_zstart : ndarray (optional)
            Vertical location of start positions for lines of sight from
            camera H.
        H_zstop : ndarray (optional)
            Vertical location of stop positions for lines of sight from
            camera H.
        T : ndarray (optional)
            Brightness from camera T (first axis is time, second channel)
        T_error : ndarray (optional)
            Uncertainty in brightness for camera T.
        T_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed from camera T.
        T_Rstart : ndarray (optional)
            Major radius of start positions for lines of sight from camera T.
        T_Rstop : ndarray (optional)
            Major radius of stop positions for lines of sight from camera T.
        T_zstart : ndarray (optional)
            Vertical location of start positions for lines of sight from
            camera T.
        T_zstop : ndarray (optional)
            Vertical location of stop positions for lines of sight from
            camera T.
        V : ndarray (optional)
            Brightness from camera V (first axis is time, second channel)
        V_error : ndarray (optional)
            Uncertainty in brightness for camera V.
        V_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed from camera V.
        V_Rstart : ndarray (optional)
            Major radius of start positions for lines of sight from camera V.
        V_Rstop : ndarray (optional)
            Major radius of stop positions for lines of sight from camera V.
        V_zstart : ndarray (optional)
            Vertical location of start positions for lines of sight from
            camera V.
        V_zstop : ndarray (optional)
            Vertical location of stop positions for lines of sight from
            camera V.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_radiation' "
            "method.".format(self.__class__.__name__)
        )

    def get_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: int = 0,
        quantities: Set[str] = {"H", "V"},
    ) -> Dataset:
        """Reads spectroscopic measurements of effective charge.

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
            Which physical quantitie(s) to read from the database.  Options are
            "H" (horizontal line of sight) and "V" (vertical).

        Returns
        -------
        :
            A dataset containing the requested effective charge data.

        """

    def _get_spectroscopy(
        self, uid: str, calculation: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw spectroscopic data for effective charge from the
        database. Data outside the desired time range will be
        discarded.

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
            Which physical quantitie(s) to read from the database.  Options are
            "H" (horizontal line of sight) and "V" (vertical).

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken
        H : ndarray (optional)
            Effective charge along horizontal line of sight (first axis is
            time).
        H_error : ndarray (optional)
            Uncertainty in horizontal effective charge measurement
        H_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed from horizontal line of sight
        H_Rstart : float (optional)
            Major radius of start positions for horizontal line of sight.
        H_Rstop : float (optional)
            Major radius of stop position for horizontal line of sight from.
        H_zstart : float (optional)
            Vertical location of start position for horizontal line of sight.
        H_zstop : float (optional)
            Vertical location of start position for horizontal lines of sight.
        V : ndarray (optional)
            Effective charge along vertical line of sight (first axis is time).
        V_error : ndarray (optional)
            Uncertainty in vertical effective charge measurement
        V_records : List[str] (optional)
            Representations (e.g., paths) for the records in the database used
            to access data needed from vertical line of sight
        V_Rstart : float (optional)
            Major radius of start positions for vertical line of sight.
        V_Rstop : float (optional)
            Major radius of stop position for vertical line of sight from.
        V_zstart : float (optional)
            Vertical location of start position for vertical line of sight.
        V_zstop : float (optional)
            Vertical location of start position for vertical lines of sight.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_spectroscopy' "
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
