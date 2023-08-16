"""Experimental design for reading data from disk/database.
"""

from copy import deepcopy
import datetime
from numbers import Number
import os
from typing import Any
from typing import Collection
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple
import warnings

import numpy as np
import prov.model as prov
from xarray import DataArray

from .available_quantities import AVAILABLE_QUANTITIES
from .selectors import choose_on_plot
from .selectors import DataSelector
from ..abstractio import BaseIO
from ..converters import FluxSurfaceCoordinates
from ..converters import LinesOfSightTransform
from ..converters import MagneticCoordinates
from ..converters import TransectCoordinates
from ..converters import TrivialTransform
from ..datatypes import ArrayType
from ..numpy_typing import ArrayLike
from ..numpy_typing import RevisionLike
from ..session import hash_vals
from ..session import Session
from ..utilities import to_filename

# TODO: Place this in some global location?
CACHE_DIR = ".indica"


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
    INSTRUMENT_METHODS: Dict[str, str]
        Mapping between instrument (DDA in JET) names and method to use to assemble that
        data. Implementation-specific.
    entity: prov.model.ProvEntity
        An entity representing this object in provenance documents. It is used
        to provide information on the object's own provenance.
    NAMESPACE: Classvar[Tuple[str, str]]
        The abbreviation and full URL for the PROV namespace of the database
        the class reads from.
    prov_id: str
        The hash used to identify this object in provenance documents.

    """

    INSTRUMENT_METHODS: Dict[str, str] = {}
    _AVAILABLE_QUANTITIES = AVAILABLE_QUANTITIES
    _IMPLEMENTATION_QUANTITIES: Dict[str, Dict[str, ArrayType]] = {}

    _RECORD_TEMPLATE = "{}-{}-{}-{}-{}"
    NAMESPACE: Tuple[str, str] = ("impurities", "https://ccfe.ukaea.uk")

    def __init__(
        self,
        tstart: float,
        tend: float,
        max_freq: float,
        sess: Session,
        selector: DataSelector = choose_on_plot,
        **kwargs: Any,
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
            Maximum frequency of data-sampling, above which some sort of
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
        self._reader_cache_id: str
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

    def get(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = 0,
        quantities: Set[str] = set(),
    ) -> Dict[str, DataArray]:
        """Reads data for the requested instrument. In general this will be
        the method you want to use when reading.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            The instrument which measured this data (DDA at JET)
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database. Defaults to
            all available quantities for that instrument.

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.
        """
        if instrument not in self.INSTRUMENT_METHODS:
            raise ValueError(
                "{} does not support reading for instrument {}".format(
                    self.__class__.__name__, instrument
                )
            )
        method = getattr(self, self.INSTRUMENT_METHODS[instrument])
        if not quantities:
            quantities = set(self.available_quantities(instrument))
        return method(uid, instrument, revision, quantities)

    def get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
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
            Which physical quantitie(s) to read from the database..

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_thomson_scattering(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        ticks = np.arange(database_results["length"])
        diagnostic_coord = instrument + "_coord"
        times = database_results["times"]
        R = database_results["R"]
        z = database_results["z"]
        R_coord = DataArray(R, coords=[(diagnostic_coord, ticks)])
        z_coord = DataArray(z, coords=[(diagnostic_coord, ticks)])
        transform = TransectCoordinates(R_coord, z_coord)
        coords: Dict[Hashable, ArrayLike] = {
            "t": times,
            diagnostic_coord: ticks,
            transform.x2_name: 0,
            "R": R_coord,
            "z": z_coord,
        }
        dims = ["t", diagnostic_coord]
        data = {}
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read Thomson scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            drop = self._select_channels(
                "thomson", uid, instrument, quantity, quant_data, transform.x1_name
            )
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "thomson_scattering",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)
        return data

    def _get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
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
            Which physical quantitie(s) to read from the database.

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

        For each quantity requested there will also be the items:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_thomson_scattering' "
            "method.".format(self.__class__.__name__)
        )

    def get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_charge_exchange(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        ticks = np.arange(database_results["length"])
        diagnostic_coord = instrument + "_coord"
        data = {}
        # needs change (see GET_RADIATION) - but must still work for other readers
        R_coord = DataArray(database_results["R"], coords=[(diagnostic_coord, ticks)])
        z_coord = DataArray(database_results["z"], coords=[(diagnostic_coord, ticks)])
        transform = TransectCoordinates(R_coord, z_coord)
        times = database_results["times"]
        coords: Dict[Hashable, Any] = {
            "t": times,
            diagnostic_coord: ticks,
            transform.x2_name: 0,
            "R": R_coord,
            "z": z_coord,
        }
        dims = ["t", diagnostic_coord]
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        # TODO: why use ffill as method??? Temporarily removed...
        texp = DataArray(
            database_results["texp"], coords=[("t", times)]
        ).indica.inclusive_timeslice(self._tstart, self._tend)
        if downsample_ratio > 1:
            # Seems to be some sort of bug setting the coordinate when
            # coarsening a 1-D array
            new_t = texp.t.coarsen(t=downsample_ratio, boundary="trim").mean()
            texp = (
                texp.coarsen(t=downsample_ratio, boundary="trim")
                .mean()
                .assign_coords(t=new_t)
            )

        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read thomson_scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            meta = {
                "datatype": (
                    available_quantities[quantity][0],
                    database_results["element"],
                ),
                "transform": transform,
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "exposure_time": texp,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            drop = self._select_channels(
                "cxrs", uid, instrument, quantity, quant_data, diagnostic_coord
            )
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "cxrs",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.drop_sel({diagnostic_coord: drop})

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        length : int
            Number of channels in data
        R : ndarrays
            Major radius positions for each channel
        z : ndarray
            Vertical position of each channel
        element : str
            The element this ion data is for
        texp : ndarray
            Exposure times
        times : ndarray
            The times at which measurements were taken

        For each quantity requested there will also be the items:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_charge_exchange' "
            "method.".format(self.__class__.__name__)
        )

    def get_equilibrium(
        self, uid: str, instrument: str, revision: RevisionLike, quantities: Set[str]
    ) -> Dict[str, DataArray]:
        """Reads equilibrium data.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the code used to calculate this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.

        """
        global_quantities = {
            "psi",
            "rmag",
            "zmag",
            "rgeo",
            "faxs",
            "fbnd",
            "ipla",
            "wp",
        }
        separatrix_quantities = {"rbnd", "zbnd"}
        flux_quantities = {"f", "ftor", "vjac", "rmji", "rmjo"}
        available_quantities = self.available_quantities(instrument)
        if len({"rmji", "rmjo"} & quantities) > 0:
            quantities.add("zmag")
        if "faxs" in quantities:
            quantities |= {"rmag", "zmag"}
        database_results = self._get_equilibrium(uid, instrument, revision, quantities)
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        diagnostic_coord = "rho_poloidal"
        times = database_results["times"]
        times_unique, ind_unique = np.unique(times, return_index=True)
        downsample_ratio = int(
            np.ceil(
                (len(times_unique) - 1)
                / (times_unique[-1] - times_unique[0])
                / self._max_freq
            )
        )
        coords_1d: Dict[Hashable, ArrayLike] = {"t": times}
        dims_1d = ("t",)
        trivial_transform = TrivialTransform()
        if len(flux_quantities & quantities) > 0:
            rho = np.sqrt(database_results["psin"])
            coords_2d: Dict[Hashable, ArrayLike] = {"t": times, diagnostic_coord: rho}
        else:
            rho = None
            coords_2d = {}
        flux_transform = FluxSurfaceCoordinates(
            "poloidal",
        )
        dims_2d = ("t", diagnostic_coord)
        if len(separatrix_quantities & quantities):
            coords_sep: Dict[Hashable, ArrayLike] = {"t": times}
        else:
            coords_sep = {}
        dims_sep = ("t", "arbitrary_index")
        if "psi" in quantities:
            coords_3d: Dict[Hashable, ArrayLike] = {
                "t": database_results["times"],
                "R": database_results["psi_r"],
                "z": database_results["psi_z"],
            }
        else:
            coords_3d = {}
        dims_3d = ("t", "z", "R")
        data: Dict[str, DataArray] = {}
        sorted_quantities = sorted(quantities)
        if "rmag" in quantities:
            sorted_quantities.remove("rmag")
            sorted_quantities.insert(0, "rmag")
        if "zmag" in quantities:
            sorted_quantities.remove("zmag")
            sorted_quantities.insert(0, "zmag")
        # Get rmag, zmag if need any of rmji, rmjo, faxs
        for quantity in sorted_quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read thomson_scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )
            meta = {
                "datatype": available_quantities[quantity],
                "transform": trivial_transform
                if quantity in global_quantities | separatrix_quantities
                else flux_transform,
            }
            dims: Tuple[str, ...]
            if quantity == "psi":
                coords = coords_3d
                dims = dims_3d
            elif quantity in global_quantities:
                coords = coords_1d
                dims = dims_1d
            elif quantity in separatrix_quantities:
                coords = coords_sep
                dims = dims_sep
            else:
                coords = coords_2d
                dims = dims_2d
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)

            if len(times) != len(times_unique):
                print(
                    """Equilibrium time axis does not have
                    unique elements...correcting..."""
                )
                quant_data = quant_data.isel(t=ind_unique)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "equilibrium",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                [],
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            if quantity in {"rmji", "rmjo"}:
                quant_data.coords["z"] = data["zmag"]
            elif quantity == "faxs":
                quant_data.coords["R"] = data["rmag"]
                quant_data.coords["z"] = data["zmag"]

            data[quantity] = quant_data

        return data

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for equilibrium from the database. Data outside
        the desired time range will be discarded.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data)
        instrument
            Name of the code used to calculate this data
        revision
            An object (of implementation-dependent type) specifying what
            version of data to get. Default is the most recent.
        quantities
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            Times at which data is sampled.

        For each quantity requested there will also be items

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.

        When ``psi`` is requested, the following will be present as well:

        psi_r : ndarray (optional)
            Major radii at which psi is given
        psi_z : ndarray (optional)
            Vertical positions at which psi is given

        When at least one of "f", "ftor", or "vjac" is requested then
        the results will also include:

        psin : ndarray
            Normalised poloidal flux locations at which data is sampled.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_equilibrium' "
            "method.".format(self.__class__.__name__)
        )

    def get_cyclotron_emissions(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
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
        quantities
            Which physical quantitie(s) to fetch the data for.

        Returns
        -------
        :
            A dictionary containing the electron temperature.

        """
        available_quantities = self.available_quantities(instrument)
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read cyclotron emission data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

        database_results = self._get_cyclotron_emissions(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
        transform = MagneticCoordinates(
            database_results["z"], instrument, database_results["machine_dims"]
        )
        coords: Dict[Hashable, ArrayLike] = {
            "t": times,
            transform.x1_name: database_results["Btot"],
            transform.x2_name: 0,
            "z": database_results["z"],
        }
        dims = ["t", transform.x1_name]
        data = {}
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        for quantity in quantities:
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            drop = self._select_channels(
                "cyclotron",
                uid,
                instrument,
                quantity,
                quant_data,
                transform.x1_name,
                database_results["bad_channels"],
            )
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "cyclotron_emissions",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_cyclotron_emissions(
        self, uid: str, instrument: str, revision: RevisionLike, quantities: Set[str]
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
        quantities
            Which physical quantitie(s) to fetch the data for.

        Returns
        -------
        A dictionary containing the following items:

        length : int
            Number of channels in data.
        z : float
            Vertical position of line of sight
        Btot : ndarray
            The magnetic field strengths at which measurements were taken
        times : ndarray
            The times at which measurements were taken
        bad_channels : List[float]
            Btot values for channels which have not been properly calibrated.
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_cyclotron' "
            "method.".format(self.__class__.__name__)
        )

    def get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """Reads data on irradiance.

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
            Which cameras to read quantitie(s) from.

        Returns
        -------
        :
            A dictionary containing the requested radiation values.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_radiation(uid, instrument, revision, quantities)
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        data = {}
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read radiation data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )

            times = database_results[quantity + "_times"]
            downsample_ratio = int(
                np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
            )
            transform = LinesOfSightTransform(
                x_start=database_results[quantity + "_xstart"],
                z_start=database_results[quantity + "_zstart"],
                y_start=database_results[quantity + "_ystart"],
                x_end=database_results[quantity + "_xstop"],
                z_end=database_results[quantity + "_zstop"],
                y_end=database_results[quantity + "_ystop"],
                name=f"{instrument}_{quantity}",
                machine_dimensions=database_results["machine_dims"],
            )
            # print(transform.x1_name, quantity)
            coords = [
                ("t", times),
                (transform.x1_name, np.arange(database_results["length"][quantity])),
            ]
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords
                ).indica.inclusive_timeslice(
                    self._tstart,
                    self._tend,
                ),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            drop = self._select_channels(
                "radiation", uid, instrument, quantity, quant_data, transform.x1_name
            )
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "radiation",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for irradiance from the database. Data outside
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        length : Dict[str, int]
            Number of channels in data for each camera
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_times : ndarray
            The times at which measurements were taken
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_xstart : ndarray
            x value of start positions for lines of sight for this data.
        <quantity>_xstop : ndarray
            x value of stop positions for lines of sight for this data.
        <quantity>_ystart : ndarray
            y value of start positions for lines of sight for this data.
        <quantity>_ystop : ndarray
            y value of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_radiation' "
            "method.".format(self.__class__.__name__)
        )

    def get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested effective charge data.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_bremsstrahlung_spectroscopy(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
        data = {}
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read bremsstrahlung data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )
            downsample_ratio = int(
                np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
            )
            transform = LinesOfSightTransform(
                x_start=database_results[quantity + "_xstart"],
                z_start=database_results[quantity + "_zstart"],
                y_start=database_results[quantity + "_ystart"],
                x_end=database_results[quantity + "_xstop"],
                z_end=database_results[quantity + "_zstop"],
                y_end=database_results[quantity + "_ystop"],
                name=f"{instrument}_{quantity}",
                machine_dimensions=database_results["machine_dims"],
            )
            coords: Dict[Hashable, Any] = {"t": times}
            dims = ["t"]
            if database_results["length"][quantity] > 1:
                dims.append(transform.x1_name)
                coords[transform.x1_name] = np.arange(
                    database_results["length"][quantity]
                )
            else:
                coords[transform.x1_name] = 0
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            if len(database_results[quantity + "_xstart"]) > 1:
                drop = self._select_channels(
                    "bremsstrahlung",
                    uid,
                    instrument,
                    quantity,
                    quant_data,
                    transform.x1_name,
                )
            else:
                drop = []
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "bremsstrahlung_spectroscopy",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_xstart : ndarray
            x value of start positions for lines of sight for this data.
        <quantity>_xstop : ndarray
            x value of stop positions for lines of sight for this data.
        <quantity>_ystart : ndarray
            y value of start positions for lines of sight for this data.
        <quantity>_ystop : ndarray
            y value of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    def get_vuv_w_analyser(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_vuv_w_analyser(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
        data: Dict[str, DataArray] = {}
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read VUV W analyser data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )
            downsample_ratio = int(
                np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
            )
            transform = LinesOfSightTransform(
                x_start=database_results[quantity + "_xstart"],
                z_start=database_results[quantity + "_zstart"],
                y_start=database_results[quantity + "_ystart"],
                x_end=database_results[quantity + "_xstop"],
                z_end=database_results[quantity + "_zstop"],
                y_end=database_results[quantity + "_ystop"],
                name=f"{instrument}_{quantity}",
                machine_dimensions=database_results["machine_dims"],
            )
            coords: Dict[Hashable, Any] = {"t": times}
            dims = ["t"]
            coords[transform.x1_name] = 0
            meta = {
                "datatype": available_quantities[quantity],
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()

            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "vuv_w_analyser",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                [],
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_vuv_w_analyser(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "{} does not implement a '_get_vuv_w_analyser' "
            "method.".format(self.__class__.__name__)
        )

    def get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """Reads spectroscopic measurements of He-like emission.

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested data.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_helike_spectroscopy(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
        wavelength = database_results["wavelength"]
        transform = LinesOfSightTransform(
            x_start=database_results["xstart"],
            z_start=database_results["zstart"],
            y_start=database_results["ystart"],
            x_end=database_results["xstop"],
            z_end=database_results["zstop"],
            y_end=database_results["ystop"],
            name=f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        coords_1d: Dict[Hashable, Any] = {"t": times}
        dims_1d: list = ["t"]
        if database_results["length"] > 1:
            dims_1d.append(transform.x1_name)
            coords_1d[transform.x1_name] = np.arange(database_results["length"])
        else:
            coords_1d[transform.x1_name] = 0
        dims_spectra = deepcopy(dims_1d)
        dims_spectra.append("wavelength")
        coords_spectra = deepcopy(coords_1d)
        coords_spectra["wavelength"] = wavelength

        data: dict = {}
        drop: list = []
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read He-like spectroscopy data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )
            meta = {
                "datatype": available_quantities[quantity],
                "transform": transform,
            }

            quantity_error = quantity + "_error"
            if quantity == "spectra":
                dims = dims_spectra
                coords = coords_spectra
            else:
                dims = dims_1d
                coords = coords_1d

            if quantity_error in database_results.keys():
                meta["error"] = DataArray(
                    database_results[quantity_error], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend)
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                if quantity_error in database_results.keys():
                    quant_data.attrs["error"] = np.sqrt(
                        (quant_data.attrs["error"] ** 2)
                        .coarsen(t=downsample_ratio, boundary="trim")
                        .mean()
                        / downsample_ratio
                    )
            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "helike_spectroscopy",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Reads spectroscopic measurements of He-like emission.

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_xstart : ndarray
            x value of start positions for lines of sight for this data.
        <quantity>_xstop : ndarray
            x value of stop positions for lines of sight for this data.
        <quantity>_ystart : ndarray
            y value of start positions for lines of sight for this data.
        <quantity>_ystop : ndarray
            y value of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_helike_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    def get_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """Reads filtered radiation diodes

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested data.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_filters(uid, instrument, revision, quantities)
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
        transform = LinesOfSightTransform(
            x_start=database_results["xstart"],
            z_start=database_results["zstart"],
            y_start=database_results["ystart"],
            x_end=database_results["xstop"],
            z_end=database_results["zstop"],
            y_end=database_results["ystop"],
            name=f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        coords: Dict[Hashable, Any] = {"t": times}
        dims = ["t"]
        if database_results["length"] > 1:
            dims.append(transform.x1_name)
            coords[transform.x1_name] = np.arange(database_results["length"])
        else:
            coords[transform.x1_name] = 0

        data: dict = {}
        drop: list = []
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read filtered diode data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "filters",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)
        return data

    def _get_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Reads filtered radiation diodes

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_xstart : ndarray
            x value of start positions for lines of sight for this data.
        <quantity>_xstop : ndarray
            x value of stop positions for lines of sight for this data.
        <quantity>_ystart : ndarray
            y value of start positions for lines of sight for this data.
        <quantity>_ystop : ndarray
            y value of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_filters' "
            "method.".format(self.__class__.__name__)
        )

    def get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """Reads interferometer electron density.

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested data.

        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_interferometry(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        if len(database_results) == 0:
            return database_results

        times = database_results["times"]
        transform = LinesOfSightTransform(
            x_start=database_results["xstart"],
            z_start=database_results["zstart"],
            y_start=database_results["ystart"],
            x_end=database_results["xstop"],
            z_end=database_results["zstop"],
            y_end=database_results["ystop"],
            name=f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        coords: Dict[Hashable, Any] = {"t": times}
        dims = ["t"]
        if database_results["length"] > 1:
            dims.append(transform.x1_name)
            coords[transform.x1_name] = np.arange(database_results["length"])
        else:
            coords[transform.x1_name] = 0

        data: dict = {}
        drop: list = []
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read interferometry data for quantity {}".format(
                        self.__class__.__name__, quantity
                    )
                )
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(
                    database_results[quantity + "_error"], coords, dims
                ).indica.inclusive_timeslice(self._tstart, self._tend),
                "transform": transform,
            }

            quant_data = DataArray(
                database_results[quantity],
                coords,
                dims,
                attrs=meta,
            ).indica.inclusive_timeslice(self._tstart, self._tend)
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim"
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim")
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "interferometry",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
            data[quantity] = quant_data.indica.ignore_data(drop, transform.x1_name)

        self._warn_less_than_zero(instrument, data)
        return data

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Reads interferometer electron density

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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken
        machine_dims
            A tuple describing the size of the Tokamak domain. It should have
            the form ``((Rmin, Rmax), (zmin, zmax))``.

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_xstart : ndarray
            x value of start positions for lines of sight for this data.
        <quantity>_xstop : ndarray
            x value of stop positions for lines of sight for this data.
        <quantity>_ystart : ndarray
            y value of start positions for lines of sight for this data.
        <quantity>_ystop : ndarray
            y value of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    # TODO: method reading ASTRA data works, but requires tests to be written
    # Astra reader is currently working but not tested, code will be uncommented
    # when tests have been written
    #
    # def get_astra(
    #     self, uid: str, instrument: str, revision: RevisionLike, quantities: Set[str]
    # ) -> Dict[str, DataArray]:
    #     """Reads ASTRA data.
    #
    #     Parameters
    #     ----------
    #     uid
    #         User ID (i.e., which user created this data)
    #     instrument
    #         Name of the code used to calculate this data
    #     revision
    #         An object (of implementation-dependent type) specifying what
    #         version of data to get. Default is the most recent.
    #     quantities
    #         Which physical quantitie(s) to read from the database.
    #
    #     Returns
    #     -------
    #     :
    #         A dictionary containing the requested physical quantities.
    #
    #     """
    #     available_quantities = self.available_quantities(instrument)
    #     database_results = self._get_astra(uid, instrument, revision, quantities)
    #     if len(database_results) == 0:
    #         print(f"No data from {uid}.{instrument}:{revision}")
    #         return database_results
    #     _revision = database_results["revision"]
    #
    #     data: Dict[str, DataArray] = {}
    #
    #     # Reorganise coordinate system to match Indica default rho-poloidal
    #     rhop_psin = np.sqrt(database_results["psin"])
    #     rhop_interp = np.linspace(0, 1.0, 65)
    #     rhot_astra = database_results["rho"] / np.max(database_results["rho"])
    #     rhot_rhop = []
    #     for it in range(len(database_results["times"])):
    #         ftor_tmp = database_results["ftor"][it, :]
    #         psi_tmp = database_results["psi"][it, :]
    #         rhot_tmp = np.sqrt(ftor_tmp / ftor_tmp[-1])
    #         rhop_tmp = np.sqrt((psi_tmp - psi_tmp[0]) / (psi_tmp[-1] - psi_tmp[0]))
    #         rhot_xpsn = np.interp(rhop_interp, rhop_tmp, rhot_tmp)
    #         rhot_rhop.append(rhot_xpsn)
    #
    #     rhot_rhop = DataArray(
    #         np.array(rhot_rhop),
    #         {"t": database_results["times"], "rho_poloidal": rhop_interp},
    #         dims=["t", "rho_poloidal"],
    #     ).indica.inclusive_timeslice(self._tstart, self._tend)
    #
    #     radial_coords = {"rho_toroidal": rhot_astra, "rho_poloidal": rhop_psin}
    #
    #     sorted_quantities = sorted(quantities)
    #     for quantity in sorted_quantities:
    #         if quantity not in available_quantities:
    #             raise ValueError(
    #                 "{} can not read astra data for "
    #                 "quantity {}".format(self.__class__.__name__, quantity)
    #             )
    #
    #         if "PROFILES.ASTRA" in database_results[f"{quantity}_records"][0]:
    #             name_coord = "rho_toroidal"
    #         elif "PROFILES.PSI_NORM" in database_results[f"{quantity}_records"][0]:
    #             name_coord = "rho_poloidal"
    #         else:
    #             name_coord = ""
    #
    #         coords = {"t": database_results["times"]}
    #         dims = ["t"]
    #         if len(name_coord) > 0:
    #             coords[name_coord] = radial_coords[name_coord]
    #             dims.append(name_coord)
    #
    #         trivial_transform = TrivialTransform()
    #         meta = {
    #             "datatype": available_quantities[quantity],
    #             "transform": trivial_transform,
    #         }
    #
    #         quant_data = DataArray(
    #             database_results[quantity],
    #             coords,
    #             dims,
    #             attrs=meta,
    #         ).indica.inclusive_timeslice(self._tstart, self._tend)
    #
    #         # TODO: careful with interpolation on new rho_poloidal array...
    #         # Interpolate ASTRA profiles on new rhop_interp array
    #         # Interpolate PSI_NORM profiles on same coordinate system
    #         if name_coord == "rho_toroidal":
    #             rho_toroidal_0 = quant_data.rho_toroidal.min()
    #             quant_interp = quant_data.interp(rho_toroidal=rhot_rhop).drop(
    #                 "rho_toroidal"
    #             )
    #             quant_interp.loc[dict(rho_poloidal=0)] = quant_data.sel(
    #                 rho_toroidal=rho_toroidal_0
    #             )
    #             quant_data = quant_interp.interpolate_na("rho_poloidal")
    #         elif name_coord == "rho_poloidal":
    #             quant_data = quant_data.interp(rho_poloidal=rhop_interp)
    #
    #         quant_data.name = instrument + "_" + quantity
    #         quant_data.attrs["partial_provenance"] = self.create_provenance(
    #             "astra",
    #             uid,
    #             instrument,
    #             _revision,
    #             quantity,
    #             database_results[quantity + "_records"],
    #             [],
    #         )
    #
    #         quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
    #
    #         data[quantity] = quant_data
    #
    #     return data
    #
    # def _get_astra(
    #     self,
    #     uid: str,
    #     instrument: str,
    #     revision: RevisionLike,
    #     quantities: Set[str],
    # ) -> Dict[str, Any]:
    #     """Reads ASTRA data
    #
    #     Parameters
    #     ----------
    #     uid
    #         User ID (i.e., which user created this data)
    #     instrument
    #         Name of the instrument which measured this data
    #     revision
    #         An object (of implementation-dependent type) specifying what
    #         version of data to get. Default is the most recent.
    #     quantities
    #         Which physical quantitie(s) to read from the database.
    #
    #     Returns
    #     -------
    #     A dictionary containing the following items:
    #
    #     times : ndarray
    #         The times at which measurements were taken
    #     machine_dims
    #         A tuple describing the size of the Tokamak domain. It should have
    #         the form ``((Rmin, Rmax), (zmin, zmax))``.
    #
    #     For each requested quantity, the following items will also be present:
    #
    #     <quantity> : ndarray
    #         The data itself (first axis is time, second channel)
    #     <quantity>_records : List[str]
    #         Representations (e.g., paths) for the records in the database used
    #         to access data needed for this data.
    #
    #     """
    #     raise NotImplementedError(
    #         "{} does not implement a '_get_spectroscopy' "
    #         "method.".format(self.__class__.__name__)
    #     )

    def create_provenance(
        self,
        diagnostic: str,
        uid: str,
        instrument: str,
        revision: RevisionLike,
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
                str(s) for s in self.available_quantities(instrument)[quantity]
            ),
            "ignored_channels": str(ignored),
            "uid": uid,
            "instrument": instrument,
            "diagnostic": diagnostic,
            "revision": revision,
            "quantity": quantity,
        }
        activity_id = hash_vals(agent=self.prov_id, date=end_time)
        activity = self.session.prov.activity(
            activity_id,
            self._start_time,
            end_time,
            {prov.PROV_TYPE: "ReadData"},
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
        category: str,
        uid: str,
        instrument: str,
        quantity: str,
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
        user's choices will be cached for reuse later, overwriting any
        existing records which were loaded.

        Parameters
        ----------
        category:
            type of data being fetched (based on name of the reader method used).
        uid
            User ID (i.e., which user created this data).
        instrument
            Name of the instrument which measured this data.
        quantities
            Which physical quantity this data represents.
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
        cache_key = self._RECORD_TEMPLATE.format(
            self._reader_cache_id, category, instrument, uid, quantity
        )
        cache_name = to_filename(cache_key)
        cache_file = os.path.expanduser(
            os.path.join("~", CACHE_DIR, self.__class__.__name__, cache_name)
        )
        os.makedirs(os.path.dirname(cache_file), 0o755, exist_ok=True)
        intrinsic_bad = self._get_bad_channels(uid, instrument, quantity)
        dtype = data.coords[channel_dim].dtype
        if os.path.exists(cache_file):
            cached_vals = np.loadtxt(cache_file, dtype)
            if cached_vals.ndim == 0:
                cached_vals = np.array([cached_vals])
        else:
            cached_vals = np.array(intrinsic_bad)
        ignored = self._selector(
            data, channel_dim, [*intrinsic_bad, *bad_channels], cached_vals
        )
        form = "%d" if np.issubdtype(dtype, np.integer) else "%.18e"
        np.savetxt(cache_file, np.array(ignored), form)
        return ignored

    def _set_times_item(
        self,
        results: Dict[str, Any],
        times: np.ndarray,
    ):
        """Add the "times" data to the dictionary, if not already
        present.

        """
        if "times" not in results:
            results["times"] = times

    def _get_bad_channels(
        self, uid: str, instrument: str, quantity: str
    ) -> List[Number]:
        """Returns a list of channels which are known to be bad for all pulses
        on this instrument. Typically this would be for reasons of
        geometry (e.g., lines of sight facing the diverter). This
        should be overridden with machine-specific information.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data).
        instrument
            Name of the instrument which measured this data.
        quantities
            Which physical quantity this data represents.

        Returns
        -------
        :
            A list of channels known to be problematic. These will be ignored
            by default.

        """
        return []

    def _warn_less_than_zero(self, instrument: str, data: Dict[str, DataArray]):
        for key, val in data.items():
            if np.any(val < 0):
                warnings.warn(
                    f"{key} in {instrument} contains values less than 0.", UserWarning
                )

    @classmethod
    def available_quantities(cls, instrument):
        """Return the quantities which can be read for the specified
        instrument."""
        if instrument not in cls.INSTRUMENT_METHODS:
            raise ValueError("Can not read data for instrument {}".format(instrument))
        if instrument in cls._IMPLEMENTATION_QUANTITIES:
            return cls._IMPLEMENTATION_QUANTITIES[instrument]
        else:
            return cls._AVAILABLE_QUANTITIES[cls.INSTRUMENT_METHODS[instrument]]
