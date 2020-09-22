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

from .selectors import choose_on_plot
from .selectors import DataSelector
from ..abstractio import BaseIO
from ..converters import FluxSurfaceCoordinates
from ..converters import TransectCoordinates
from ..converters import TrivialTransform
from ..datatypes import ArrayType
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
    DDA_METHODS: Dict[str, str]
        Mapping between instrument/DDA names and method to use to assemble that
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

    DDA_METHODS: Dict[str, str] = {}
    # Mapping between methods for reading data and the quantities which can be
    # fetched. An implementation may override this for specific DDAs.
    _AVAILABLE_QUANTITIES: Dict[str, Dict[str, ArrayType]] = {
        "get_thomson_scattering": {
            "ne": ("number_density", "electrons"),
            "te": ("temperature", "electrons"),
        },
        "get_charge_exchange": {
            "angf": ("angular_freq", None),
            "conc": ("concentration", None),
            "ti": ("temperature", None),
        },
        "get_bremsstrahlung_spectroscopy": {
            "h": ("effective_charge", "plasma"),
            "v": ("effective_charge", "plasma"),
        },
        "get_equilibrium": {
            "f": ("f_value", "plasma"),
            "faxs": ("magnetic_flux", "mag_axis"),
            "fbnd": ("magnetic_flux", "separatrix"),
            "ftor": ("toroidal_flux", "plasma"),
            "rmji": ("major_rad", "hfs"),
            "rmjo": ("major_rad", "lfs"),
            "psi": ("magnetic_flux", "plasma"),
            "vjac": ("volume_jacobian", "plasma"),
            "rmag": ("major_rad", "mag_axis"),
            "rbnd": ("major_rad", "separatrix"),
            "zmag": ("z", "mag_axis"),
            "zbnd": ("z", "separatrix"),
        },
        "get_cyclotron_emissions": {"te": ("temperature", "electrons"),},
        "get_radiation": {"h": ("luminous_flux", None), "v": ("luminous_flux", None),},
    }
    # Quantities available for specific DDAs in a given
    # implementation. Override values given in _AVAILABLE_QUANTITIES.
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
        self._tstart = tstart - 0.5
        self._tend = tend + 0.5
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
        revision: int = 0,
        quantities: Set[str] = set(),
    ) -> Dict[str, DataArray]:
        """Reads data for the requested instrument/DDA. In general this will be
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
        if instrument not in self.DDA_METHODS:
            raise ValueError(
                "{} does not support reading for instrument {}".format(
                    self.__class__.__name__, instrument
                )
            )
        method = getattr(self, self.DDA_METHODS[instrument])
        if not quantities:
            quantities = set(self.available_quantities(instrument))
        return method(uid, instrument, revision, quantities)

    def get_thomson_scattering(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
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
        ticks = np.arange(database_results["length"])
        diagnostic_coord = instrument + "_coord"
        times = database_results["times"]
        coords = [("t", times), (diagnostic_coord, ticks)]
        data = {}
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        transform = TransectCoordinates(database_results["R"], database_results["z"])
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read Thomson scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            cachefile = self._RECORD_TEMPLATE.format(
                self._reader_cache_id, "thomson", instrument, uid, quantity
            )
            meta = {
                "datatype": available_quantities[quantity],
                "error": DataArray(database_results[quantity + "_error"], coords).sel(
                    t=slice(self._tstart, self._tend)
                ),
                "transform": transform,
            }
            quant_data = DataArray(database_results[quantity], coords, attrs=meta,).sel(
                t=slice(self._tstart, self._tend)
            )
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim", keep_attrs=True
                ).mean()
                quant_data.attrs["error"] = np.sqrt(
                    (quant_data.attrs["error"] ** 2)
                    .coarsen(t=downsample_ratio, boundary="trim", keep_attrs=True)
                    .mean()
                    / downsample_ratio
                )
            quant_data.name = instrument + "_" + quantity
            drop = self._select_channels(cachefile, quant_data, diagnostic_coord)
            quant_data.attrs["provenance"] = self.create_provenance(
                "thomson_scattering",
                uid,
                instrument,
                revision,
                quantity,
                database_results[quantity + "_records"],
                drop,
            )
            data[quantity] = quant_data.indica.ignore_data(drop, diagnostic_coord)
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
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
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
                self._reader_cache_id, "cxrs", instrument, uid, quantity
            )
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
        self, uid: str, calculation: str, revision: int, quantities: Set[str]
    ) -> Dict[str, DataArray]:
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        :
            A dictionary containing the requested physical quantities.

        """
        location_quantities = {"psi", "rmag", "zmag", "faxs", "fbnd"}
        separatrix_quantities = {"rbnd", "zbnd"}
        flux_quantities = {"f", "ftor", "vjac", "rmji", "rmjo"}
        available_quantities = self.available_quantities(calculation)
        database_results = self._get_equilibrium(uid, calculation, revision, quantities)
        diagnostic_coord = "rho_poloidal"
        times = database_results["times"]
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        coords_1d = {"t": times}
        dims_1d = coords_1d.keys()
        trivial_transform = TrivialTransform(0.0, 0.0, 0.0, 0.0, 0.0)
        if len(flux_quantities & quantities):
            rho = np.sqrt(database_results["psin"])
            coords_2d = {"t": times, diagnostic_coord: rho}
            flux_transform = FluxSurfaceCoordinates(
                "poloidal", rho, 0.0, 0.0, 0.0, np.expand_dims(times, 1)
            )
        else:
            rho = None
            coords_2d = {}
            flux_transform = FluxSurfaceCoordinates("poloidal", 0.0, 0.0, 0.0, 0.0, 0.0)
        dims_2d = coords_2d.keys()
        if len(separatrix_quantities & quantities):
            dims_sep = ["t", "arbitrary_index"]
            coords_sep = {"t": times}
        else:
            dims_sep = []
            coords_sep = {}
        if "psi" in quantities:
            coords_3d = {
                "t": database_results["times"],
                "R": database_results["psi_r"],
                "z": database_results["psi_z"],
            }
        else:
            coords_3d = {}
        dims_3d = coords_3d.keys()
        data = {}
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read thomson_scattering data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            meta = {
                "datatype": available_quantities[quantity],
                "transform": trivial_transform
                if quantity in location_quantities | separatrix_quantities
                else flux_transform,
            }
            coords, dims = (
                (coords_3d, dims_3d)
                if quantity == "psi"
                else (coords_1d, dims_1d)
                if quantity in location_quantities
                else (coords_sep, dims_sep)
                if quantity in separatrix_quantities
                else (coords_2d, dims_2d)
            )
            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim", keep_attrs=True
                ).mean()
            quant_data.name = calculation + "_" + quantity
            quant_data.attrs["provenance"] = self.create_provenance(
                "equilibrium",
                uid,
                calculation,
                revision,
                quantity,
                database_results[quantity + "_records"],
                [],
            )
            data[quantity] = quant_data
        return data

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
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
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

    def _get_cyclotron_emissions(
        self, uid: str, calculation: str, revision: int, quantities: Set[str]
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
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """Reads data on irradiance.

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
            Which cameras to read quantitie(s) from.

        Returns
        -------
        :
            A dictionary containing the requested radiation values.
        """

    def _get_radiation(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Gets raw data for irradiance from the database. Data outside
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        length : Dict[str, int]
            Number of channels in data for each camera
        times : ndarray
            The times at which measurements were taken

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_Rstart : ndarray
            Major radius of start positions for lines of sight for this data.
        <quantity>_Rstop : ndarray
            Major radius of stop positions for lines of sight for this data.
        <quantity>_zstart : ndarray
            Vertical location of start positions for lines of sight for this data.
        <quantity>_zstop : ndarray
            Vertical location of stop positions for lines of sight for this data.
        <quantity>_Tstart : ndarray
            Toroidal offset of start positions for lines of sight for this data.
        <quantity>_Tstop : ndarray
            Toroidal offset of stop positions for lines of sight for this data.

        """
        raise NotImplementedError(
            "{} does not implement a '_get_radiation' "
            "method.".format(self.__class__.__name__)
        )

    #    def _get_bolometry(
    #        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    #    ) -> Dict[str, Any]:
    #        """Gets raw data for bolometric irradiance from the database. Data outside
    #        the desired time range will be discarded.
    #
    #        Parameters
    #        ----------
    #        uid
    #            User ID (i.e., which user created this data)
    #        instrument
    #            Name of the instrument/DDA which measured this data
    #        revision
    #            An object (of implementation-dependent type) specifying what
    #            version of data to get. Default is the most recent.
    #        quantities
    #            Which physical quantitie(s) to read from the database.
    #
    #        Returns
    #        -------
    #        A dictionary containing the following items:
    #
    #        length : Dict[str, int]
    #            Number of channels in data for each camera
    #        times : ndarray
    #            The times at which measurements were taken
    #
    #        For each requested quantity, the following items will also be present:
    #
    #        <quantity> : ndarray
    #            The data itself (first axis is time, second channel)
    #        <quantity>_error : ndarray
    #            Uncertainty in the data
    #        <quantity>_records : List[str]
    #            Representations (e.g., paths) for the records in the database used
    #            to access data needed for this data.
    #        <quantity>_Rstart : ndarray
    #            Major radius of start positions for lines of sight for this data.
    #        <quantity>_Rstop : ndarray
    #            Major radius of stop positions for lines of sight for this data.
    #        <quantity>_zstart : ndarray
    #            Vertical location of start positions for lines of sight for this data.
    #        <quantity>_zstop : ndarray
    #            Vertical location of stop positions for lines of sight for this data.
    #
    #        """
    #        raise NotImplementedError(
    #            "{} does not implement a '_get_bolometry' "
    #            "method.".format(self.__class__.__name__)
    #        )

    def get_bremsstrahlung_spectroscopy(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
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

    def _get_bremsstrahlung_spectroscopy(
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
            Which physical quantitie(s) to read from the database.

        Returns
        -------
        A dictionary containing the following items:

        times : ndarray
            The times at which measurements were taken

        For each requested quantity, the following items will also be present:

        <quantity> : ndarray
            The data itself (first axis is time, second channel)
        <quantity>_error : ndarray
            Uncertainty in the data
        <quantity>_records : List[str]
            Representations (e.g., paths) for the records in the database used
            to access data needed for this data.
        <quantity>_Rstart : float
            Major radius of start position for line of sight for this data.
        <quantity>_Rstop : float
            Major radius of stop position for line of sight for this data.
        <quantity>_zstart : float
            Vertical location of start position for line of sight for this data.
        <quantity>_zstop : float
            Vertical location of stop position for line of sight for this data.

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
            prov.PROV_VALUE: ",".join(self.available_quantities(instrument)[quantity]),
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
        user's choices will be cached for reuse later, overwriting any
        existing records which were loaded.

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
        dtype = data.coords[channel_dim].dtype
        if os.path.exists(cache_file):
            cached_vals = np.loadtxt(cache_file, dtype)
            if cached_vals.ndim == 0:
                cached_vals = np.array([cached_vals])
        else:
            cached_vals = []
        ignored = self._selector(data, channel_dim, bad_channels, cached_vals)
        form = "%d" if np.issubdtype(dtype, np.integer) else "%.18e"
        np.savetxt(cache_file, ignored, form)
        return ignored

    def _set_times_item(
        self, results: Dict[str, Any], times: np.ndarray,
    ):
        """Add the "times" data to the dictionary, if not already
        present.

        """
        if "times" not in results:
            results["times"] = times

    def available_quantities(self, instrument):
        """Return the quantities which can be read for the specified
        instrument/DDA."""
        if instrument not in self.DDA_METHODS:
            raise ValueError("Can not read data for instrument {}".format(instrument))
        if instrument in self._IMPLEMENTATION_QUANTITIES:
            return self._IMPLEMENTATION_QUANTITIES[instrument]
        else:
            return self._AVAILABLE_QUANTITIES[self.DDA_METHODS[instrument]]
