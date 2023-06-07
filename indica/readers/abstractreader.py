"""Experimental design for reading data from disk/database.
"""

from copy import deepcopy
import datetime
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import prov.model as prov
import xarray as xr
from xarray import DataArray

from .available_quantities import AVAILABLE_QUANTITIES
from ..abstractio import BaseIO
from ..converters.line_of_sight import LineOfSightTransform
from ..converters.transect import TransectCoordinates
from ..datatypes import ArrayType
from ..numpy_typing import RevisionLike
from ..session import hash_vals
from ..session import Session

# TODO: Place this in some global location?
CACHE_DIR = ".indica"


class DataReader(BaseIO):
    """Abstract base class to read data in from a database.

    This defines the interface used by all concrete objects which read
    data from the disc, a database, etc. It is a `context manager
    <https://protect-eu.mimecast.com/s/f7vJCpzxoFzjOpcPXjtX?domain=docs.python.org>`_
    and can be used in a `with statement
    <https://protect-eu.mimecast.com/s/ITLqCq2ypIOpkJuX7qUj?domain=docs.python.org>`_.

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
        self.session.prov.add_namespace(self.NAMESPACE[0], self.NAMESPACE[1])
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
        **kwargs,
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
        return method(uid, instrument, revision, quantities, **kwargs)

    def get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads data based on Thomson Scattering.
        """
        database_results = self._get_thomson_scattering(
            uid, instrument, revision, quantities
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        channel = np.arange(database_results["length"])
        times = database_results["times"]
        x = database_results["x"]
        y = database_results["y"]
        z = database_results["z"]
        R = database_results["R"]
        x_coord = DataArray(x, coords=[("channel", channel)])
        y_coord = DataArray(y, coords=[("channel", channel)])
        z_coord = DataArray(z, coords=[("channel", channel)])
        R_coord = DataArray(R, coords=[("channel", channel)])
        if x_coord.equals(y_coord):
            x_coord = R_coord
            y_coord = xr.zeros_like(x_coord)
        transform = TransectCoordinates(
            x_coord,
            y_coord,
            z_coord,
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        coords = [
            ("t", times),
            ("channel", channel),
        ]
        data = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform,
            )

            quant_data = quant_data.assign_coords(x=("channel", x_coord))
            quant_data = quant_data.assign_coords(y=("channel", y_coord))
            quant_data = quant_data.assign_coords(z=("channel", z_coord))
            quant_data = quant_data.assign_coords(R=("channel", R_coord))

            data[quantity] = quant_data

        return data

    def _get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for Thomson scattering from the database
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
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads Charge-exchange-spectroscopy data
        """
        database_results = self._get_charge_exchange(
            uid, instrument, revision, quantities
        )
        channel = np.arange(database_results["length"])
        times = database_results["times"]
        x = database_results["x"]
        y = database_results["y"]
        z = database_results["z"]
        R = database_results["R"]
        x_coord = DataArray(x, coords=[("channel", channel)])
        y_coord = DataArray(y, coords=[("channel", channel)])
        z_coord = DataArray(z, coords=[("channel", channel)])
        R_coord = DataArray(R, coords=[("channel", channel)])
        if x_coord.equals(y_coord):
            x_coord = R_coord
            y_coord = xr.zeros_like(x_coord)
        transform = TransectCoordinates(
            x_coord,
            y_coord,
            z_coord,
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        coords = [
            ("t", times),
            ("channel", channel),
        ]

        location = database_results["location"]
        direction = database_results["direction"]
        if location is not None and direction is not None:
            los_transform = LineOfSightTransform(
                location[:, 0],
                location[:, 1],
                location[:, 2],
                direction[:, 0],
                direction[:, 1],
                direction[:, 2],
                f"{instrument}",
                machine_dimensions=database_results["machine_dims"],
                dl=dl,
                passes=passes,
            )

        data = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform,
            )
            if location is not None and direction is not None:
                quant_data.attrs["los_transform"] = los_transform

            quant_data = quant_data.assign_coords(x=("channel", x_coord))
            quant_data = quant_data.assign_coords(y=("channel", y_coord))
            quant_data = quant_data.assign_coords(z=("channel", z_coord))
            quant_data = quant_data.assign_coords(R=("channel", R_coord))

            data[quantity] = quant_data

        return data

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for CXRS diagnostic from the database
        """
        raise NotImplementedError(
            "{} does not implement a '_get_charge_exchange' "
            "method.".format(self.__class__.__name__)
        )

    def get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, DataArray]:
        """
        Reads equilibrium data
        """

        database_results = self._get_equilibrium(uid, instrument, revision, quantities)

        _coords: dict = {}
        rho = np.sqrt(database_results["psin"])
        _coords["psin"] = [("psin", database_results["psin"])]
        _coords["psi"] = [
            ("t", database_results["times"]),
            ("z", database_results["psi_z"]),
            ("R", database_results["psi_r"]),
        ]

        global_quantities: list = [
            "rmag",
            "zmag",
            "rgeo",
            "faxs",
            "fbnd",
            "ipla",
            "wp",
            "df",
        ]
        global_coords = [("t", database_results["times"])]
        for quant in global_quantities:
            _coords[quant] = global_coords

        separatrix_quantities: list = ["rbnd", "zbnd"]
        if "rbnd" in database_results.keys():
            sep_coords = [
                ("t", database_results["times"]),
                ("arbitrary_index", np.arange(np.size(database_results["rbnd"][0, :]))),
            ]
            for quant in separatrix_quantities:
                _coords[quant] = sep_coords

        flux_quantities: list = ["f", "ftor", "vjac", "ajac", "rmji", "rmjo"]
        for quant in flux_quantities:
            _coords[quant] = [("t", database_results["times"]), ("rho_poloidal", rho)]

        times = database_results["times"]
        times_unique, ind_unique = np.unique(times, return_index=True)
        rmag = self.assign_dataarray(
            uid,
            instrument,
            "rmag",
            database_results,
            _coords["rmag"],
            include_error=False,
        )
        zmag = self.assign_dataarray(
            uid,
            instrument,
            "zmag",
            database_results,
            _coords["zmag"],
            include_error=False,
        )
        data: Dict[str, DataArray] = {}
        for quantity in quantities:
            coords = _coords[quantity]
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                include_error=False,
            )
            if quantity in {"rmji", "rmjo"}:
                quant_data.coords["z"] = zmag
            elif quantity == "faxs":
                quant_data.coords["R"] = rmag
                quant_data.coords["z"] = zmag

            if len(times) != len(times_unique):
                print(
                    """Equilibrium time axis does not have
                    unique elements...correcting..."""
                )
                quant_data = quant_data.isel(t=ind_unique)
            data[quantity] = quant_data

        return data

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for equilibrium from the database
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
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        raise NotImplementedError("Needs to be reimplemented")

    def _get_cyclotron_emissions(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for electron cyclotron emission diagnostic data from the database.
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
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads data from radiation diagnostics e.g. bolometry and SXR
        """

        database_results = self._get_radiation(
            uid,
            instrument,
            revision,
            quantities,
        )
        location = database_results["location"]
        direction = database_results["direction"]
        transform = LineOfSightTransform(
            location[:, 0],
            location[:, 1],
            location[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
            dl=dl,
            passes=passes,
        )

        coords = [("t", database_results["times"])]
        if database_results["length"] > 1:
            coords.append(("channel", np.arange(database_results["length"])))

        data = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform=transform,
            )
            data[quantity] = quant_data

        return data

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for radiation diagnostics from the database
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
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads spectroscopic measurements of effective charge
        """
        database_results = self._get_bremsstrahlung_spectroscopy(
            uid,
            instrument,
            revision,
            quantities,
        )
        location = database_results["location"]
        direction = database_results["direction"]
        transform = LineOfSightTransform(
            location[:, 0],
            location[:, 1],
            location[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
            dl=dl,
            passes=passes,
        )

        coords = [("t", database_results["times"])]
        if database_results["length"] > 1:
            coords.append(("channel", np.arange(database_results["length"])))
        data = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform=transform,
            )
            data[quantity] = quant_data

        return data

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw spectroscopic data for effective charge from the database
        """
        raise NotImplementedError(
            "{} does not implement a '_get_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    def get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads spectroscopic measurements of He-like emission
        """

        database_results = self._get_helike_spectroscopy(
            uid,
            instrument,
            revision,
            quantities,
        )

        location = database_results["location"]
        direction = database_results["direction"]
        transform = LineOfSightTransform(
            location[:, 0],
            location[:, 1],
            location[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
            dl=dl,
            passes=passes,
        )

        _coords: dict = {}
        _coords["1d"] = [("t", database_results["times"])]
        if database_results["length"] > 1:
            _coords["1d"].append(("channel", np.arange(database_results["length"])))
        _coords["spectra"] = deepcopy(_coords["1d"])
        _coords["spectra"].append(("wavelength", database_results["wavelength"]))

        data: dict = {}
        for quantity in quantities:
            if quantity in _coords:
                coords = _coords["spectra"]
            else:
                coords = _coords["1d"]

            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform=transform,
            )
            data[quantity] = quant_data

        return data

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Reads spectroscopic measurements of He-like emission data from database
        """
        raise NotImplementedError(
            "{} does not implement a '_get_helike_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    def get_diode_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads filtered radiation diodes
        """
        database_results = self._get_diode_filters(
            uid,
            instrument,
            revision,
            quantities,
        )
        location = database_results["location"]
        direction = database_results["direction"]
        transform = LineOfSightTransform(
            location[:, 0],
            location[:, 1],
            location[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
            dl=dl,
            passes=passes,
        )
        coords = [("t", database_results["times"])]
        if database_results["length"] > 1:
            coords.append(("channel", np.arange(database_results["length"])))

        data: dict = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform=transform,
            )
            data[quantity] = quant_data

        return data

    def _get_diode_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Reads filtered radiation diodes data from database
        """
        raise NotImplementedError(
            "{} does not implement a '_get_diode_filters' "
            "method.".format(self.__class__.__name__)
        )

    def get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 2,
    ) -> Dict[str, DataArray]:
        """
        Reads interferometer diagnostic data
        """
        database_results = self._get_interferometry(
            uid,
            instrument,
            revision,
            quantities,
        )
        location = database_results["location"]
        direction = database_results["direction"]
        transform = LineOfSightTransform(
            location[:, 0],
            location[:, 1],
            location[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
            dl=dl,
            passes=passes,
        )
        coords = [("t", database_results["times"])]
        if database_results["length"] > 1:
            coords.append(("channel", np.arange(database_results["length"])))

        data: dict = {}
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                transform=transform,
            )
            data[quantity] = quant_data

        return data

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Reads interferometer diagnostic data from database
        """
        raise NotImplementedError(
            "{} does not implement a '_get_spectroscopy' "
            "method.".format(self.__class__.__name__)
        )

    def get_astra(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads ASTRA data
        """
        database_results = self._get_astra(uid, instrument, revision, quantities)

        # Reorganise coordinate system to match Indica default rho-poloidal
        psin = database_results["psin"]
        rhop_psin = np.sqrt(psin)
        rhop_interp = np.linspace(0, 1.0, 65)
        rhot_astra = database_results["rho"] / np.max(database_results["rho"])
        rhot_rhop = []
        for it in range(len(database_results["times"])):
            ftor_tmp = database_results["ftor"][it, :]
            psi_tmp = database_results["psi_1d"][it, :]
            rhot_tmp = np.sqrt(ftor_tmp / ftor_tmp[-1])
            rhop_tmp = np.sqrt((psi_tmp - psi_tmp[0]) / (psi_tmp[-1] - psi_tmp[0]))
            rhot_xpsn = np.interp(rhop_interp, rhop_tmp, rhot_tmp)
            rhot_rhop.append(rhot_xpsn)

        rhot_rhop = DataArray(
            np.array(rhot_rhop),
            {"t": database_results["times"], "rho_poloidal": rhop_interp},
            dims=["t", "rho_poloidal"],
        ).sel(t=slice(self._tstart, self._tend))

        radial_coords = {
            "rho_toroidal": rhot_astra,
            "rho_poloidal": rhop_psin,
            "R": database_results["psi_r"],
            "z": database_results["psi_z"],
            "arbitrary_index": database_results["boundary_index"],
        }

        data: dict = {}
        for quantity in quantities:
            if "PROFILES.ASTRA" in database_results[f"{quantity}_records"][0]:
                name_coords = ["rho_toroidal"]
            elif "PROFILES.PSI_NORM" in database_results[f"{quantity}_records"][0]:
                name_coords = ["rho_poloidal"]
            elif "PSI2D" in database_results[f"{quantity}_records"][0]:
                name_coords = ["z", "R"]
            elif "BOUNDARY" in database_results[f"{quantity}_records"][0]:
                name_coords = ["arbitrary_index"]
            else:
                name_coords = []

            coords: list = [("t", database_results["times"])]
            if len(name_coords) > 0:
                for coord in name_coords:
                    coords.append((coord, radial_coords[coord]))

            if len(np.shape(database_results[quantity])) != len(coords):
                continue

            quant_data = self.assign_dataarray(
                uid,
                instrument,
                quantity,
                database_results,
                coords,
                include_error=False,
            )

            # Convert radial coordinate to rho_poloidal
            # TODO: Check interpolatoin on rho_poloidal array...
            if "rho_toroidal" in quant_data.dims:
                rho_toroidal_0 = quant_data.rho_toroidal.min()
                quant_interp = quant_data.interp(rho_toroidal=rhot_rhop).drop_vars(
                    "rho_toroidal"
                )
                quant_interp.loc[dict(rho_poloidal=0)] = quant_data.sel(
                    rho_toroidal=rho_toroidal_0
                )
                quant_data = quant_interp.interpolate_na("rho_poloidal")
            elif "rho_poloidal" in coords:
                quant_data = quant_data.interp(rho_poloidal=rhop_interp)

            data[quantity] = quant_data

        return data

    def _get_astra(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Reads ASTRA data from database
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
        revision: RevisionLike,
        quantity: str,
        data_objects: Iterable[str],
    ) -> prov.ProvEntity:
        """Create a provenance entity for the given set of data. This should
        be attached as metadata.

        Note that this method just creates the provenance data
        appropriate for the arguments it has been provided with. It
        does not check that these arguments are actually valid and
        that the provenance corresponds to actually existing data.

        Parameters
        ----------
        data_objects
            Identifiers for the database entries or files which the data was
            read from.

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
            date=end_time,
        )
        attrs = {
            prov.PROV_TYPE: "DataArray",
            prov.PROV_VALUE: ",".join(
                str(s) for s in self.available_quantities(instrument)[quantity]
            ),
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

    def available_quantities(self, instrument) -> dict:
        """Return the quantities which can be read for the specified
        instrument."""
        if instrument not in self.INSTRUMENT_METHODS:
            raise ValueError("Can not read data for instrument {}".format(instrument))
        if instrument in self._IMPLEMENTATION_QUANTITIES:
            return self._IMPLEMENTATION_QUANTITIES[instrument]
        else:
            return self._AVAILABLE_QUANTITIES[self.INSTRUMENT_METHODS[instrument]]

    def assign_dataarray(
        self,
        uid: str,
        instrument: str,
        quantity: str,
        database_results: Dict[str, DataArray],
        coords: List,
        transform=None,
        include_error: bool = True,
    ) -> DataArray:
        """

        Parameters
        ----------

        uid
            User ID
        instrument
            The instrument name
        quantity
            The physical quantity to assign
        database_results
            Dictionary output of private reader methods
        coords
            DataArray coordinate list.
        transform
            Coordinate transform.
        include_error
            Add error to DataArray attributes

        Returns
        -------

        """
        available_quantities = self.available_quantities(instrument)

        quant_data = DataArray(
            database_results[quantity],
            coords,
        )
        if "t" in quant_data.dims:
            quant_data = quant_data.sel(t=slice(self._tstart, self._tend))

        quant_data.attrs = {
            "datatype": available_quantities[quantity],
        }

        if include_error:
            if quantity + "_error" in database_results:
                quant_error = DataArray(
                    database_results[quantity + "_error"], coords=coords
                )
                if "t" in quant_data.dims:
                    quant_error = quant_error.sel(t=slice(self._tstart, self._tend))
            else:
                quant_error = xr.zeros_like(quant_data)
            quant_data.attrs["error"] = quant_error

        if transform is not None:
            quant_data.attrs["transform"] = transform

        if "times" in database_results:
            times = database_results["times"]
            downsample_ratio = int(
                np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
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
        quant_data.attrs["partial_provenance"] = self.create_provenance(
            self.INSTRUMENT_METHODS[instrument],
            uid,
            instrument,
            database_results["revision"],
            quantity,
            database_results[quantity + "_records"],
        )
        quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]
        quant_data = quant_data

        return quant_data
