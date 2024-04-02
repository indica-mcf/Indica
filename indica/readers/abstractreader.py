"""Experimental design for reading data from disk/database.
"""

import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from indica.abstractio import BaseIO
from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters.transect import TransectCoordinates
from indica.datatypes import ArrayType
from indica.numpy_typing import OnlyArray
from indica.numpy_typing import RevisionLike
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.session import hash_vals
from indica.session import Session
from indica.utilities import format_coord
from indica.utilities import format_dataarray


def instatiate_line_of_sight(
    location: OnlyArray,
    direction: OnlyArray,
    instrument: str,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]],
    dl: float,
    passes: int,
) -> LineOfSightTransform:
    return LineOfSightTransform(
        location[:, 0],
        location[:, 1],
        location[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        f"{instrument}",
        machine_dimensions=machine_dimensions,
        dl=dl,
        passes=passes,
    )


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
        self._start_time = None
        self.session = sess
        self.session.prov.add_namespace(self.NAMESPACE[0], self.NAMESPACE[1])
        prov_attrs: Dict[str, Any] = dict(tstart=tstart, tend=tend, **kwargs)
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

        transform = TransectCoordinates(
            database_results["x"],
            database_results["y"],
            database_results["z"],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        database_results["channel"] = np.arange(database_results["length"])

        data = {}
        dims = ["t", "channel"]
        for quantity in quantities:
            data[quantity] = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=transform,
            )

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

        transform = TransectCoordinates(
            database_results["x"],
            database_results["y"],
            database_results["z"],
            f"{instrument}",
            machine_dimensions=database_results["machine_dims"],
        )
        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )

        database_results["channel"] = np.arange(database_results["length"])
        if "wavelength" in database_results.keys():
            database_results["pixel"] = np.arange(len(database_results["wavelength"]))

        data = {}
        for quantity in quantities:
            if quantity == "spectra" or quantity == "fit":
                dims = ["t", "channel", "wavelength"]
            else:
                dims = ["t", "channel"]

            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=transform,
            )
            quant_data.attrs["los_transform"] = los_transform
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

    def get_spectrometer(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
        passes: int = 1,
    ) -> Dict[str, DataArray]:
        """
        Reads spectroscopy data
        TODO: find better way to filter non-acquired channels
        TODO: check spectra uncertainty...
        """
        database_results = self._get_spectrometer(uid, instrument, revision, quantities)

        if instrument == "pi":
            has_data = np.arange(21, 28)
        else:
            has_data = np.where(
                np.isfinite(database_results["spectra"][0, :, 0])
                * (database_results["spectra"][0, :, 0] > 0)
            )[0]
        database_results["spectra"] = database_results["spectra"][:, has_data, :]
        database_results["spectra_error"] = database_results["spectra"] * 0.0

        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        database_results["channel"] = np.arange(len(has_data))

        data = {}
        for quantity in quantities:
            if quantity == "spectra":
                dims = ["t", "channel", "wavelength"]
            else:
                dims = ["t", "channel"]

            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
            )

            data[quantity] = quant_data

        return data

    def _get_spectrometer(
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

        # Reorganise coordinates
        database_results["rho_poloidal"] = np.sqrt(database_results["psin"])
        database_results["z"] = database_results["psi_z"]
        database_results["R"] = database_results["psi_r"]
        if "rbnd" in database_results.keys():
            database_results["arbitrary_index"] = np.arange(
                np.size(database_results["rbnd"][0, :])
            )

        # Group variables for coordinate assignement
        sep_vars = ["rbnd", "zbnd"]
        flux_vars = ["f", "ftor", "vjac", "ajac", "rmji", "rmjo"]

        # Check that times are unique
        correct_times: bool = False
        t_unique, ind_unique = np.unique(database_results["t"], return_index=True)
        if len(database_results["t"]) != len(t_unique):
            correct_times = True

        data: Dict[str, DataArray] = {}
        for quantity in quantities:
            if quantity == "psi":
                dims = ["t", "z", "R"]
            elif quantity in sep_vars:
                dims = ["t", "arbitrary_index"]
            elif quantity in flux_vars:
                dims = ["t", "rho_poloidal"]
            else:  # global quantities
                dims = ["t"]

            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                include_error=False,
            )

            if correct_times:
                print(f"{instrument}: correcting non-unique times")
                quant_data = quant_data.isel(t=ind_unique)

            data[quantity] = quant_data

        # Add additional coordinates
        if "rmji" in quantities:
            data["rmji"].coords["z"] = data["zmag"]
        if "rmji" in quantities:
            data["rmjo"].coords["z"] = data["zmag"]
        if "faxs" in quantities:
            data["faxs"].coords["R"] = data["rmag"]
            data["faxs"].coords["z"] = data["zmag"]

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
        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        database_results["channel"] = np.arange(database_results["length"])

        data = {}
        dims = ["t", "channel"]
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
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
        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        if database_results["length"] > 1:
            database_results["channel"] = np.arange(database_results["length"])

        data = {}
        dims = ["t"]
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
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

        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        database_results["channel"] = np.arange(database_results["length"])

        data: dict = {}
        for quantity in quantities:
            if quantity == "spectra":
                dims = ["t", "wavelength"]
            else:
                dims = ["t"]

            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
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
        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        database_results["channel"] = np.arange(database_results["length"])

        data: dict = {}
        dims = ["t", "channel"]
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
            )
            data[quantity] = quant_data.assign_coords(
                label=("channel", database_results["labels"])
            )

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
        los_transform = instatiate_line_of_sight(
            database_results["location"],
            database_results["direction"],
            instrument,
            database_results["machine_dims"],
            dl,
            passes,
        )
        database_results["channel"] = np.arange(database_results["length"])

        data: dict = {}
        dims = ["t"]
        for quantity in quantities:
            quant_data = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=los_transform,
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

        TODO: must be fixed post change to new DATATYPES and AVAILABLE_QUANTITIES
        """
        database_results = self._get_astra(uid, instrument, revision, quantities)

        # Reorganise coordinate system to match Indica default rho-poloidal
        t = database_results["t"]
        t = DataArray(t, coords=[("t", t)], attrs={"long_name": "t", "units": "s"})
        psin = database_results["psin"]
        rhop_psin = np.sqrt(psin)
        rhop_interp = np.linspace(0, 1.0, 65)
        rhot_astra = database_results["rho"] / np.max(database_results["rho"])
        rhot_rhop = []
        for it in range(len(database_results["t"])):
            ftor_tmp = database_results["ftor"][it, :]
            psi_tmp = database_results["psi_1d"][it, :]
            rhot_tmp = np.sqrt(ftor_tmp / ftor_tmp[-1])
            rhop_tmp = np.sqrt((psi_tmp - psi_tmp[0]) / (psi_tmp[-1] - psi_tmp[0]))
            rhot_xpsn = np.interp(rhop_interp, rhop_tmp, rhot_tmp)
            rhot_rhop.append(rhot_xpsn)

        rhot_rhop = DataArray(
            np.array(rhot_rhop),
            {"t": t, "rho_poloidal": rhop_interp},
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

            coords: list = [("t", t)]
            if len(name_coords) > 0:
                for coord in name_coords:
                    coords.append((coord, radial_coords[coord]))

            if len(np.shape(database_results[quantity])) != len(coords):
                continue

            quant_data = self.assign_dataarray(
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

    # def create_provenance(
    #     self,
    #     diagnostic: str,
    #     uid: str,
    #     instrument: str,
    #     revision: RevisionLike,
    #     quantity: str,
    #     data_objects: Iterable[str],
    # ) -> prov.ProvEntity:
    #     """Create a provenance entity for the given set of data. This should
    #     be attached as metadata.
    #
    #     Note that this method just creates the provenance data
    #     appropriate for the arguments it has been provided with. It
    #     does not check that these arguments are actually valid and
    #     that the provenance corresponds to actually existing data.
    #
    #     Parameters
    #     ----------
    #     data_objects
    #         Identifiers for the database entries or files which the data was
    #         read from.
    #
    #     Returns
    #     -------
    #     :
    #         A provenance entity for the newly read-in data.
    #     """
    #     end_time = datetime.datetime.now()
    #     entity_id = hash_vals(
    #         creator=self.prov_id,
    #         diagnostic=diagnostic,
    #         uid=uid,
    #         instrument=instrument,
    #         revision=revision,
    #         quantity=quantity,
    #         date=end_time,
    #     )
    #     attrs = {
    #         prov.PROV_TYPE: "DataArray",
    #         prov.PROV_VALUE: ",".join(
    #             str(s) for s in self.available_quantities(instrument)[quantity]
    #         ),
    #         "uid": uid,
    #         "instrument": instrument,
    #         "diagnostic": diagnostic,
    #         "revision": revision,
    #         "quantity": quantity,
    #     }
    #     activity_id = hash_vals(agent=self.prov_id, date=end_time)
    #     activity = self.session.prov.activity(
    #         activity_id,
    #         self._start_time,
    #         end_time,
    #         {prov.PROV_TYPE: "ReadData"},
    #     )
    #     activity.wasAssociatedWith(self.session.agent)
    #     activity.wasAssociatedWith(self.agent)
    #     activity.wasInformedBy(self.session.session)
    #     entity = self.session.prov.entity(entity_id, attrs)
    #     entity.wasGeneratedBy(activity, end_time)
    #     entity.wasAttributedTo(self.session.agent)
    #     entity.wasAttributedTo(self.agent)
    #     for data in data_objects:
    #         # TODO: Find some way to avoid duplicate records
    #         data_entity = self.session.prov.entity(self.NAMESPACE[0] + ":" + data)
    #         entity.wasDerivedFrom(data_entity)
    #         activity.used(data_entity)
    #         return entity

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
        instrument: str,
        quantity: str,
        database_results: Dict[str, DataArray],
        dims: List[str],
        transform=None,
        include_error: bool = True,
    ) -> DataArray:
        """

        Parameters
        ----------
        instrument
            The instrument name
        quantity
            The physical quantity to assign
        database_results
            Dictionary output of private reader methods
        coords
            List of coordinate names from database_results.
        transform
            Coordinate transform.
        include_error
            Add error to DataArray attributes

        Returns
        -------
        DataArray with assigned coordinates, transform, error, long_name and units
        """
        # Build coordinate dictionary
        coords = []
        for dim in dims:
            coords.append((dim, format_coord(database_results[dim], dim)))

        # Build DataArray data with coordinates and long_name + units
        var_name = self.available_quantities(instrument)[quantity]
        data = format_dataarray(database_results[quantity], var_name, coords)
        if "t" in data.dims:
            data = data.sel(t=slice(self._tstart, self._tend))

        # ..do the same with the error
        error = xr.zeros_like(data)
        if quantity + "_error" in database_results:
            error = format_dataarray(
                database_results[quantity + "_error"], var_name, coords
            )
            if "t" in error.dims:
                error = error.sel(t=slice(self._tstart, self._tend))

        if include_error:
            data.attrs["error"] = error

        if transform is not None:
            data.attrs["transform"] = transform

        return data
