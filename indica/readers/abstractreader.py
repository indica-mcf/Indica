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
import xarray as xr

import numpy as np
import prov.model as prov
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from .available_quantities import AVAILABLE_QUANTITIES
from .selectors import choose_on_plot
from .selectors import DataSelector
from ..abstractio import BaseIO
from ..converters import FluxSurfaceCoordinates
from ..converters import MagneticCoordinates
from indica.converters.transect import TransectCoordinates
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
        dl: float = 0.005,
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
        dl
            spatial precision for line-of-sight coordinate transform

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
        return method(uid, instrument, revision, quantities, dl)

    def get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, DataArray]:
        """
        Reads data based on Thomson Scattering.
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_thomson_scattering(
            uid, instrument, revision, quantities, dl
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        ticks = np.arange(database_results["length"])
        # diagnostic_coord = instrument + "_coord"
        diagnostic_coord = "channel"
        times = database_results["times"]
        x = database_results["x"]
        y = database_results["y"]
        z = database_results["z"]
        R = database_results["R"]  # np.sqrt(x**2 + y**2)
        x_coord = DataArray(x, coords=[(diagnostic_coord, ticks)])
        y_coord = DataArray(y, coords=[(diagnostic_coord, ticks)])
        z_coord = DataArray(z, coords=[(diagnostic_coord, ticks)])
        R_coord = DataArray(R, coords=[(diagnostic_coord, ticks)])
        null_array = xr.full_like(x_coord, 0.0)
        if all(x_coord == null_array) and all(y_coord == null_array):
            x_coord = R_coord
        transform = TransectCoordinates(x_coord, y_coord, z_coord, f"{instrument}",)
        coords: Dict[Hashable, ArrayLike] = {
            "t": times,
            diagnostic_coord: ticks,
            transform.x2_name: 0,
            "x": x_coord,
            "y": y_coord,
            "z": z_coord,
            "R": R_coord,
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

            quant_data = DataArray(database_results[quantity], coords, dims,).sel(
                t=slice(self._tstart, self._tend)
            )
            quant_error = DataArray(
                database_results[quantity + "_error"], coords, dims
            ).sel(t=slice(self._tstart, self._tend))
            quant_data.attrs = {
                "datatype": available_quantities[quantity],
                "error": quant_error,
                "transform": transform,
            }
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
            drop: list = []
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
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads Charge-exchange-spectroscopy data
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_charge_exchange(
            uid, instrument, revision, quantities, dl,
        )
        # return database_results
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        ticks = np.arange(database_results["length"])
        # diagnostic_coord = instrument + "_coord"
        diagnostic_coord = "channel"
        data = {}
        # needs change (see GET_RADIATION) - but must still work for other readers
        x_coord = DataArray(database_results["x"], coords=[(diagnostic_coord, ticks)])
        y_coord = DataArray(database_results["y"], coords=[(diagnostic_coord, ticks)])
        z_coord = DataArray(database_results["z"], coords=[(diagnostic_coord, ticks)])
        R_coord = DataArray(database_results["R"], coords=[(diagnostic_coord, ticks)])
        # R_coord = np.sqrt(x_coord**2 + y_coord**2)
        transform = TransectCoordinates(x_coord, y_coord, z_coord, f"{instrument}",)
        times = database_results["times"]
        coords: Dict[Hashable, Any] = {
            "t": times,
            diagnostic_coord: ticks,
            transform.x2_name: 0,
            "x": x_coord,
            "y": y_coord,
            "z": z_coord,
            "R": R_coord,
        }
        dims = ["t", diagnostic_coord]
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        # TODO: why use ffill as method??? Temporarily removed...
        texp = DataArray(database_results["texp"], coords=[("t", times)]).sel(
            t=slice(self._tstart, self._tend)
        )
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
                ).sel(t=slice(self._tstart, self._tend)),
                "exposure_time": texp,
            }
            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))
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
            drop: list = []
            # drop = self._select_channels(
            #     "cxrs", uid, instrument, quantity, quant_data, diagnostic_coord
            # )
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

        return data

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
        dl: float = 0.005,
    ) -> Dict[str, DataArray]:
        """
        Reads equilibrium data
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
            "df",
        }
        separatrix_quantities = {"rbnd", "zbnd"}
        flux_quantities = {"f", "ftor", "vjac", "ajac", "rmji", "rmjo"}
        available_quantities = self.available_quantities(instrument)
        if "rmji" in quantities and "rmjo" in quantities:
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
        coords_1d: Dict[Hashable, ArrayLike] = {"t": times}
        dims_1d = ("t",)
        coords_psin: dict = {}
        dims_psin: tuple = ()
        trivial_transform = TrivialTransform()
        if len(flux_quantities & quantities) > 0:
            psin = database_results["psin"]
            coords_psin = {"psin": psin}
            dims_psin = ("psin",)
            rho = np.sqrt(database_results["psin"])
            coords_2d: Dict[Hashable, ArrayLike] = {"t": times, diagnostic_coord: rho}
        else:
            coords_2d = {}
        flux_transform = FluxSurfaceCoordinates("poloidal",)
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
        # TODO: add this in the available_quantities.py
        if "psin" not in quantities:
            sorted_quantities.insert(0, "psin")
            available_quantities["psin"] = ("poloidal_flux", "normalised")

        for quantity in sorted_quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} cannot read data for "
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
            elif quantity == "psin":
                coords = coords_psin
                dims = dims_psin
            else:
                coords = coords_2d
                dims = dims_2d

            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            )
            if "t" in dims:
                quant_data = quant_data.sel(t=slice(self._tstart, self._tend))

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
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads electron cyclotron emission data
        """
        available_quantities = self.available_quantities(instrument)
        for quantity in quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read cyclotron emission data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

        database_results = self._get_cyclotron_emissions(
            uid, instrument, revision, quantities, dl
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
                ).sel(t=slice(self._tstart, self._tend)),
                "transform": transform,
            }
            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))
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
            drop: list = []
            # drop = self._select_channels(
            #     "cyclotron",
            #     uid,
            #     instrument,
            #     quantity,
            #     quant_data,
            #     transform.x1_name,
            #     database_results["bad_channels"],
            # )
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
        return data

    def _get_cyclotron_emissions(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads data from radiation diagnostics e.g. bolometry and SXR
        """

        available_quantities = self.available_quantities(instrument)
        database_results = self._get_radiation(
            uid, instrument, revision, quantities, dl
        )
        # return database_results
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

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
            database_results["machine_dims"],
            dl=dl,
        )
        data = {}
        quantity = "brightness"
        if quantity not in available_quantities:
            raise ValueError(
                "{} can not read radiation data for quantity {}".format(
                    self.__class__.__name__, quantity
                )
            )

        times = database_results["times"]
        downsample_ratio = int(
            np.ceil((len(times) - 1) / (times[-1] - times[0]) / self._max_freq)
        )
        coords = [
            ("t", times),
            (transform.x1_name, np.arange(database_results["length"])),
        ]
        meta = {
            "datatype": available_quantities[quantity],
            "error": DataArray(database_results[quantity + "_error"], coords).sel(
                t=slice(self._tstart, self._tend),
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
        drop: list = []
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

        return data

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads spectroscopic measurements of effective charge
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_bremsstrahlung_spectroscopy(
            uid, instrument, revision, quantities, dl
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]
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
            database_results["machine_dims"],
            dl=dl,
        )
        data = {}
        quantity = "zeff"
        if quantity not in available_quantities:
            raise ValueError(
                "{} can not read bremsstrahlung data for quantity {}".format(
                    self.__class__.__name__, quantity
                )
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
        meta = {
            "datatype": available_quantities[quantity],
            "error": DataArray(database_results[quantity + "_error"], coords, dims).sel(
                t=slice(self._tstart, self._tend)
            ),
            "transform": transform,
        }
        quant_data = DataArray(
            database_results[quantity], coords, dims, attrs=meta,
        ).sel(t=slice(self._tstart, self._tend))
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
        drop: list = []
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
        return data

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads spectroscopic measurements of He-like emission
        """

        available_quantities = self.available_quantities(instrument)
        database_results = self._get_helike_spectroscopy(
            uid, instrument, revision, quantities, dl
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return {}
        _revision = database_results["revision"]

        times = database_results["times"]
        wavelength = database_results["wavelength"]
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
            database_results["machine_dims"],
            dl=dl,
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
                ).sel(t=slice(self._tstart, self._tend))
            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))
            if downsample_ratio > 1:
                quant_data = quant_data.coarsen(
                    t=downsample_ratio, boundary="trim", keep_attrs=True
                ).mean()
                if quantity_error in database_results.keys():
                    quant_data.attrs["error"] = np.sqrt(
                        (quant_data.attrs["error"] ** 2)
                        .coarsen(t=downsample_ratio, boundary="trim", keep_attrs=True)
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
        return data

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads filtered radiation diodes
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_diode_filters(
            uid, instrument, revision, quantities, dl
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        times = database_results["times"]

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
            database_results["machine_dims"],
            dl=dl,
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
        quantity = "brightness"
        if quantity not in available_quantities:
            raise ValueError(
                "{} can not read filtered diode data for quantity {}".format(
                    self.__class__.__name__, quantity
                )
            )
        meta = {
            "datatype": available_quantities[quantity],
            "error": DataArray(database_results[quantity + "_error"], coords, dims).sel(
                t=slice(self._tstart, self._tend)
            ),
            "transform": transform,
        }
        quant_data = DataArray(
            database_results[quantity], coords, dims, attrs=meta,
        ).sel(t=slice(self._tstart, self._tend))
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

    def _get_diode_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads interferometer diagnostic data
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_interferometry(
            uid, instrument, revision, quantities, dl
        )
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        if len(database_results) == 0:
            return database_results

        times = database_results["times"]
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
            database_results["machine_dims"],
            dl=dl,
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
                ).sel(t=slice(self._tstart, self._tend)),
                "transform": transform,
            }

            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))
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
        return data

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
    ) -> Dict[str, DataArray]:
        """
        Reads ASTRA data
        """
        available_quantities = self.available_quantities(instrument)
        database_results = self._get_astra(uid, instrument, revision, quantities, dl)
        if len(database_results) == 0:
            print(f"No data from {uid}.{instrument}:{revision}")
            return database_results
        _revision = database_results["revision"]

        data: Dict[str, DataArray] = {}

        # Reorganise coordinate system to match Indica default rho-poloidal
        psin = database_results["psin"]
        rhop_psin = np.sqrt(psin)
        rhop_interp = np.linspace(0, 1.0, 65)
        rhot_astra = database_results["rho"] / np.max(database_results["rho"])
        rhot_rhop = []
        for it in range(len(database_results["times"])):
            ftor_tmp = database_results["ftor"][it, :]
            psi_tmp = database_results["psi"][it, :]
            rhot_tmp = np.sqrt(ftor_tmp / ftor_tmp[-1])
            rhop_tmp = np.sqrt((psi_tmp - psi_tmp[0]) / (psi_tmp[-1] - psi_tmp[0]))
            rhot_xpsn = np.interp(rhop_interp, rhop_tmp, rhot_tmp)
            rhot_rhop.append(rhot_xpsn)

        rhot_rhop = DataArray(
            np.array(rhot_rhop),
            {"t": database_results["times"], "rho_poloidal": rhop_interp},
            dims=["t", "rho_poloidal"],
        ).sel(t=slice(self._tstart, self._tend))

        radial_coords = {"rho_toroidal": rhot_astra, "rho_poloidal": rhop_psin}

        sorted_quantities = sorted(quantities)
        for quantity in sorted_quantities:
            if quantity not in available_quantities:
                raise ValueError(
                    "{} can not read astra data for "
                    "quantity {}".format(self.__class__.__name__, quantity)
                )

            if "PROFILES.ASTRA" in database_results[f"{quantity}_records"][0]:
                name_coord = "rho_toroidal"
            elif "PROFILES.PSI_NORM" in database_results[f"{quantity}_records"][0]:
                name_coord = "rho_poloidal"
            else:
                name_coord = ""

            coords = {"t": database_results["times"]}
            dims = ["t"]
            if len(name_coord) > 0:
                coords[name_coord] = radial_coords[name_coord]
                dims.append(name_coord)

            trivial_transform = TrivialTransform()
            meta = {
                "datatype": available_quantities[quantity],
                "transform": trivial_transform,
            }

            quant_data = DataArray(
                database_results[quantity], coords, dims, attrs=meta,
            ).sel(t=slice(self._tstart, self._tend))

            # TODO: careful with interpolation on new rho_poloidal array...
            # Interpolate ASTRA profiles on new rhop_interp array
            # Interpolate PSI_NORM profiles on same coordinate system
            if name_coord == "rho_toroidal":
                rho_toroidal_0 = quant_data.rho_toroidal.min()
                quant_interp = quant_data.interp(rho_toroidal=rhot_rhop).drop_vars(
                    "rho_toroidal"
                )
                quant_interp.loc[dict(rho_poloidal=0)] = quant_data.sel(
                    rho_toroidal=rho_toroidal_0
                )
                quant_data = quant_interp.interpolate_na("rho_poloidal")
            elif name_coord == "rho_poloidal":
                quant_data = quant_data.interp(rho_poloidal=rhop_interp)

            quant_data.name = instrument + "_" + quantity
            quant_data.attrs["partial_provenance"] = self.create_provenance(
                "astra",
                uid,
                instrument,
                _revision,
                quantity,
                database_results[quantity + "_records"],
                [],
            )

            quant_data.attrs["provenance"] = quant_data.attrs["partial_provenance"]

            data[quantity] = quant_data

        return data

    def _get_astra(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
        self, results: Dict[str, Any], times: np.ndarray,
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

    def available_quantities(self, instrument):
        """Return the quantities which can be read for the specified
        instrument."""
        if instrument not in self.INSTRUMENT_METHODS:
            raise ValueError("Can not read data for instrument {}".format(instrument))
        if instrument in self._IMPLEMENTATION_QUANTITIES:
            return self._IMPLEMENTATION_QUANTITIES[instrument]
        else:
            return self._AVAILABLE_QUANTITIES[self.INSTRUMENT_METHODS[instrument]]
