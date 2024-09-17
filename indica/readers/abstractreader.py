"""Experimental design for reading data from disk/database."""

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
from indica.numpy_typing import OnlyArray
from indica.numpy_typing import RevisionLike
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
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
    INSTRUMENT_METHODS: Dict[str, str]
        Mapping between instrument (DDA in JET) names and method to use to assemble that
        data. Implementation-specific.
    NAMESPACE: Classvar[Tuple[str, str]]
        The abbreviation and full URL for the namespace of the database
        the class reads from.
    """

    INSTRUMENT_METHODS: Dict[str, str] = {}
    _AVAILABLE_QUANTITIES = AVAILABLE_QUANTITIES
    _IMPLEMENTATION_QUANTITIES: Dict[str, Dict[str, str]] = {}

    _RECORD_TEMPLATE = "{}-{}-{}-{}-{}"
    NAMESPACE: Tuple[str, str] = ("experiment", "server")

    def __init__(
        self,
        tstart: float,
        tend: float,
        **kwargs: Any,
    ):
        """This should be called by constructors on subtypes.

        Parameters
        ----------
        tstart
            Start of time range for which to get data.
        tend
            End of time range for which to get data.
        kwargs
            Any other arguments which should be recorded for the reader.
        """
        self._reader_cache_id: str
        self._tstart = tstart
        self._tend = tend
        self._start_time = None

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
        **kwargs,
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
            if database_results.get(quantity) is None:
                continue
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

    def get_ppts(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        database_results = self._get_ppts(uid, instrument, revision, quantities)
        database_results["channel"] = np.arange(database_results["length"])

        coords_chan = {"channel": database_results["channel"]}
        coords = {
            "t": database_results["t"],
            "channel": database_results["channel"],
        }
        rho_poloidal_data = xr.DataArray(
            database_results["rho_poloidal_data"], coords=coords
        )
        rho_poloidal_data = rho_poloidal_data.sel(t=slice(self._tstart, self._tend))
        R_data = xr.DataArray(database_results["R_data"], coords=coords_chan)
        z_data = xr.DataArray(database_results["z_data"], coords=coords_chan)

        data = {}
        for quantity in quantities:
            if database_results.get(quantity) is None:
                continue
            if "_R" in quantity:
                dims = ["t", "R"]
            elif "_rho" in quantity:
                dims = ["t", "rho_poloidal"]
            elif "_data" in quantity:
                dims = ["t", "channel"]
            elif "R_shift" in quantity:
                dims = ["t"]
            else:
                raise ValueError(f"Unknown quantity: {quantity}")
            if quantity == "R_shift":
                include_error = False
            else:
                include_error = True

            data[quantity] = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=None,
                include_error=include_error,
            )

            if "_R" in quantity:
                data[quantity] = data[quantity].assign_coords(
                    z=("R", database_results["z"])
                )

            if "_data" in quantity:
                data[quantity] = data[quantity].assign_coords(
                    rho_poloidal=(("t", "channel"), rho_poloidal_data.data)
                )
                data[quantity] = data[quantity].assign_coords(
                    z=("channel", z_data.data)
                )
                data[quantity] = data[quantity].assign_coords(
                    R=("channel", R_data.data)
                )
        return data

    def _get_ppts(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for PPTS analysis from the database
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a '_get_ppts' " "method."
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

        database_results["channel"] = np.arange(database_results["length"])
        if "wavelength" in database_results.keys():
            database_results["pixel"] = np.arange(len(database_results["wavelength"]))

        data = {}
        for quantity in quantities:
            if database_results.get(quantity) is None:
                continue
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
            if (
                database_results["location"] is not None
                and database_results["direction"] is not None
            ):
                quant_data.attrs["los_transform"] = instatiate_line_of_sight(
                    database_results["location"],
                    database_results["direction"],
                    instrument,
                    database_results["machine_dims"],
                    dl,
                    passes,
                )
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
        """
        database_results = self._get_spectrometer(uid, instrument, revision, quantities)

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
        for quantity in quantities:
            if database_results.get(quantity) is None:
                continue
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
        dl: float = None,
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
        **kwargs,
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
            if database_results.get(quantity) is None:
                continue
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
            if database_results.get(quantity) is None:
                continue
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
            if database_results.get(quantity) is None:
                continue
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
            if database_results.get(quantity) is None:
                continue
            if quantity in ["spectra", "raw_spectra"]:
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
            if database_results.get(quantity) is None:
                continue
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
            if database_results.get(quantity) is None:
                continue
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

    def get_zeff(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        database_results = self._get_zeff(uid, instrument, revision, quantities)

        data = {}
        for quantity in quantities:
            if database_results.get(quantity) is None:
                continue
            _path: str = database_results[f"{quantity}_records"]
            print(_path)
            if "global" in _path.lower():
                dims = ["t"]
            elif "profiles" in _path.lower():
                dims = ["t", "rho_poloidal"]
            else:
                raise ValueError(f"Unknown quantity: {quantity}")

            data[quantity] = self.assign_dataarray(
                instrument,
                quantity,
                database_results,
                dims,
                transform=None,
            )
        return data

    def _get_zeff(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """
        Gets raw data for ZEFF analysis from the database
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a '_get_zeff' " "method."
        )

    # def get_astra(
    #     self,
    #     uid: str,
    #     instrument: str,
    #     revision: RevisionLike,
    #     quantities: Set[str],
    #     **kwargs,
    # ) -> Dict[str, DataArray]:
    #     """
    #     Reads ASTRA data
    #     --> PSIN from PSI2D is reference coordinate,
    #         ASTRA profiles interpolated accordingly
    #     """
    #     database_results = self._get_astra(uid, instrument, revision, quantities)
    #
    #     # Reorganise coordinates
    #     # --> ASTRA profiles interpolated on reference PSI_NORM.PSIN grid
    #     database_results["z"] = database_results["psi_z"]
    #     database_results["R"] = database_results["psi_r"]
    #     database_results["rho_poloidal"] = np.sqrt(database_results["psin"])
    #     database_results["rho_toroidal"] = np.sqrt(database_results["ftor"])
    #     rho_poloidal_astra = np.sqrt(database_results["psin_astra"])
    #     for quantity in quantities:
    #
    #
    #     if "rbnd" in database_results.keys():
    #         database_results["arbitrary_index"] = np.arange(
    #             np.size(database_results["rbnd"][0, :])
    #         )
    #
    #
    #
    #     return database_results
    #
    #     # Group variables for coordinate assignement
    #     sep_vars = ["rbnd", "zbnd"]
    #     flux_vars = ["f", "ftor", "volume", "area"]
    #
    #     radial_coords = {
    #         "rho_toroidal": rhot_astra,
    #         "rho_poloidal": rhop_psin,
    #         "R": database_results["psi_r"],
    #         "z": database_results["psi_z"],
    #         "arbitrary_index": database_results["boundary_index"],
    #     }
    #     data: Dict[str, DataArray] = {}
    #     for quantity in quantities:
    #         print(quantity)
    #         if quantity == "psi":
    #             dims = ["t", "z", "R"]
    #         elif quantity in sep_vars:
    #             dims = ["t", "arbitrary_index"]
    #         elif quantity in flux_vars:
    #             dims = ["t", "rho_poloidal"]
    #         else:  # global quantities
    #             dims = ["t"]
    #
    #         quant_data = self.assign_dataarray(
    #             instrument,
    #             quantity,
    #             database_results,
    #             dims,
    #             include_error=False,
    #         )
    #
    #         data[quantity] = quant_data
    #
    #     if "faxs" in quantities:
    #         data["faxs"].coords["R"] = data["rmag"]
    #         data["faxs"].coords["z"] = data["zmag"]
    #
    #
    #
    #
    #     return
    #
    #
    #     data: dict = {}
    #     for quantity in quantities:
    #         if "PROFILES.ASTRA" in database_results[f"{quantity}_records"][0]:
    #             name_coords = ["rho_toroidal"]
    #         elif "PROFILES.PSI_NORM" in database_results[f"{quantity}_records"][0]:
    #             name_coords = ["rho_poloidal"]
    #         elif "PSI2D" in database_results[f"{quantity}_records"][0]:
    #             name_coords = ["z", "R"]
    #         elif "BOUNDARY" in database_results[f"{quantity}_records"][0]:
    #             name_coords = ["arbitrary_index"]
    #         else:
    #             name_coords = []
    #
    #         coords: dict = {"t": t}
    #         if len(name_coords) > 0:
    #             for coord in name_coords:
    #                 coords[coord] = radial_coords[coord]
    #
    #         if len(np.shape(database_results[quantity])) != len(coords.keys()):
    #             continue
    #
    #         quant_data = self.assign_dataarray(
    #             instrument,
    #             quantity,
    #             database_results,
    #             coords,
    #             include_error=False,
    #         )
    #
    #         # Convert radial coordinate to rho_poloidal
    #         # TODO: Check interpolatoin on rho_poloidal array...
    #         if "rho_toroidal" in quant_data.dims:
    #             rho_toroidal_0 = quant_data.rho_toroidal.min()
    #             quant_interp = quant_data.interp(rho_toroidal=rhot_rhop).drop_vars(
    #                 "rho_toroidal"
    #             )
    #             quant_interp.loc[dict(rho_poloidal=0)] = quant_data.sel(
    #                 rho_toroidal=rho_toroidal_0
    #             )
    #             quant_data = quant_interp.interpolate_na("rho_poloidal")
    #         elif "rho_poloidal" in coords:
    #             quant_data = quant_data.interp(rho_poloidal=rhop_interp)
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
    #     """
    #     Reads ASTRA data from database
    #     """
    #     raise NotImplementedError(
    #         "{} does not implement a '_get_spectroscopy' "
    #         "method.".format(self.__class__.__name__)
    #     )

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
        coords = {}
        for dim in dims:
            coords[dim] = database_results[dim]

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

        data = data.sortby(dims)

        if include_error:
            data = data.assign_coords(error=(data.dims, error.data))

        if transform is not None:
            data.attrs["transform"] = transform

        return data
