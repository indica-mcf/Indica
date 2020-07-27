"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import DataArray

from .numpy_typing import ArrayLike
from .offset import interactive_offset_choice
from .offset import OffsetPicker
from .session import global_session
from .session import hash_vals
from .session import Session


class Equilibrium:
    """Class to hold and interpolate equilibrium data.

    At instantiation it will require calibration to select an offset
    along the major radius. Electron temperature data is provided for
    this purpose. Once calibrated, the electron temperature at
    normalised flux surface rho = 1 should be about 100eV.

    Parameters
    ----------
    equilibrium_data : Dict[str, DataArray]
        A collection of equilibrium data rea in using
        :py:meth:`~src.readers.DataReader.get_equilibrium`. TODO: List full set
        of required quantities.
    T_e : Optional[DataArray]
        Electron temperature data (from HRTS on JET).
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.
    offset_picker: OffsetPicker
        A callback which determines by how much to offset the equilibrium data
        along the major radius. Allows the user to select this interactively.

    """

    def __init__(
        self,
        equilibrium_data: Dict[str, DataArray],
        T_e: Optional[DataArray] = None,
        sess: Session = global_session,
        ofsset_picker: OffsetPicker = interactive_offset_choice,
    ):
        self._session = sess
        # TODO: Collect necessary data from ``equilbrium_data`` and
        # interpolate as needed
        # TODO: calibrate the equilibrium
        self.prov_id = hash_vals(**equilibrium_data)
        self.provenance = sess.prov.entity(
            self.prov_id, {prov.PROV_TYPE: "Equilibrium"},
        )
        sess.prov.generation(
            self.provenance, sess.session, time=datetime.datetime.now()
        )
        sess.prov.attribution(self.provenance, sess.agent)
        # TODO: Add PROV dependencies to ``equilibrium_data``

    def Btot(
        self, R: ArrayLike, z: ArrayLike, t: Optional[ArrayLike] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Total magnetic field strength at this location in space.

        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertial position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.

        Returns
        -------
        Btot
            Total magnetic field strength at the given location and times.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'Btot' method.".format(self.__class__.__name__)
        )

    def R_lfs(
        self, rho: ArrayLike, t: Optional[ArrayLike] = None, kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Major radius position of the given flux surface on the Low Flux
         Side of the magnetic axis.

        Parameters
        ----------
        rho
            Flux values for the locations.
        t
            Times at which to get the major radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.


        Returns
        -------
        R_lfs
            Major radius on the LFS for the given flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        R, z, t = self.spatial_coords(rho, 0.0, t, kind)
        return R, t

    def R_hfs(
        self, rho: ArrayLike, t: Optional[ArrayLike] = None, kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Major radius position of the given flux surface on the High Flux
         Side of the magnetic axis.

        Parameters
        ----------
        rho
            Flux values for the locations.
        t
            Times at which to get the major radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.


        Returns
        -------
        R_rfs
            Major radius on the RFS for the given flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.

        """
        R, z, t = self.spatial_coords(rho, np.pi, t, kind)
        return R, t

    def enclosed_volume(
        self, rho: ArrayLike, t: Optional[ArrayLike] = None, kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Returns the volume enclosed by the specified flux surface.

        Parameters
        ----------
        rho
            Flux surfaces to get the enclosed volumes for.
        t
            Times at which to get the major radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        :
            Volumes of space enclosed by the flux surfaces.
        """
        raise NotImplementedError(
            "{} does not implement an 'enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

    def minor_radius(
        self,
        rho: ArrayLike,
        theta: ArrayLike,
        t: Optional[ArrayLike] = None,
        kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Minor radius at the given locations in the tokamak.

        Parameters
        ----------
        rho
            Flux surfaces on which the locations fall.
        theta
            Poloidal positions on which the locations fall.
        t
            Times at which to get the minor radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        minor_radius
            Minor radius of the locations.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement a 'minor_radius' "
            "method.".format(self.__class__.__name__)
        )

    def flux_coords(
        self,
        R: ArrayLike,
        z: ArrayLike,
        t: Optional[ArrayLike] = None,
        kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert to the flux surface coordinate system.

        Parameters
        ----------
        R
            Major radius positions.
        z
            Vertical positions.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        rho
            Flux surface for each position.
        theta
            Poloidal angle along flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement a 'flux_coords' "
            "method.".format(self.__class__.__name__)
        )

    def spatial_coords(
        self,
        rho: ArrayLike,
        theta: ArrayLike,
        t: Optional[ArrayLike] = None,
        kind: str = "poloidal",
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert to the spatial coordinate system.

        Parameters
        ----------
        rho
            Flux surface coordinate.
        theta
            Angular position.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        R
            Major radius positions.
        z
            Vertical positions.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement a 'spatial_coords' "
            "method.".format(self.__class__.__name__)
        )

    def convert_flux_coords(
        self,
        rho: ArrayLike,
        theta: ArrayLike,
        t: Optional[ArrayLike] = None,
        from_kind: Optional[str] = "poloidal",
        to_kind: Optional[str] = "toroidal",
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Convert between different coordinate systems.

        Parameters
        ----------
        rho
            Input flux surface coordinate.
        theta
            Input angular position.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        from_kind
            The type of flux surfaces used for the input coordinates. May be
            "toroidal", "poloidal", plus optional extras depending on
            implementation.
        to_kind
            The type of flux surfaces on which to calculate the output
            coordinates. May be "toroidal", "poloidal", plus optional extras
            depending on implementation.

        Returns
        -------
        rho
            New flux surface for each position.
        theta
            Poloidal angle along new flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'convert_flux_coords' "
            "method.".format(self.__class__.__name__)
        )
