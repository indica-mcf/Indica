"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import Optional

import numpy as np
import prov.model as prov
from xarray import DataArray

import session


class Equilibrium:
    """Abstract base class for reading in equilibrium data.
    """

    def __init__(
        self,
        R_ax: DataArray,
        z_ax: DataArray,
        R_sep: DataArray,
        z_sep: DataArray,
        tstart: float,
        tend: float,
        sess: session.Session = session.global_session,
        **kwargs
    ):
        self.R_ax = R_ax
        self.z_ax = z_ax
        self.R_sep = R_sep
        self.z_sep = z_sep
        self.tstart = tstart
        self.tend = tend
        self._session = sess
        self.prov_id = session.hash_vals(
            equilib_type=self.__class__.__name__,
            R_ax=R_ax,
            z_ax=z_ax,
            R_sep=R_sep,
            z_sep=z_sep,
            tstart=tstart,
            tend=tend,
            **kwargs
        )
        self.provenance = session.prov.entity(
            self.prov_id,
            dict(
                **{"tstart": tstart, "tend": tend, prov.PROV_TYPE: "Equilibrium"},
                **kwargs
            ),
        )
        session.prov.generation(
            self.provenance, session.session, time=datetime.datetime.now()
        )
        session.prov.attribution(self.provenance, session.agent)
        self.provenance.wasDerivedFrom(R_ax.attrs["provenance"])
        self.provenance.wasDerivedFrom(z_ax.attrs["provenance"])
        self.provenance.wasDerivedFrom(R_sep.attrs["provenance"])
        self.provenance.wasDerivedFrom(z_sep.attrs["provenance"])

    def calibrate(self, T_e: DataArray):
        """Works out the offset needed for the for the equilibrium to line up
        properly. (I.e., to ensure the electron temperature is about
        100eV at the separatrix).

        Parameters
        ----------
        T_e
            Electron temperature data (from HRTS on JET).

        """
        # TODO: Define an interface so can pass in a prompt function
        # (with sane default).
        # TODO: Actually write this
        # TODO: Determine what to do with result (return it, use internally,
        #       etc.)
        # TODO: Maybe I should call this with the constructor.
        pass

    def Btot(
        self, R: np.ArrayLike, z: np.ArrayLike, t: Optional[np.ArrayLike] = None
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike]:
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
            "{} does not implement an 'Btot' " "method.".format(self.__class__.__name__)
        )

    def R_lfs(
        self,
        rho: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        kind: str = "toroidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike]:
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
        self,
        rho: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        kind: str = "toroidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike]:
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
        self, rho: np.ArrayLike, t: Optional[np.ArrayLike] = None
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike]:
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
        rho: np.ArrayLike,
        theta: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        kind: str = "toroidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike]:
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
        R: np.ArrayLike,
        z: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        kind: str = "toroidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike, np.ArrayLike]:
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
        rho: np.ArrayLike,
        theta: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        kind: str = "toroidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike, np.ArrayLike]:
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
        rho: np.ArrayLike,
        theta: np.ArrayLike,
        t: Optional[np.ArrayLike] = None,
        from_kind: Optional[str] = "toroidal",
        to_kind: Optional[str] = "poloidal",
    ) -> np.Tuple[np.ArrayLike, np.ArrayLike, np.ArrayLike]:
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
