"""Abstract interface for equilibrium data class.
"""

from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import prov
from xarray import DataArray

from .numpy_typing import LabeledArray
from .session import Session


class AbstractEquilibrium(ABC):
    """Abstract class describing the interface to the equilibrium
    data. This is used to resolve circular dependencies.

    """

    rmag: DataArray
    zmag: DataArray
    provenance: prov.model.ProvEntity
    _session: Session

    @abstractmethod
    def Btot(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
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

    @abstractmethod
    def R_lfs(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
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
        raise NotImplementedError(
            "{} does not implement an 'R_lfs' method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def R_hfs(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
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
        R_hfs
            Major radius on the HFS for the given flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.

        """
        raise NotImplementedError(
            "{} does not implement an 'R_hfs' method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def cross_sectional_area(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        ntheta: Optional[int] = 12,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """Calculates the cross-sectional area inside the flux surface rho and at
        given time t.

        Parameters
        ----------
        rho
            Values of rho at which to calculate the cross-sectional area.
        t
            Values of time at which to calculate the cross-sectional area.
        ntheta
            Number subdivisions of 2 * pi to integrate over for the cross-
            sectional area.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        area
            Cross-sectional areas calculated at rho and t.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def enclosed_volume(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, DataArray, LabeledArray]:
        """Returns the volume enclosed by the specified flux surface.

        Parameters
        ----------
        rho
            Flux surfaces to get the enclosed volumes for.
        t
            Times at which to get the enclosed volume. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        vol
            Volumes of space enclosed by the flux surfaces.
        area
            Cross sectional area enclosed by the flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def invert_enclosed_volume(
        self,
        vol: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Returns the value of the flux surface enclosing the specified volume.

        Parameters
        ----------
        vol
            Volumes of space enclosed by the flux surfaces.
        t
            Times at which to get the enclosed volume. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        rho
            Flux surfaces for the enclosed volumes.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'invert_enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def minor_radius(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
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
            "{} does not implement an 'minor_radius' method.".format(
                self.__class__.__name__
            )
        )

    @abstractmethod
    def flux_coords(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
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
            "{} does not implement an 'flux_coords' method.".format(
                self.__class__.__name__
            )
        )

    @abstractmethod
    def spatial_coords(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
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
            "{} does not implement an 'spatial_coords' method.".format(
                self.__class__.__name__
            )
        )

    @abstractmethod
    def convert_flux_coords(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        from_kind: Optional[str] = "poloidal",
        to_kind: Optional[str] = "toroidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Convert between different coordinate systems.

        Parameters
        ----------
        rho
            Input flux surface coordinate.
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
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'convert_flux_coords' method.".format(
                self.__class__.__name__
            )
        )
