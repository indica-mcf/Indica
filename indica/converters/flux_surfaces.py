"""Class to handle conversions to and from flux surface coordinates."""

from typing import Optional

import numpy as np

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


class FluxSurfaceCoordinates(CoordinateTransform):
    """Class for polar coordinate systems using flux surfaces
    for the radial coordinate.

    Parameters
    ----------
    kind
        The type of flux surface to use. Must be a valid argument for methods
        on the :py:class:`indica.equilibrium.Equilibrium` class.

    """

    x2_name = "theta"

    def __init__(
        self,
        kind: str,
    ):
        self.flux_kind = kind
        self.x1_name = "rho_" + kind

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        Parameters
        ----------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        R
            Major radius coordinate
        z
            Height coordinate

        """
        return self.equilibrium.spatial_coords(x1, x2, t, self.flux_kind)[0:2]

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

        Parameters
        ----------
        R
            Major radius coordinate
        z
            Height coordinate
        t
            Time coordinate)

        Returns
        -------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.

        """
        return self.equilibrium.flux_coords(R, z, t, self.flux_kind)[0:2]

    def _convert_to_vol(
        self, rho: LabeledArray, theta: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Coordinates:
        """Convert from this coordinate system to one using volume enclosed by
        the flux surfaces as a coordinate.

        Parameters
        ----------
        rho
            The first spatial coordinate in this system.
        theta
            The second spatial coordinate in this system.
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        vol
            Volume enclosed by the flux surface rho.
        theta
            Poloidal angle coordinate

        """
        vol, t, _ = self.equilibrium.enclosed_volume(rho, t, self.flux_kind)
        return vol, theta

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and np.all(self.flux_kind == other.flux_kind)
