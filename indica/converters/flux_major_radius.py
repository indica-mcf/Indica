"""Defines a coordinate system for use when estimating emissivity data."""

from typing import Callable
from typing import cast
from typing import Dict
from typing import Optional

import numpy as np
from xarray import DataArray

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from ..numpy_typing import LabeledArray


class FluxMajorRadCoordinates(CoordinateTransform):
    """A coordinate system that uses a flux surface :math:`\\rho` and
    major radius to determine spatial positions. This is used, e.g.,
    when estimating an emissivity profile of the plasma based on X-ray
    data. Note that this coordinate system loses information about
    vertical position. When converting to (R,z) coordinates, the
    results will have ``z >= 0``.

    Parameters
    ----------
    flux_surfaces : FluxSurfaceCoordinates
        The flux surface coordinate system to use for :math:`\\rho`.

    """

    _INVERSE_CONVERSION_METHODS: Dict[str, str] = {
        "FluxSurfaceCoordinates": "_convert_from_flux_coords"
    }

    x2_name = "R"

    def __init__(self, flux_surfaces: FluxSurfaceCoordinates):
        self.flux_surfaces = flux_surfaces
        self.equilibrium = flux_surfaces.equilibrium
        self.flux_kind = flux_surfaces.flux_kind
        self.x1_name = flux_surfaces.x1_name

    def get_converter(
        self, other: CoordinateTransform, reverse=False
    ) -> Optional[Callable[[LabeledArray, LabeledArray, LabeledArray], Coordinates]]:
        """Checks if there is a shortcut to convert between these coordiantes,
        returning it if so. This can sometimes save the step of
        converting to (R, z) coordinates first.

        Parameters
        ----------
        other
            The other transform whose coordinate system you want to convert to.
        reverse
            If True, try to return a function which converts _from_ ``other``
            to this coordinate system.

        Returns
        -------
        :
            If a shortcut function is available, return it. Otherwise, None.

        Note
        ----
        Implementations should call ``other.get_converter(self, reverse=True``. For
        obvious reasons, however, they should **only do this when
        ``reverse == False``**.

        """
        if reverse:
            if other == self.flux_surfaces:
                return self._convert_from_flux_coords
            else:
                return None
        return other.get_converter(self, True)

    def _convert_from_flux_coords(
        self, rho: LabeledArray, theta: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from to a flux coordinate system to this one.

        Parameters
        ----------
        rho
            The flux surface value.
        theta
            The poloidal angle on the flux surface.
        t
            The time coordinate

        Returns
        -------
        x1
            The first spatial coordinate in this system.
        x2
            The second spatial coordinate in this system.

        """
        R, z = self.flux_surfaces.convert_to_Rz(rho, theta, t)
        return rho, R

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
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
        theta_grid = DataArray(
            np.linspace(0.0, np.pi, 100), dims=("theta",), name="theta"
        )
        R, z = self.flux_surfaces.convert_to_Rz(x1, theta_grid, t)
        theta_vals = cast(DataArray, R).indica.invert_interp(x2, target="theta")
        return x2, cast(DataArray, z).indica.interp2d(theta=theta_vals)

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
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
        rho, theta = self.flux_surfaces.convert_from_Rz(R, z, t)
        return rho, R

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and self.flux_surfaces == other.flux_surfaces
