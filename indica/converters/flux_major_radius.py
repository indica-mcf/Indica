"""Defines a coordinate system for use when estimating emissivity data."""

from typing import cast
from typing import Dict

import numpy as np
from xarray import DataArray

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from .flux_surfaces import FluxSurfaceCoordinates
from ..numpy_typing import LabeledArray


class FluxMajorRadCoordinates(CoordinateTransform):
    """A coordinate system that uses a flux surface \rho and major radius
    to determine spatial positions. This is used, e.g., when
    estimating an emissivity profile of the plasma based on X-ray
    data. Note that this coordinate system loses information about
    vertical position. When converting to (R,z) coordinates, the
    results will have ``z >= 0``.

    Parameters
    ----------
    flux_surfaces : FluxSurfaceCoordinates
        The flux surface coordinate system to use for \rho.

    """

    _INVERSE_CONVERSION_METHODS: Dict[str, str] = {
        "LinesOfSightCoordinates": "_convert_from_flux_coords"
    }

    def __init__(self, flux_surfaces: FluxSurfaceCoordinates):
        super().__init__(
            flux_surfaces.default_x1,
            flux_surfaces.default_x2,
            flux_surfaces.default_R,
            flux_surfaces.default_z,
            flux_surfaces.default_t,
        )
        self.flux_surfaces = flux_surfaces
        self.equilibriub = flux_surfaces.equilibrium
        self.flux_kind = flux_surfaces.flux_kind

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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        R, z, t = self.flux_surfaces.convert_to_Rz(rho, theta, t)
        return rho, R, t

    def _convert_to_Rz(
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
        t
            Time coordinate (if one passed as an argument, then is just a
            pointer to that)

        """
        theta_grid = DataArray(
            np.linspace(0.0, np.pi, 100), dims=("theta",), name="theta"
        )
        R, z, t = self.flux_surfaces.convert_to_Rz(x1, theta_grid, t)
        theta_vals = cast(DataArray, R).indica.invert_interp(x2, target="theta")
        return x2, cast(DataArray, z).indica.interp2d(theta=theta_vals), t

    def _convert_from_Rz(
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
        t
            The time coordinate (if one pass as an argument then is just a
            pointer to that)

        """
        rho, theta, t = self.flux_surfaces.convert_from_Rz(R, z, t)
        return rho, R, t

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        return result and self.flux_surfaces == other.flux_surfaces
