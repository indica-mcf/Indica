"""
Coordinate system representing a collection of lines of sight.
TODO: modified with respect to master
"""

from typing import Callable
from typing import Optional
from typing import Tuple

import xarray as xr
import numpy as np
from scipy.optimize import root
from xarray import DataArray
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray
from indica.converters import FluxSurfaceCoordinates


class LinesOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate in this system is an index indicating which
    line-of-site a location is on. The second coordinate ranges from 0
    to 1 (inclusive) and indicates the position of a location along
    the line-of-sight. Note that diagnostic using this coordinate
    system will usually only be indexed in the first coordinate, as
    the measurements were integrated along the line-of-sight.

    If not passed to the constructor, the default grid for converting
    from the x-y-z system is chosen as follows:

    - The x-grid ranges from ``min(x_start.min(), x_end.min())`` to
      ``max(x_start.max(), x_end.max())`` with ``num_points`` intervals.
    - The y-grid ranges from ``min(y_start.min(), y_end.min())`` to
      ``max(y_start.max(), y_end.max())`` with ``num_points`` intervals.
    - The z-grid ranges from ``min(z_start.min(), z_end.min())`` to
      ``max(z_start.max(), z_end.max())`` with ``num_points`` intervals.

    Parameters
    ----------
    x_start
        1-D array of x positions of the start for each line-of-sight.
    y_start
        1-D array of y positions for the start of each line-of-sight.
    z_start
        1-D array of z positions of the start for each line-of-sight.
    x_end
        1-D array of x positions of the end for each line-of-sight.
    y_end
        1-D array of y positions for the end of each line-of-sight.
    z_end
        1-D array of z positions of the end for each line-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in xy-z space:
        ``((xymin, xymax), (zmin, zmax)`` where the machine is symmetric
        with respect to the z-axis. Defaults to values for JET.

    """

    def __init__(
        self,
        x_start: np.ndarray,
        y_start: np.ndarray,
        z_start: np.ndarray,
        x_end: np.ndarray,
        y_end: np.ndarray,
        z_end: np.ndarray,
        name: str,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
    ):
        lengths = _get_wall_intersection_distances(
            x_start, y_start, z_start, x_end, y_end, z_end, machine_dimensions
        )
        new_length = max(lengths)
        los_lengths = np.sqrt(
            (x_start - x_end) ** 2 + (y_start - y_end) ** 2 + (z_start - z_end) ** 2
        )
        factor = new_length / los_lengths
        self.name = f"{name}_line_of_sight_transform"
        self.x_start = DataArray(x_start)
        self.y_start = DataArray(y_start)
        self.z_start = DataArray(z_start)
        self._original_x_end = DataArray(x_end)
        self._original_y_end = DataArray(y_end)
        self._original_z_end = DataArray(z_end)
        self._machine_dims = machine_dimensions
        self.x_end = DataArray(x_start + factor * (x_end - x_start))
        self.y_end = DataArray(y_start + factor * (y_end - y_start))
        self.z_end = DataArray(z_start + factor * (z_end - z_start))
        self.index_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x2_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        if len(x_start) == 1:
            self.x1 = np.array(0)
        else:
            self.x1 = np.arange(0, len(x_start) + 1)
        self.x1_name = "channel"
        self.x2_name = "los_position"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = result and np.all(self.x_start == other.x_start)
        result = result and np.all(self.y_start == other.y_start)
        result = result and np.all(self.z_start == other.z_start)
        result = result and np.all(self.x_end == other.x_end)
        result = result and np.all(self.y_end == other.y_end)
        result = result and np.all(self.z_end == other.z_end)
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x_0 = x_s + (x_e - x_s) * x2
        y_0 = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        return np.sqrt(x_0 ** 2 + y_0 ** 2), z

    def convert_to_xy(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x_0 = x_s + (x_e - x_s) * x2
        y_0 = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        return x_0, y_0

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            x1 = x[0]
            x2 = x[1]
            c = np.ceil(x1).astype(int)
            f = np.floor(x1).astype(int)
            x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
            x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
            y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
            y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
            z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
            z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
            x_0 = x_s + (x_e - x_s) * x2
            y_0 = y_s + (y_e - y_s) * x2
            x = np.sign(x_0) * np.sqrt(x_0 ** 2 + y_0 ** 2)
            dx_0dx1 = (self.x_start[c] - self.x_start[f]) * (1 - x2) + (
                self.x_end[c] - self.x_end[f]
            ) * x2
            dx_0dx2 = x_e - x_s
            dy_0dx1 = (self.y_start[c] - self.y_start[f]) * (1 - x2) + (
                self.y_end[c] - self.y_end[f]
            ) * x2
            dy_0dx2 = y_e - y_s
            dzdx1 = (self.z_start[c] - self.z_start[f]) * (1 - x2) + (
                self.z_end[c] - self.z_end[f]
            ) * x2
            dzdx2 = z_e - z_s
            return [
                [
                    2 / x * (x_0 * dx_0dx1 + y_0 * dy_0dx1),
                    2 / x * (x_0 * dx_0dx2 + y_0 * dy_0dx2),
                ],
                [dzdx1, dzdx2],
            ]

        def forward(x):
            x_prime, z_prime = self.convert_to_Rz(x[0], x[1], 0)
            return [x_prime - R, z_prime - z]

        @np.vectorize
        def invert(R, z):
            """Perform a nonlinear-solve for an R-z pair."""
            # x = [R, z]
            result = root(forward, [0, 0.5], jac=jacobian)
            if not result.success:
                raise RuntimeWarning(
                    "Solver did not fully converge while inverting lines of sight."
                )
            return result.x[0], result.x[1]

        Rz = invert(R, z)
        return Rz[0], Rz[1]
        # # TODO: Consider if there is some way to invert this exactly,
        # # rather than rely on interpolation (which is necessarily
        # # inexact, as well as computationally expensive).
        # if not self.index_inversion:
        #     R_vals, z_vals, _ = self.convert_to_Rz()
        #     points = np.stack((np.ravel(R_vals), np.ravel(z_vals))).T
        #     index_vals = self.default_x1 * np.ones_like(self.default_x2)
        #     x2_vals = np.ones_like(self.default_x1) * self.default_x2
        #     interp2d = (
        #         LinearNDInterpolator if np.all(self.T_start == 0.0) and
        #         np.all(self.T_end == 0.0) else CloughTocher2DInterpolator
        #     )
        #     self.index_inversion = interp2d(points, np.ravel(index_vals))
        #     self.x2_inversion = interp2d(points, np.ravel(x2_vals))
        #     self.index_inversion = Rbf(R_vals, z_vals, index_vals)
        #     self.x2_inversion = Rbf(R_vals, z_vals, x2_vals)
        # assert self.x2_inversion is not None
        # x1 = self.index_inversion(R, z)
        # x2 = self.x2_inversion(R, z)
        # return x1, x2

    def distance(
        self, direction: str, x1: LabeledArray, x2: LabeledArray, t: LabeledArray,
    ) -> LabeledArray:
        """Implementation of calculation of physical distances between points
        in this coordinate system. This accounts for potential toroidal skew of
        lines.

        """
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        z_s = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        z_e = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        x = x_s + (x_e - x_s) * x2
        y = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2
        spacings = np.sqrt(
            x.diff(direction) ** 2 + y.diff(direction) ** 2 + z.diff(direction) ** 2
        )
        result = zeros_like(x)
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates, force:bool=False):
        """
        Set flux surface transform to perform remapping from physical to flux space
        """
        if not hasattr(self, "flux_transform") or force:
            self.flux_transform = flux_transform
        elif self.flux_transform != flux_transform:
            raise Exception("Attempt to set flux surface transform twice.")

    def remap_los(
        self, t: LabeledArray = None, npts: int = 100,
    ):
        """
        Remap LOS from physical to flux space
        """
        if not hasattr(self, "flux_transform"):
            return None
        if not hasattr(self.flux_transform, "equilibrium"):
            return None

        geo_attrs = {}
        x2_arr = np.linspace(0, 1, npts)
        x2 = DataArray(x2_arr, dims=self.x2_name)
        dl = self.distance(self.x2_name, DataArray(0), x2[0:2], 0)[1]
        geo_attrs["x2"] = x2
        geo_attrs["dl"] = dl
        geo_attrs["R"], geo_attrs["z"] = self.convert_to_Rz(self.x1, x2, 0)

        rho, _ = self.flux_transform.convert_from_Rz(geo_attrs["R"], geo_attrs["z"], t=t)
        rho = xr.where(rho >= 0, rho, 0.0)
        rho.coords[self.x2_name] = x2
        geo_attrs["rho"] = rho

        for name, value in geo_attrs.items():
            setattr(self, name, value)

        return geo_attrs

    def simple_los_int_1d(
        self, profile_1d: DataArray, t:LabeledArray=None, passes=1, remap=False,
    ):
        """
        Calculate LOS integral of a specified 1D profile

        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        passes
            Number of passes across the plasma for the integral

        Returns
        -------
        Line of sight integral and value mapped along the LOS
        """
        if not hasattr(self, "flux_transform"):
            raise Exception("Missing flux surface transform")
        if not hasattr(self.flux_transform, "equilibrium"):
            raise Exception("Missing equilibrium in flux surface transform")
        if not hasattr(self, "rho") or remap:
            self.remap_los(t=t)

        if t is not None:
            rho = self.rho.interp(t=t, method="linear")
        else:
            rho = self.rho
        along_los = profile_1d.interp(rho_poloidal=rho)
        along_los = xr.where(rho <= 1, along_los, 0,)
        los_integral = passes * along_los.sum(self.x2_name) * self.dl

        return los_integral, along_los


def _get_wall_intersection_distances(
    x_start: np.ndarray,
    y_start: np.ndarray,
    z_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
    z_end: np.ndarray,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
) -> np.ndarray:
    """
    Calculate the lenght needed for a line of sight to intersect with a wall of
    the Tokamak.

    Parameters
    ----------
    x_start
        1-D array of x positions of the start for each line-of-sight.
    y_start
        1-D array of y positions for the start of each line-of-sight.
    z_start
        1-D array of z positions of the start for each line-of-sight.
    x_end
        1-D array of x positions of the end for each line-of-sight.
    y_end
        1-D array of y positions for the end of each line-of-sight.
    z_end
        1-D array of z positions of the end for each line-of-sight.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in xy-z space:
        ``((xymin, xymax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    lengths
        The length of each line of sight for it to intersect a Tokamak wall.

    """
    # TODO: generalize for (x, y, z) plane looking for intersection with the inner column
    opposite_x = np.where(
        x_end - x_start < 0, machine_dimensions[0][0], machine_dimensions[0][1]
    )
    opposite_z = np.where(
        z_end - z_start < 0, machine_dimensions[1][0], machine_dimensions[1][1]
    )
    # Calculate where LOS intersects opposite R-surface
    a = (x_end - x_start) ** 2 + (y_end - y_start) ** 2
    b = 2 * (x_start * (x_end - x_start) + y_start * (y_end - y_start))
    c = x_start ** 2 + y_start ** 2 - opposite_x ** 2
    factor = np.where(x_end - x_start < 0, -1, 1)
    mask = b ** 2 - 4 * a * c < 0
    # Check line of sight actually intersects the expected wall
    opposite_x[mask] = machine_dimensions[0][1]
    a[mask] = (x_end[mask] - x_start[mask]) ** 2 + (y_end[mask] - y_start[mask]) ** 2
    b[mask] = 2 * (
        x_start[mask] * (x_end[mask] - x_start[mask])
        + y_start[mask] * (y_end[mask] - y_start[mask])
    )
    c[mask] = x_start[mask] ** 2 + y_start[mask] ** 2 - opposite_x[mask] ** 2
    factor[mask] *= -1
    x2_trial = (-b + factor * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    z_trial = z_start + x2_trial * (z_end - z_start)

    x2_opposite = np.zeros_like(x2_trial)
    if not np.array_equal(z_start, z_end):
        x2_opposite = (opposite_z - z_start) / (z_end - z_start)

    x2 = np.where(
        np.logical_and(
            machine_dimensions[1][0] <= z_trial, z_trial <= machine_dimensions[1][1]
        ),
        x2_trial,
        x2_opposite,
    )
    return x2 * np.sqrt(
        (x_start - x_end) ** 2 + (y_start - y_end) ** 2 + (z_start - z_end) ** 2
    )
