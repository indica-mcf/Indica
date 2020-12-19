"""Coordinate system representing a collection of lines of sight.
"""

from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.optimize import root
from xarray import DataArray
from xarray import zeros_like

from .abstractconverter import Coordinates
from .abstractconverter import CoordinateTransform
from ..numpy_typing import LabeledArray


class LinesOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate in this system is an index indicating which
    line-of-site a location is on. The second coordinate ranges from 0
    to 1 (inclusive) and indicates the position of a location along
    the line-of-sight. Note that diagnostic using this coordinate
    system will usually only be indexed in the first coordinate, as
    the measurements were integrated along the line-of-sight.

    If not passed to the constructor, the default grid for converting
    from the R-z system is chosen as follows:

    - The R-grid ranges from ``min(R_start.min(), R_end.min())`` to
      ``max(R_start.max(), R_end.max())`` with ``num_points`` intervals.
    - The z-grid ranges from ``min(z_start.min(), z_end.min())`` to
      ``max(z_start.max(), z_end.max())`` with ``num_points`` intervals.

    Parameters
    ----------
    R_start
        1-D array of major radii of the start for each line-of-sight.
    z_start
        1-D array of vertical positions of the start for each line-of-sight.
    T_start
        1-D array of toroidal offset for the start of each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    T_end
        1-D array of toroidal offset for the end of each line-of-sight.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.

    """

    def __init__(
        self,
        R_start: np.ndarray,
        z_start: np.ndarray,
        T_start: np.ndarray,
        R_end: np.ndarray,
        z_end: np.ndarray,
        T_end: np.ndarray,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
    ):
        lengths = _get_wall_intersection_distances(
            R_start, z_start, T_start, R_end, z_end, T_end, machine_dimensions
        )
        new_length = max(lengths)
        los_lengths = np.sqrt(
            (R_start - R_end) ** 2 + (z_start - z_end) ** 2 + (T_start - T_end) ** 2
        )
        factor = new_length / los_lengths
        self.R_start = DataArray(R_start)
        self.z_start = DataArray(z_start)
        self.T_start = DataArray(T_start)
        self._original_R_end = DataArray(R_end)
        self._original_z_end = DataArray(z_end)
        self._original_T_end = DataArray(T_end)
        self._machine_dims = machine_dimensions
        self.R_end = DataArray(R_start + factor * (R_end - R_start))
        self.z_end = DataArray(z_start + factor * (z_end - z_start))
        self.T_end = DataArray(T_start + factor * (T_end - T_start))
        self.index_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None
        self.x2_inversion: Optional[
            Callable[[LabeledArray, LabeledArray], LabeledArray]
        ] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = result and np.all(self.R_start == other.R_start)
        result = result and np.all(self.z_start == other.z_start)
        result = result and np.all(self.T_start == other.T_start)
        result = result and np.all(self.R_end == other.R_end)
        result = result and np.all(self.z_end == other.z_end)
        result = result and np.all(self.T_end == other.T_end)
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        Rs = (self.R_start[c] - self.R_start[f]) * (x1 - f) + self.R_start[f]
        Re = (self.R_end[c] - self.R_end[f]) * (x1 - f) + self.R_end[f]
        zs = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        ze = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        Ts = (self.T_start[c] - self.T_start[f]) * (x1 - f) + self.T_start[f]
        Te = (self.T_end[c] - self.T_end[f]) * (x1 - f) + self.T_end[f]
        R0 = Rs + (Re - Rs) * x2
        T0 = Ts + (Te - Ts) * x2
        z = zs + (ze - zs) * x2
        return np.sign(R0) * np.sqrt(R0 ** 2 + T0 ** 2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            x1 = x[0]
            x2 = x[1]
            c = np.ceil(x1).astype(int)
            f = np.floor(x1).astype(int)
            Rs = (self.R_start[c] - self.R_start[f]) * (x1 - f) + self.R_start[f]
            Re = (self.R_end[c] - self.R_end[f]) * (x1 - f) + self.R_end[f]
            zs = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
            ze = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
            Ts = (self.T_start[c] - self.T_start[f]) * (x1 - f) + self.T_start[f]
            Te = (self.T_end[c] - self.T_end[f]) * (x1 - f) + self.T_end[f]
            R0 = Rs + (Re - Rs) * x2
            T0 = Ts + (Te - Ts) * x2
            R = np.sign(R0) * np.sqrt(R0 ** 2 + T0 ** 2)
            dR0dx1 = (self.R_start[c] - self.R_start[f]) * (1 - x2) + (
                self.R_end[c] - self.R_end[f]
            ) * x2
            dR0dx2 = Re - Rs
            dTdx1 = (self.T_start[c] - self.T_start[f]) * (1 - x2) + (
                self.T_end[c] - self.T_end[f]
            ) * x2
            dTdx2 = Te - Ts
            dzdx1 = (self.z_start[c] - self.z_start[f]) * (1 - x2) + (
                self.z_end[c] - self.z_end[f]
            ) * x2
            dzdx2 = ze - zs
            return [
                [
                    2 / R * (R0 * dR0dx1 + T0 * dTdx1),
                    2 / R * (R0 * dR0dx2 + T0 * dTdx2),
                ],
                [dzdx1, dzdx2],
            ]

        def forward(x):
            R_prime, z_prime, _ = self.convert_to_Rz(x[0], x[1], 0)
            return [R_prime - R, z_prime - z]

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
        Rs = (self.R_start[c] - self.R_start[f]) * (x1 - f) + self.R_start[f]
        Re = (self.R_end[c] - self.R_end[f]) * (x1 - f) + self.R_end[f]
        zs = (self.z_start[c] - self.z_start[f]) * (x1 - f) + self.z_start[f]
        ze = (self.z_end[c] - self.z_end[f]) * (x1 - f) + self.z_end[f]
        Ts = (self.T_start[c] - self.T_start[f]) * (x1 - f) + self.T_start[f]
        Te = (self.T_end[c] - self.T_end[f]) * (x1 - f) + self.T_end[f]
        R = Rs + (Re - Rs) * x2
        T = Ts + (Te - Ts) * x2
        z = zs + (ze - zs) * x2
        slc1 = {direction: slice(0, -1)}
        slc2 = {direction: slice(1, None)}
        spacings = np.sqrt(
            (R[slc2] - R[slc1]) ** 2
            + (z[slc2] - z[slc1]) ** 2
            + (T[slc2] - T[slc1]) ** 2
        )
        result = zeros_like(R)
        result[slc2] = spacings.cumsum(direction)
        return result


def _get_wall_intersection_distances(
    R_start: np.ndarray,
    z_start: np.ndarray,
    T_start: np.ndarray,
    R_end: np.ndarray,
    z_end: np.ndarray,
    T_end: np.ndarray,
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
    R_start
        1-D array of major radii of the start for each line-of-sight.
    z_start
        1-D array of vertical positions of the start for each line-of-sight.
    T_start
        1-D array of toroidal offset for the start of each line-of-sight.
    R_end
        1-D array of major radii of the end for each line-of-sight.
    z_end
        1-D array of vertical positions of the end for each line-of-sight.
    T_end
        1-D array of toroidal offset for the end of each line-of-sight.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in R-z space:
        ``((Rmin, Rmax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    lengths
        The length of each line of sight for it to intersect a Tokamak wall.

    """
    opposite_R = np.where(
        R_end - R_start < 0, machine_dimensions[0][0], machine_dimensions[0][1]
    )
    opposite_z = np.where(
        z_end - z_start < 0, machine_dimensions[1][0], machine_dimensions[1][1]
    )
    # Calculate where LOS intersects opposite R-surface
    a = (R_end - R_start) ** 2 + (T_end - T_start) ** 2
    b = 2 * (R_start * (R_end - R_start) + T_start * (T_end - T_start))
    c = R_start ** 2 + T_start ** 2 - opposite_R ** 2
    factor = np.where(R_end - R_start < 0, -1, 1)
    mask = b ** 2 - 4 * a * c < 0
    # Check line of sight actually intersects the expected wall
    opposite_R[mask] = machine_dimensions[0][1]
    a[mask] = (R_end[mask] - R_start[mask]) ** 2 + (T_end[mask] - T_start[mask]) ** 2
    b[mask] = 2 * (
        R_start[mask] * (R_end[mask] - R_start[mask])
        + T_start[mask] * (T_end[mask] - T_start[mask])
    )
    c[mask] = R_start[mask] ** 2 + T_start[mask] ** 2 - opposite_R[mask] ** 2
    factor[mask] *= -1
    x2_trial = (-b + factor * np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    z_trial = z_start + x2_trial * (z_end - z_start)

    x2 = np.where(
        np.logical_and(
            machine_dimensions[1][0] <= z_trial, z_trial <= machine_dimensions[1][1]
        ),
        x2_trial,
        (opposite_z - z_start) / (z_end - z_start),
    )
    return x2 * np.sqrt(
        (R_start - R_end) ** 2 + (z_start - z_end) ** 2 + (T_start - T_end) ** 2
    )
