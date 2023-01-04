"""Coordinate system representing a collection of lines of sight.
"""

from typing import cast
from typing import Tuple

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import root
import xarray as xr
from xarray import DataArray
from xarray import Dataset
from xarray import zeros_like

from .abstractconverter_rho import Coordinates
from .abstractconverter_rho import CoordinateTransform
from ..numpy_typing import LabeledArray
from ..numpy_typing import OnlyArray


class LineOfSightTransform(CoordinateTransform):
    """Coordinate system for data collected along a number of lines-of-sight.

    The first coordinate in this system is an index indicating which
    line-of-site a location is on. The second coordinate ranges from 0
    to 1 (inclusive) and indicates the position of a location along
    the line-of-sight. Note that diagnostic using this coordinate
    system will usually only be indexed in the first coordinate, as
    the measurements were integrated along the line-of-sight.

    Parameters
    ----------
    origin_x
        An array giving x positions for the origin of the lines-of-sight.
    origin_y
        An array giving y positions for the origin of the lines-of-sight.
    origin_z
        An array giving z positions for the origin of the lines-of-sight.
    direction_x
        An array giving x positions for the direction of the lines-of-sight.
    direction_y
        An array giving y positions for the direction of the lines-of-sight.
    direction_z
        An array giving z positions for the direction of the lines-of-sight.
    name
        The name to refer to this coordinate system by, typically taken
        from the instrument it describes.
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.
    dl
        A float giving the distance between coordinates along the
        line-of-sight. Default to 0.01 metres.
    passes
        Number of passes across the plasma (e.g. typical interferometer
        with corner cube has passes=2)

    """

    def __init__(
        self,
        origin_x: OnlyArray,
        origin_y: OnlyArray,
        origin_z: OnlyArray,
        direction_x: OnlyArray,
        direction_y: OnlyArray,
        direction_z: OnlyArray,
        name: str = "",
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        dl: float = 0.01,
        passes: int = 1,
        debug=False,
    ):

        self.name = f"{name}_line_of_sight_transform"
        self.x1_name = "channel"
        self.x2_name = "los_position"
        self._machine_dims = machine_dimensions
        self.dl_target = dl
        self.passes = passes

        self.dl: list = []
        self.x: list = []
        self.y: list = []
        self.z: list = []
        self.R: list = []
        self.phi: list = []
        self.rho: list = []
        self.theta: list = []
        self.along_los: list = []
        self.los_integral: DataArray
        self.t: LabeledArray
        self.x2: list = []

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.direction_z = direction_z
        self.origin = np.array([origin_x, origin_y, origin_z]).transpose()
        self.direction = np.array([direction_x, direction_y, direction_z]).transpose()

        # Wall boundaries
        npts = 1000
        self.angles = np.linspace(0.0, 2 * np.pi, npts)
        self.x_wall_inner = self._machine_dims[0][0] * np.cos(self.angles)
        self.x_wall_outer = self._machine_dims[0][1] * np.cos(self.angles)
        self.y_wall_inner = self._machine_dims[0][0] * np.sin(self.angles)
        self.y_wall_outer = self._machine_dims[0][1] * np.sin(self.angles)
        self.z_wall_lower = self._machine_dims[1][0]
        self.z_wall_upper = self._machine_dims[1][1]

        # Number of lines of sight
        self.x1: list = list(np.arange(0, len(origin_x)))

        # Calculate start and end coordinates, R, z and phi for all LOS
        x_start, y_start, z_start = [], [], []
        x_end, y_end, z_end = [], [], []
        for channel in self.x1:
            origin = (origin_x[channel], origin_y[channel], origin_z[channel])
            direction = (
                direction_x[channel],
                direction_y[channel],
                direction_z[channel],
            )
            _start, _end = _find_wall_intersections(
                origin, direction, machine_dimensions=machine_dimensions, plot=debug
            )
            if debug:
                print(f"Channel {channel}")
                print(_start)
                print(_end)
            x_start.append(_start[0])
            y_start.append(_start[1])
            z_start.append(_start[2])
            x_end.append(_end[0])
            y_end.append(_end[1])
            z_end.append(_end[2])

        self.x_start = DataArray(x_start, coords=[(self.x1_name, self.x1)])
        self.y_start = DataArray(y_start, coords=[(self.x1_name, self.x1)])
        self.z_start = DataArray(z_start, coords=[(self.x1_name, self.x1)])
        self.x_end = DataArray(x_end, coords=[(self.x1_name, self.x1)])
        self.y_end = DataArray(y_end, coords=[(self.x1_name, self.x1)])
        self.z_end = DataArray(z_end, coords=[(self.x1_name, self.x1)])

        self.x2, self.dl = self.set_dl(dl)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        result = self._abstract_equals(other)
        result = cast(bool, result and np.all(self.x_start == other.x_start))
        result = cast(bool, result and np.all(self.z_start == other.z_start))
        result = cast(bool, result and np.all(self.y_start == other.y_start))
        result = cast(bool, result and np.all(self.x_end == other.x_end))
        result = cast(bool, result and np.all(self.z_end == other.z_end))
        result = cast(bool, result and np.all(self.y_end == other.y_end))
        result = cast(bool, result and np.all(self.dl == other.dl))
        result = cast(bool, result and np.all(self.x2 == other.x2))
        result = cast(bool, result and np.all(self.R == other.R))
        result = cast(bool, result and np.all(self.phi == other.phi))
        result = result and self._machine_dims == other._machine_dims
        return result

    def convert_to_xy(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        x = self.x_start[x1] + (self.x_end[x1] - self.x_start[x1]) * x2
        y = self.y_start[x1] + (self.y_end[x1] - self.y_start[x1]) * x2
        return x, y

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        x = self.x_start[x1] + (self.x_end[x1] - self.x_start[x1]) * x2
        y = self.y_start[x1] + (self.y_end[x1] - self.y_start[x1]) * x2
        z = self.z_start[x1] + (self.z_end[x1] - self.z_start[x1]) * x2
        return np.sqrt(x**2 + y**2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        def jacobian(x):
            # x1 = x[0]
            x2 = x[1]
            x = self.x_start + (self.x_end - self.x_start) * x2
            y = self.y_start + (self.y_end - self.y_start) * x2
            # z = self.z_start + (self.z_end - self.z_start) * x2
            dxdx1 = 0.0
            dxdx2 = self.x_end - self.x_start
            dydx1 = 0.0
            dydx2 = self.y_end - self.y_start
            dzdx1 = 0.0
            dzdx2 = self.z_end - self.z_start
            return [
                [
                    2 / x * (x * dxdx1 + y * dydx1),
                    2 / x * (x * dxdx2 + y * dydx2),
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

        raise NotImplementedError("Not checked for multiple LOS...")

        Rz = invert(R, z)
        return Rz[0], Rz[1]
        # # TODO: Consider if there is some way to invert this exactly,
        # # rather than rely on interpolation (which is necessarily
        # # inexact, as well as computationally expensive).

    def convert_to_rho(self, t: LabeledArray = None) -> Coordinates:
        """
        Convert R, z to rho given the flux surface transform
        """
        self.check_equilibrium()

        rho = []
        theta = []
        _rho: DataArray
        _theta: DataArray
        for channel in self.x1:
            _rho, _theta, _ = self.equilibrium.flux_coords(
                self.R[channel], self.z[channel], t=t
            )
            drop_vars = ["R", "z"]
            for var in drop_vars:
                if var in _rho.coords:
                    _rho = _rho.drop_vars(var)
                if var in _theta.coords:
                    _theta = _theta.drop_vars(var)

            _rho.assign_coords({self.x2_name: self.x2[channel].values})
            _theta.assign_coords({self.x2_name: self.x2[channel].values})

            rho.append(xr.where(_rho >= 0, _rho, np.nan))
            theta.append(xr.where(_rho >= 0, _theta, np.nan))

        self.t = t
        self.rho = rho
        self.theta = theta

        return rho, theta

    def distance(
        self,
        direction: str,
        x1: LabeledArray,
        x2: LabeledArray,
        t: LabeledArray,
    ) -> np.ndarray:
        """Implementation of calculation of physical distances between points
        in this coordinate system. This accounts for potential toroidal skew of
        lines.

        """
        x = self.x_start[x1] + (self.x_end[x1] - self.x_start[x1]) * x2
        y = self.y_start[x1] + (self.y_end[x1] - self.y_start[x1]) * x2
        z = self.z_start[x1] + (self.z_end[x1] - self.z_start[x1]) * x2
        spacings = np.sqrt(
            x.diff(direction) ** 2 + z.diff(direction) ** 2 + y.diff(direction) ** 2
        )
        result = zeros_like(x)
        result[{direction: slice(1, None)}] = spacings.cumsum(direction)
        return result.values

    def set_dl(
        self,
        dl: float,
    ) -> tuple:
        """
        Set spatial resolutions of the lines of sight, and calculate spatial
        coordinates along the LOS

        Parameters
        ----------
        dl
            Spatial resolution (m)

        Returns
        -------
        x2
            The list of values 0 and 1 specifying the position along the grid-line
            each point separated by dl in Cartesian coordinates
        dl
            Recomputed dl to fit an integral number of points in-between start & end
        """

        x: list = []
        y: list = []
        z: list = []
        R: list = []
        phi: list = []
        x2: list = []
        dl_new: list = []

        # Length of the lines of sight
        los_length = np.sqrt(
            (self.x_end - self.x_start) ** 2
            + (self.y_end - self.y_start) ** 2
            + (self.z_end - self.z_start) ** 2
        )

        # Find the number of points
        npts = np.ceil(los_length / dl).astype(int).values

        # Set dl, calculate dl, x, y, z, R
        for x1 in self.x1:
            ind = np.linspace(0, 1, npts[x1], dtype=float)
            _x2 = DataArray(ind, dims=self.x2_name)
            x2.append(_x2)
            dl_new.append(self.distance(self.x2_name, 0, _x2[0:2], 0)[1])

            _x, _y = self.convert_to_xy(x1, _x2, 0)
            _R, _z = self.convert_to_Rz(x1, _x2, 0)
            _phi = np.arctan2(_y, _x)
            x.append(_x)
            y.append(_y)
            R.append(_R)
            z.append(_z)
            phi.append(_phi)

        self.x2 = x2
        self.dl = dl_new
        self.x = x
        self.y = y
        self.z = z
        self.R = R
        self.phi = phi
        self.impact_xyz = self._impact_parameter_xyz()

        return x2, dl_new

    def map_to_los(
        self,
        profile_1d: DataArray,
        t: LabeledArray = None,
        limit_to_sep: bool = True,
        calc_rho: bool = False,
    ) -> list:
        """
        Map 1D profile to lines-of-sight
        TODO: extend for 2D interpolation to (R, z) instead of rho
        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0
        calc_rho
            Calculate rho for specified time-points

        Returns
        -------
            Interpolation of the input profile along the LOS
        """
        self.check_equilibrium()
        self.check_rho(t, calc_rho)

        along_los = []
        for channel in self.x1:
            if "t" in self.rho[channel].dims:
                if t.__len__() == 1:
                    rho=self.rho[channel].sel(t=t[0])
                else:
                    rho = self.rho[channel].interp(t=t)
            else:
                rho = self.rho[channel]

            if "t" in profile_1d.dims:
                if t.__len__() == 1:
                    profile = profile_1d.sel(t=t[0])
                else:
                    profile = profile_1d.interp(t=t)
            else:
                profile = profile_1d

            _along_los = profile.interp(rho_poloidal=rho)
            if limit_to_sep:
                _along_los = xr.where(
                    rho <= 1,
                    _along_los,
                    0,
                )
            along_los.append(_along_los)

        self.along_los = along_los

        return along_los

    def check_rho(self, t: LabeledArray, calc_rho: bool = False):
        """
        Check requested times
        """
        self.check_equilibrium()

        if len(self.rho) == 0 or calc_rho:
            self.convert_to_rho(t=t)
            return

        rho_t = self.rho[0].t
        if np.array_equal(rho_t, t):
            return

        if (np.min(t) > np.min(rho_t)) * (np.max(t) < np.max(rho_t)):
            return

        equil_t = self.equilibrium.rho.t
        equil_ok = (np.min(t) > np.min(equil_t)) * (np.max(t) < np.max(equil_t))
        if equil_ok:
            self.convert_to_rho(t=t)
        else:
            raise ValueError("Inserted time is not available in Equilibrium object")

    def integrate_on_los(
        self,
        profile_1d: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
        calc_rho=False,
    ) -> DataArray:
        """
        Integrate 1D profile along LOS
        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0

        Returns
        -------
        Line of sight integral along the LOS
        """
        along_los = self.map_to_los(
            profile_1d,
            t=t,
            limit_to_sep=limit_to_sep,
            calc_rho=calc_rho,
        )

        _los_integral = []
        for channel in self.x1:
            _along_los = along_los[channel].drop_vars("rho_poloidal")
            _los_integral.append(
                self.passes * _along_los.sum(self.x2_name) * self.dl[channel]
            )

        los_integral = xr.concat(_los_integral, self.x1_name).assign_coords(
            {self.x1_name: self.x1}
        )

        if len(los_integral.channel) == 1:
            los_integral = los_integral.sel(channel=0)

        self.los_integral = los_integral

        return los_integral

    def _impact_parameter_xyz(self):
        """Calculate the impact parameter in Cartesian space"""
        value = []
        index = []
        for ch in self.x1:
            distance = np.sqrt(self.x[ch] ** 2 + self.y[ch] ** 2 + self.z[ch] ** 2)
            _index = distance.argmin()
            index.append(_index)
            value.append(distance[_index])

        index = xr.concat(index, "channel")
        value = xr.concat(value, "channel")

        impact_xyz = Dataset({"index": index, "value": value})

        return impact_xyz

    def get_plasma_boundaries(self, tplot: float = None):
        boundaries = None
        if hasattr(self, "equilibrium"):
            if tplot is None:
                if self.rho is not None:
                    _rho = self.rho[0]
                else:
                    _rho = self.equilibrium.rho
                tplot = _rho.t.sel(t=_rho.t.mean(), method="nearest").values

            if self.rho is None:
                self.convert_to_rho(t=tplot)

            rho_equil = self.equilibrium.rho.sel(t=tplot, method="nearest")
            rho_equil = xr.where(rho_equil < 1.05, rho_equil, np.nan)
            core_ind = np.where(np.isfinite(rho_equil.interp(z=0)))[0]
            R_lfs = rho_equil.R[core_ind[0]].values
            R_hfs = rho_equil.R[core_ind[-1]].values
            x_plasma_inner = R_hfs * np.cos(self.angles)
            x_plasma_outer = R_lfs * np.cos(self.angles)
            y_plasma_inner = R_hfs * np.sin(self.angles)
            y_plasma_outer = R_lfs * np.sin(self.angles)

            boundaries = (
                x_plasma_inner,
                x_plasma_outer,
                y_plasma_inner,
                y_plasma_outer,
            )

        return boundaries

    def plot_los(self, tplot: float = None, orientation: str = "xy"):
        channels = self.x1
        cols = cm.gnuplot2(np.linspace(0.75, 0.1, len(channels), dtype=float))

        if hasattr(self, "equilibrium"):
            if tplot is None:
                if self.rho is not None:
                    _rho = self.rho[0]
                else:
                    _rho = self.equilibrium.rho
                tplot = _rho.t.sel(t=_rho.t.mean(), method="nearest").values

            if self.rho is None:
                self.convert_to_rho(t=tplot)

            rho_equil = self.equilibrium.rho.sel(t=tplot, method="nearest")
            rho_equil = xr.where(rho_equil < 1.05, rho_equil, np.nan)
            R_lfs = self.equilibrium.rmjo.sel(
                rho_poloidal=1, t=tplot, method="nearest"
            ).values
            R_hfs = self.equilibrium.rmji.sel(
                rho_poloidal=1, t=tplot, method="nearest"
            ).values
            x_plasma_inner = R_hfs * np.cos(self.angles)
            x_plasma_outer = R_lfs * np.cos(self.angles)
            y_plasma_inner = R_hfs * np.sin(self.angles)
            y_plasma_outer = R_lfs * np.sin(self.angles)
            x_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.cos(
                self.angles
            )
            y_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.sin(
                self.angles
            )

        if orientation is not None and orientation == "xy":
            plt.figure()
            npts = len(self.x_wall_inner)
            plt.plot(self.x_wall_inner, self.y_wall_inner, color="k")
            plt.plot(self.x_wall_outer, self.y_wall_outer, color="k")
            if hasattr(self, "equilibrium"):
                plt.plot(x_plasma_inner, y_plasma_inner, color="red")
                plt.plot(x_plasma_outer, y_plasma_outer, color="red")
                plt.plot(x_ax, y_ax, color="red", linestyle="dashed")
            for ch in self.x1:
                plt.plot(self.x[ch], self.y[ch], color=cols[ch], linewidth=2)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("scaled")

        if orientation is not None and orientation == "xz":
            plt.figure()
            plt.plot(self.x_wall_outer, [self.z_wall_upper] * npts, color="k")
            plt.plot(self.x_wall_outer, [self.z_wall_lower] * npts, color="k")
            plt.plot(self.x_wall_inner, [self.z_wall_upper] * npts, "w")
            plt.plot(self.x_wall_inner, [self.z_wall_lower] * npts, "w")
            plt.plot(
                [self.x_wall_outer.max()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            plt.plot(
                [self.x_wall_inner.max()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            plt.plot(
                [self.x_wall_outer.min()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            plt.plot(
                [self.x_wall_inner.min()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            for ch in self.x1:
                plt.plot(self.x[ch], self.z[ch], color=cols[ch], linewidth=2)
            plt.xlabel("x")
            plt.ylabel("z")
            plt.axis("scaled")

        if orientation is not None and orientation == "Rz":
            plt.figure()
            plt.plot(
                [self.x_wall_outer.max()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            plt.plot(
                [self.x_wall_inner.max()] * 2,
                [self.z_wall_lower, self.z_wall_upper],
                color="k",
            )
            plt.plot(
                [self.x_wall_inner.max(), self.x_wall_outer.max()],
                [self.z_wall_lower] * 2,
                color="k",
            )
            plt.plot(
                [self.x_wall_inner.max(), self.x_wall_outer.max()],
                [self.z_wall_upper] * 2,
                color="k",
            )
            if hasattr(self, "equilibrium"):
                rho_equil.plot.contour(levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
            for ch in self.x1:
                plt.plot(self.R[ch], self.z[ch], color=cols[ch], linewidth=2)
            plt.xlabel("R")
            plt.ylabel("z")
            plt.axis("scaled")

        if hasattr(self, "equilibrium") and orientation is None:
            plt.figure()
            for ch in self.x1:
                self.rho[ch].sel(t=tplot, method="nearest").plot(
                    color=cols[ch], linewidth=2
                )
            plt.xlabel("Path along LOS")
            plt.ylabel("Rho")


def _find_wall_intersections(
    origin: Tuple,
    direction: Tuple,
    machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (1.83, 3.9),
        (-1.75, 2.0),
    ),
    plot=False,
):
    """Function for calculating "start" and "end" positions of the line-of-sight
    given the machine dimensions.

    The end coordinate is calculated by finding the intersections with the
    machine dimensions. If the intersection hits the inner column, the first
    intersection point becomes the "end" position of the line-of-sight. If the
    intersection misses the inner column, the last intersection point with the
    outer column becomes the "end" position.

    Parameters
    ----------
    origin
        A Tuple (1x3) giving the X, Y and Z origin positions of the line-of-sight
    direction
        A Tuple (1x3) giving the X, Y and Z direction of the line-of-sight
    machine_dimensions
        A tuple giving the boundaries of the Tokamak in x-z space:
        ``((xmin, xmax), (zmin, zmax)``. Defaults to values for JET.

    Returns
    -------
    start_coordinates
        A Tuple (1x3) giving the X, Y and Z start positions of the line-of-sight
    end_coordinates
        A Tuple (1x3) giving the X, Y and Z end positions of the line-of-sight
    """

    def line(_start, _end):
        return np.linspace(_start, _end, npts)

    def extremes(_start, _end):
        return np.array(
            [_start, _end],
            dtype=float,
        )

    # Define XYZ lines for inner and outer walls
    npts = 1000
    angles = np.linspace(0.0, 2 * np.pi, npts)
    x_wall_inner = machine_dimensions[0][0] * np.cos(angles)
    x_wall_outer = machine_dimensions[0][1] * np.cos(angles)
    y_wall_inner = machine_dimensions[0][0] * np.sin(angles)
    y_wall_outer = machine_dimensions[0][1] * np.sin(angles)
    z_wall_lower = machine_dimensions[1][0]
    z_wall_upper = machine_dimensions[1][1]

    # Define XYZ lines for LOS from origin and direction vectors
    length = (
        np.ceil(
            np.max(
                [
                    machine_dimensions[0][1] * 2,
                    machine_dimensions[1][1] - machine_dimensions[1][0],
                ]
            )
        )
        * 5
    )
    x_start = origin[0]
    x_end = origin[0] + length * direction[0]
    y_start = origin[1]
    y_end = origin[1] + length * direction[1]
    z_start = origin[2]
    z_end = origin[2] + length * direction[2]

    # Find intersections in R, z plane
    x_line = line(x_start, x_end)
    y_line = line(y_start, y_end)
    z_line = line(z_start, z_end)
    R_line = np.sqrt(x_line**2 + y_line**2)
    indices = np.where(
        (R_line >= machine_dimensions[0][0])
        * (R_line <= machine_dimensions[0][1])
        * (z_line >= machine_dimensions[1][0])
        * (z_line <= machine_dimensions[1][1])
    )[0]
    if len(indices) > 0:
        x_start = x_line[indices][0]
        x_end = x_line[indices][-1]
        y_start = y_line[indices][0]
        y_end = y_line[indices][-1]
        z_start = z_line[indices][0]
        z_end = z_line[indices][-1]

    # Find intersections with inner wall
    xx, yx, ix, jx = intersection(
        extremes(x_start, x_end), extremes(y_start, y_end), x_wall_inner, y_wall_inner
    )
    if len(xx) > 0:
        x_line = line(x_start, x_end)
        y_line = line(y_start, y_end)
        z_line = line(z_start, z_end)
        index = []
        for i in range(len(xx)):
            index.append(
                np.argmin(np.sqrt((xx[i] - x_line) ** 2 + (yx[i] - y_line) ** 2))
            )
        indices = np.arange(np.min(index))
        x_start = x_line[indices][0]
        x_end = x_line[indices][-1]
        y_start = y_line[indices][0]
        y_end = y_line[indices][-1]
        z_start = z_line[indices][0]
        z_end = z_line[indices][-1]

    if plot:
        x_line = line(x_start, x_end)
        y_line = line(y_start, y_end)
        z_line = line(z_start, z_end)
        R_line = np.sqrt(x_line**2 + y_line**2)

        plt.figure()
        plt.plot(x_wall_inner, y_wall_inner, color="orange")
        plt.plot(x_wall_outer, y_wall_outer, color="orange")
        plt.plot(x_line, y_line, "k", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")

        plt.figure()
        plt.plot(x_wall_outer, [z_wall_upper] * npts, color="orange")
        plt.plot(x_wall_outer, [z_wall_lower] * npts, color="orange")
        plt.plot(x_wall_inner, [z_wall_upper] * npts, "w")
        plt.plot(x_wall_inner, [z_wall_lower] * npts, "w")
        plt.plot([x_wall_outer.max()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot([x_wall_inner.max()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot([x_wall_outer.min()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot([x_wall_inner.min()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot(x_line, z_line, "k", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("z")

        plt.figure()
        plt.plot([x_wall_outer.max()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot([x_wall_inner.max()] * 2, [z_wall_lower, z_wall_upper], color="orange")
        plt.plot(
            [x_wall_inner.max(), x_wall_outer.max()], [z_wall_lower] * 2, color="orange"
        )
        plt.plot(
            [x_wall_inner.max(), x_wall_outer.max()], [z_wall_upper] * 2, color="orange"
        )
        plt.plot(R_line, z_line, "k", linewidth=2)
        plt.xlabel("R")
        plt.ylabel("z")

    return (x_start, y_start, z_start), (x_end, y_end, z_end)


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
    INTERSECTIONS Intersections of curves.
       Computes the (x,y) locations where two curves intersect.  The curves
       can be broken with NaNs or have vertical segments.
    usage:
    x,y=intersection(x1,y1,x2,y2)
        Example:
        a, b = 1, 2
        phi = np.linspace(3, 10, 100)
        x1 = a*phi - b*np.sin(phi)
        y1 = a - b*np.cos(phi)
        x2=phi
        y2=np.sin(phi)+2
        x,y, ix, iy=intersection(x1,y1,x2,y2)
        plt.plot(x1,y1,c='r')
        plt.plot(x2,y2,c='g')
        plt.plot(x,y,'*k')
        plt.show()
    """
    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except np.linalg.LinAlgError:
            T[:, i] = np.NaN

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    indii = ii[in_range] + T[0, in_range]
    indjj = jj[in_range] + T[1, in_range]

    return xy0[:, 0], xy0[:, 1], indii, indjj
