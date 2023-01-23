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
from .abstractconverter_rho import find_wall_intersections
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
    ):

        self.name = f"{name}_line_of_sight_transform"
        self.x1_name = "channel"
        self.x2_name = "los_position"
        self._machine_dims = machine_dimensions
        self.dl_target = dl
        self.passes = passes

        self.dl: float
        self.dl_all: list
        self.x: DataArray
        self.y: DataArray
        self.z: DataArray
        self.R: DataArray
        self.phi: DataArray
        self.rho: DataArray
        self.theta: DataArray
        self.along_los: DataArray
        self.los_integral: DataArray
        self.t: LabeledArray
        self.x2: LabeledArray

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.direction_z = direction_z
        self.origin = np.array([origin_x, origin_y, origin_z]).transpose()
        self.direction = np.array([direction_x, direction_y, direction_z]).transpose()

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
            _start, _end = find_wall_intersections(
                origin, direction, machine_dimensions=machine_dimensions
            )
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

        self.x2, self.dl = self.set_dl(self.dl_target)

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
        c = np.ceil(x1).astype(int)
        f = np.floor(x1).astype(int)
        x_s = (self.x_start[c] - self.x_start[f]) * (x1 - f) + self.x_start[f]
        x_e = (self.x_end[c] - self.x_end[f]) * (x1 - f) + self.x_end[f]
        y_s = (self.y_start[c] - self.y_start[f]) * (x1 - f) + self.y_start[f]
        y_e = (self.y_end[c] - self.y_end[f]) * (x1 - f) + self.y_end[f]
        x = x_s + (x_e - x_s) * x2
        y = y_s + (y_e - y_s) * x2

        return x, y

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
        x = x_s + (x_e - x_s) * x2
        y = y_s + (y_e - y_s) * x2
        z = z_s + (z_e - z_s) * x2

        return np.sqrt(x ** 2 + y ** 2), z

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:

        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a 'convert_from_Rz' "
            "method."
        )

    def convert_to_rho(self, t: LabeledArray = None) -> Coordinates:
        """
        Convert R, z to rho given the flux surface transform
        """
        self.check_equilibrium()

        _t = np.array(t)
        if np.size(t) == 1:
            _t = float(_t)

        rho = []
        theta = []
        _rho: DataArray
        _theta: DataArray
        for chan in self.x1:
            _rho, _theta, _ = self.equilibrium.flux_coords(
                self.R.sel(channel=chan), self.z.sel(channel=chan), t=_t
            )
            drop_vars = ["R", "z"]
            for var in drop_vars:
                if var in _rho.coords:
                    _rho = _rho.drop_vars(var)
                if var in _theta.coords:
                    _theta = _theta.drop_vars(var)

            rho.append(xr.where(_rho >= 0, _rho, np.nan))
            theta.append(xr.where(_rho >= 0, _theta, np.nan))

        self.t = _t
        self.rho = xr.concat(rho, "channel")
        self.theta = xr.concat(theta, "channel")

        return self.rho, self.theta

    def distance(
        self, direction: str, x1: LabeledArray, x2: LabeledArray, t: LabeledArray,
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

    def set_dl(self, dl: float,) -> tuple:
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

        TODO: modify to have all LOS with identical dl !!!
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
            dl_new.append(self.distance(self.x2_name, 0, _x2[0:2], 0)[1])

            _x, _y = self.convert_to_xy(x1, _x2, 0)
            _R, _z = self.convert_to_Rz(x1, _x2, 0)
            _phi = np.arctan2(_y, _x)
            x.append(_x)
            y.append(_y)
            R.append(_R)
            z.append(_z)
            phi.append(_phi)

        # Make all LOS of equal length, setting to NaN everything outside of of bounds
        npts = np.max([len(_x) for _x in x])
        x2 = np.linspace(0, 1, npts)
        _x, _y, _z, _phi = [], [], [], []
        for chan in self.x1:
            _x2 = x2[: len(x[chan])]
            x_tmp = x[chan].assign_coords({"los_position": _x2})
            y_tmp = y[chan].assign_coords({"los_position": _x2})
            z_tmp = z[chan].assign_coords({"los_position": _x2})
            phi_tmp = z[chan].assign_coords({"los_position": _x2})
            _x.append(x_tmp.interp(los_position=x2))
            _y.append(y_tmp.interp(los_position=x2))
            _z.append(z_tmp.interp(los_position=x2))
            _phi.append(phi_tmp.interp(los_position=x2))

        self.x2 = x2
        dl = np.mean(dl_new)
        self.dl_all = DataArray(dl_new, coords=[("channel", self.x1)])
        self.dl = dl
        self.x = xr.concat(_x, "channel")
        self.y = xr.concat(_y, "channel")
        self.z = xr.concat(_z, "channel")
        self.phi = xr.concat(_phi, "channel")
        self.R = np.sqrt(self.x ** 2 + self.y ** 2)
        self.impact_parameter = self.calc_impact_parameter()

        return x2, dl

    def map_to_los(
        self,
        profile_to_map: DataArray,
        t: LabeledArray = None,
        limit_to_sep: bool = True,
        calc_rho: bool = False,
    ) -> list:
        """
        Map profile to lines-of-sight
        TODO: extend for 2D interpolation to (R, z) instead of rho
        Parameters
        ----------
        profile_to_map
            DataArray of the profile to integrate
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

        profile = self.check_rho_and_profile(t, profile_to_map, calc_rho=calc_rho)
        impact_rho = self.rho.min("los_position")

        along_los = profile.interp(rho_poloidal=self.rho).drop_vars("rho_poloidal")
        if limit_to_sep:
            along_los = xr.where(self.rho <= 1, along_los, np.nan,)

        self.along_los = along_los
        self.impact_rho = impact_rho

        return along_los

    def check_rho_and_profile(
        self, t: LabeledArray, profile_to_map: DataArray, calc_rho: bool = False
    ) -> DataArray:
        """
        Check requested times
        """

        time = np.array(t)
        if time.size == 1:
            time = float(time)

        equil_t = self.equilibrium.rho.t
        equil_ok = (np.min(time) >= np.min(equil_t)) * (np.max(time) <= np.max(equil_t))
        if not equil_ok:
            print(f"Available equilibrium times {np.array(equil_t)}")
            raise ValueError(
                f"Inserted time {time} is not available in Equilibrium object"
            )

        # Make sure rho.t == requested time
        if not hasattr(self, "rho") or calc_rho:
            self.convert_to_rho(t=time)
        else:
            if not np.array_equal(self.rho.t, time):
                self.convert_to_rho(t=time)

        # Check profile
        if not hasattr(profile_to_map, "t"):
            profile = profile_to_map.expand_dims({"t": time})
        else:
            profile = profile_to_map

        if np.size(time) == 1:
            if np.isclose(profile.t, time, rtol=1.0e-4):
                if "t" in profile_to_map.dims:
                    profile = profile.sel(t=time, method="nearest")
            else:
                raise ValueError("Profile does not include requested time")
        else:
            prof_t = profile.t
            range_ok = (np.min(time) >= np.min(prof_t)) * (
                np.max(time) <= np.max(prof_t)
            )
            if range_ok:
                profile = profile.interp(t=time)
            else:
                raise ValueError("Profile does not include requested time")

        return profile

    def integrate_on_los(
        self,
        profile_to_map: DataArray,
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
            profile_to_map, t=t, limit_to_sep=limit_to_sep, calc_rho=calc_rho,
        )
        los_integral = self.passes * along_los.sum("los_position") * self.dl

        if len(los_integral.channel) == 1:
            los_integral = los_integral.sel(channel=0)

        self.los_integral = los_integral

        return los_integral

    def calc_impact_parameter(self):
        """Calculate the impact parameter in Cartesian space"""
        impact = []
        index = []
        x = []
        y = []
        z = []
        R = []
        for ch in self.x1:
            distance = np.sqrt(
                self.x.sel(channel=ch) ** 2
                + self.y.sel(channel=ch) ** 2
                + self.z.sel(channel=ch) ** 2
            )
            _index = distance.argmin()
            index.append(_index)
            impact.append(distance[_index])
            x.append(self.x.sel(channel=ch)[_index])
            y.append(self.y.sel(channel=ch)[_index])
            z.append(self.z.sel(channel=ch)[_index])
            R.append(
                np.sqrt(
                    self.x.sel(channel=ch)[_index] ** 2
                    + self.y.sel(channel=ch)[_index] ** 2
                )
            )

        impact = Dataset(
            {
                "index": xr.concat(index, "channel"),
                "value": xr.concat(impact, "channel"),
                "x": xr.concat(x, "channel"),
                "y": xr.concat(y, "channel"),
                "z": xr.concat(z, "channel"),
                "R": xr.concat(R, "channel"),
            }
        )

        return impact

    def plot_los(
        self,
        tplot: float = None,
        orientation: str = "xy",
        plot_all: bool = False,
        figure: bool = True,
    ):
        channels = np.array(self.x1)
        cols = cm.gnuplot2(np.linspace(0.75, 0.1, np.size(channels), dtype=float))

        wall_bounds, angles = self.get_machine_boundaries(
            machine_dimensions=self._machine_dims
        )
        if hasattr(self, "equilibrium"):
            if tplot is None:
                tplot = np.mean(self.equilibrium.rho.t)
            equil_bounds, angles, rho_equil = self.get_equilibrium_boundaries(tplot)
            x_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.cos(
                angles
            )
            y_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.sin(
                angles
            )

        if orientation == "xy" or plot_all:
            if figure:
                plt.figure()
            plt.plot(wall_bounds["x_in"], wall_bounds["y_in"], color="k")
            plt.plot(wall_bounds["x_out"], wall_bounds["y_out"], color="k")
            if hasattr(self, "equilibrium"):
                plt.plot(equil_bounds["x_in"], equil_bounds["y_in"], color="red")
                plt.plot(equil_bounds["x_out"], equil_bounds["y_out"], color="red")
                plt.plot(x_ax, y_ax, color="red", linestyle="dashed")
            for ch in self.x1:
                plt.plot(
                    self.x.sel(channel=ch),
                    self.y.sel(channel=ch),
                    color=cols[ch],
                    linewidth=2,
                )
                plt.plot(
                    self.impact_parameter["x"][ch],
                    self.impact_parameter["y"][ch],
                    color=cols[ch],
                    marker="o",
                )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("scaled")

        if orientation == "Rz" or plot_all:
            if figure:
                plt.figure()
            plt.plot(
                [wall_bounds["x_out"].max()] * 2,
                [wall_bounds["z_low"], wall_bounds["z_up"]],
                color="k",
            )
            plt.plot(
                [wall_bounds["x_in"].max()] * 2,
                [wall_bounds["z_low"], wall_bounds["z_up"]],
                color="k",
            )
            plt.plot(
                [wall_bounds["x_in"].max(), wall_bounds["x_out"].max()],
                [wall_bounds["z_low"]] * 2,
                color="k",
            )
            plt.plot(
                [wall_bounds["x_in"].max(), wall_bounds["x_out"].max()],
                [wall_bounds["z_up"]] * 2,
                color="k",
            )
            if hasattr(self, "equilibrium"):
                rho_equil.plot.contour(levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
            for ch in channels:
                plt.plot(
                    self.R.sel(channel=ch),
                    self.z.sel(channel=ch),
                    color=cols[ch],
                    linewidth=2,
                )
                plt.plot(
                    self.impact_parameter["R"][ch],
                    self.impact_parameter["z"][ch],
                    color=cols[ch],
                    marker="o",
                )
            plt.xlabel("R")
            plt.ylabel("z")
            plt.axis("scaled")

        if hasattr(self, "equilibrium") and plot_all:
            if figure:
                plt.figure()
            for ch in channels:
                self.rho.sel(channel=ch, t=tplot, method="nearest").plot(
                    color=cols[ch], linewidth=2
                )
            plt.xlabel("Path along LOS")
            plt.ylabel("Rho")


def example_run(
    pulse: int = None,
    plasma=None,
    origin: LabeledArray = None,
    direction: LabeledArray = None,
    plot=False,
    t: LabeledArray = None,
):
    from indica.models.equilibrium import fake_equilibrium_data, Equilibrium
    from indica.models.plasma import example_run as example_plasma
    from indica.readers import ST40Reader

    if plasma is None:
        plasma = example_plasma(pulse=pulse)
    equilibrium = plasma.equilibrium
    machine_dims = plasma.machine_dimensions

    if origin is None or direction is None:
        nchannels = 11
        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.17
        los_end[:, 1] = 0.0
        los_end[:, 2] = np.linspace(0.53, -0.53, nchannels)
        los_start = np.array([[1.0, 0, 0]] * los_end.shape[0])
        origin = los_start
        direction = los_end - los_start

    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name="",
        machine_dimensions=machine_dims,
        passes=1,
    )
    los_transform.set_equilibrium(equilibrium)

    if plot:
        los_transform.plot_los()

    return plasma, los_transform

    # Test rho calculation
    t_types = [
        float(plasma.t.mean()),
        plasma.t.mean(),
        np.array([float(plasma.t.mean())]),
        [float(plasma.t.mean())],
        plasma.t,
    ]
    for t in t_types:
        print(t)
        rho, theta = los_transform.convert_to_rho(t=t)

    # Test profile interpolation
    # plt.ion()
    for i, t in enumerate(t_types):
        profile_to_map = plasma.electron_density.sel(t=t, method="nearest")
        along_los = los_transform.map_to_los(profile_to_map, t=t)
        los_int = los_transform.integrate_on_los(profile_to_map, t=t)
        # plt.figure()
        # along_los.plot()
        # plt.figure()
        # los_int.plot()

    return plasma, los_transform
