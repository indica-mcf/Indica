"""Coordinate system for data collected on a 1-D along through the Tokamak"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from xarray import DataArray
from xarray import Dataset
from xarray import Variable
from matplotlib import cm
import matplotlib.pylab as plt

from .abstractconverter_rho import CoordinateTransform
from ..numpy_typing import Coordinates
from ..numpy_typing import LabeledArray
from ..numpy_typing import OnlyArray


class TransectCoordinates(CoordinateTransform):
    """Coordinate system for measurements taken along a 1-D line through
    the Tokamak.

    The first spatial position corresponds to an arbitrary index along
    the length of the transect. The second is the vertical offset from
    the transect. Typically, all data will fall directly on the
    transect, but it is useful to have the second coordinate so that
    the system is general and to allow conversions to other systems.

    The coordinates along the transect are the indices of the
    positions provided when intialisign the object.

    The default grid for coordinate transforms in this system is a 1-D
    array corresponding to the positions along the axis specified in
    the parmeters used to intialise the object.

    The implementation currently makes use of interpolation, so that
    it is completely general for nonuniformly spaced points and even
    for a curved line (although the latter would be only
    approximated). However, this has some computational overhead, so
    it may be changed in future.

    Parameters
    ----------
    R_positions
        1-D array of major radii of locations along the transect.
    z_positions
        1-D array of vertical position of locations along the transect.

    """

    def __init__(
        self,
        x_positions: OnlyArray,
        y_positions: OnlyArray,
        z_positions: OnlyArray,
        name: str,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
    ):
        if np.shape(x_positions) != np.shape(z_positions) or np.shape(
            y_positions
        ) != np.shape(z_positions):
            raise ValueError("x_, y_ and z_positions must have the same dimensions.")

        self.x1_name = "channel"
        self.x2_name = ""
        self.name = f"{name}_transect_transform"

        x1 = np.arange(len(x_positions))
        self.x1 = DataArray(x1, coords=[(self.x1_name, x1)])
        self.x2 = DataArray(None)

        # TODO: add intersection with first walls to restrict possible coordinates
        self._machine_dims = machine_dimensions

        R_positions = np.sqrt(x_positions ** 2 + y_positions ** 2)
        self.x_interp = interp1d(
            self.x1, x_positions, copy=False, fill_value="extrapolate"
        )
        self.y_interp = interp1d(
            self.x1, y_positions, copy=False, fill_value="extrapolate"
        )
        self.z_interp = interp1d(
            self.x1, z_positions, copy=False, fill_value="extrapolate"
        )

        self.invert_x = interp1d(
            x_positions, self.x1, copy=False, fill_value="extrapolate"
        )
        self.invert_y = interp1d(
            y_positions, self.x1, copy=False, fill_value="extrapolate"
        )
        self.invert_z = interp1d(
            z_positions, self.x1, copy=False, fill_value="extrapolate"
        )
        self.invert_R = interp1d(
            R_positions, self.x1, copy=False, fill_value="extrapolate",
        )

        x, y = self.convert_to_xy(self.x1, self.x2, None)
        R, z = self.convert_to_Rz(self.x1, self.x2, None)
        self.x = x
        self.y = y
        self.z = z
        self.R = R
        self.rho: DataArray = None

    def convert_to_Rz(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        Parameters
        ----------
        x1
            The channel
        x2
            Set to None: not needed in this coordinate transform
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        R
            Major radius coordinate
        z
            Height coordinate

        """
        dims = x1.dims if isinstance(x1, (DataArray, Variable, Dataset)) else None
        coords = x1.coords if isinstance(x1, (DataArray, Dataset)) else None
        R = DataArray(
            np.sqrt(self.x_interp(x1) ** 2 + self.y_interp(x1) ** 2), coords, dims
        )
        z = DataArray(self.z_interp(x1), coords, dims)
        return R, z

    def convert_to_xy(
        self, x1: LabeledArray, x2: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from this coordinate to the R-z coordinate system.

        Parameters
        ----------
        x1
            The channel
        x2
            Set to None: not needed in this coordinate transform
        t
            The time coordinate (if there is one, otherwise ``None``)

        Returns
        -------
        x
            x Cartesian coordinate
        y
            y Cartesian coordinate

        """
        dims = x1.dims if isinstance(x1, (DataArray, Variable, Dataset)) else None
        coords = x1.coords if isinstance(x1, (DataArray, Dataset)) else None
        x = DataArray(self.x_interp(x1), coords, dims)
        y = DataArray(self.y_interp(x1), coords, dims)
        return x, y

    def convert_from_Rz(
        self, R: LabeledArray, z: LabeledArray, t: LabeledArray
    ) -> Coordinates:
        """Convert from the master coordinate system to this coordinate.

        Parameters
        ----------
        R
            Major radius coordinate
        z
            z coordinate
        t
            Time coordinate)

        Returns
        -------
        x1
            The channel
        x2
            Not applicable to this transform: None

        """
        dims = R.dims if isinstance(R, (DataArray, Variable, Dataset)) else None
        coords = R.coords if isinstance(R, (DataArray, Dataset)) else None
        x1 = DataArray(self.invert_R(R), coords, dims)
        x2 = DataArray(None)

        return x1, x2

    def convert_to_rho(self, t: LabeledArray = None) -> Coordinates:
        """
        Convert R, z to rho given the equilibrium object
        """
        if not hasattr(self, "equilibrium"):
            raise Exception("Set equilibrium object to convert (R,z) to rho")

        rho, theta, _ = self.equilibrium.flux_coords(self.R, self.z, t=t)
        drop_vars = ["R", "z"]
        for var in drop_vars:
            if var in rho.coords:
                rho = rho.drop_vars(var)
            if var in theta.coords:
                theta = theta.drop_vars(var)

        self.t = t
        self.rho = rho
        self.theta = theta

        return rho, theta

    def map_to_rho(
        self,
        profile_1d: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
        calc_rho=False,
    ) -> list:
        """
        Map 1D profile to measurement coordinates

        Parameters
        ----------
        profile_1d
            DataArray of the 1D profile to integrate
        t
            Time for interpolation
        limit_to_sep
            Set to True if values outside of separatrix are to be set to 0
        calc_rho
            Calculate rho for specified time-point

        Returns
        -------
            Interpolation of the input profile on the diagnostic channels
        """
        self.check_equilibrium()
        self.check_rho(t, calc_rho)

        if "t" in self.rho.dims:
            rho = self.rho.interp(t=t)
        else:
            rho = self.rho

        value_at_channels = profile_1d.interp(rho_poloidal=rho)
        if limit_to_sep:
            value_at_channels = xr.where(rho <= 1, value_at_channels, 0,)

        self.value_at_channels = value_at_channels

        return value_at_channels

    def check_rho(self, t: LabeledArray, calc_rho: bool = False):
        """
        Check requested times
        """
        if self.rho is None or calc_rho:
            self.convert_to_rho(t=t)
            return

        if np.array_equal(self.t, t):
            return

        t_min = np.min(t)
        t_max = np.max(t)
        if (t_min >= np.min(self.t)) * (t_max <= np.max(self.t)):
            return

        equil_t = self.equilibrium.rho.t
        equil_ok = (t_min >= np.min(equil_t)) * (t_max <= np.max(equil_t))
        if equil_ok:
            self.convert_to_rho(t=t)
        else:
            raise ValueError("Inserted time is not available in Equilibrium object")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._abstract_equals(other)

    def plot_los(self, tplot: float = None, orientation: str = "xy", plot_all=False):
        channels = self.x1
        cols = cm.gnuplot2(np.linspace(0.75, 0.1, len(channels), dtype=float))

        wall_bounds, angles = self.get_machine_boundaries(
            machine_dimensions=self._machine_dims
        )
        if hasattr(self, "equilibrium"):
            equil_bounds, angles, rho_equil = self.get_equilibrium_boundaries(tplot)
            x_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.cos(
                angles
            )
            y_ax = self.equilibrium.rmag.sel(t=tplot, method="nearest").values * np.sin(
                angles
            )

        if orientation == "xy" or plot_all:
            plt.figure()
            plt.plot(wall_bounds["x_in"], wall_bounds["y_in"], color="k")
            plt.plot(wall_bounds["x_out"], wall_bounds["y_out"], color="k")
            if hasattr(self, "equilibrium"):
                plt.plot(equil_bounds["x_in"], equil_bounds["y_in"], color="red")
                plt.plot(equil_bounds["x_out"], equil_bounds["y_out"], color="red")
                plt.plot(x_ax, y_ax, color="red", linestyle="dashed")
            for ch in self.x1:
                plt.scatter(self.x[ch], self.y[ch], color=cols[ch], marker="o")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("scaled")

        if orientation == "Rz" or plot_all:
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
            for ch in self.x1:
                plt.scatter(self.R[ch], self.z[ch], color=cols[ch], marker="o")
            plt.xlabel("R")
            plt.ylabel("z")
            plt.axis("scaled")

        if hasattr(self, "equilibrium") and plot_all:
            plt.figure()
            plt.plot(
                self.rho.channel,
                self.rho.sel(t=tplot, method="nearest"),
                color="k",
            )
            for ch in self.x1:
                plt.plot(
                    self.rho.channel[ch],
                    self.rho.sel(channel=ch, t=tplot, method="nearest"),
                    color=cols[ch],
                    marker="o",
                )
            plt.xlabel("Channel")
            plt.ylabel("Rho")
