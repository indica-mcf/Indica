"""Coordinate system for data collected on a 1-D along through the Tokamak"""
import getpass
from typing import Tuple

import numpy as np
import xarray as xr
from xarray import DataArray

from .abstractconverter import CoordinateTransform
from ..numpy_typing import Coordinates
from ..numpy_typing import LabeledArray
from ..numpy_typing import OnlyArray

FIG_PATH = f"/home/{getpass.getuser()}/figures/Indica/transform/"


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

    Parameters
    ----------
    x_positions
        1-D array of x-locations along the transect.
    y_positions
        1-D array of y-locations along the transect.
    z_positions
        1-D array of z-locations along the transect.

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

        self.x1_name: str = "channel"
        self.x2_name: str = ""
        self.instrument_name = name
        self.name: str = f"{name}_transect_transform"

        x1 = np.arange(len(x_positions))
        self.x1: DataArray = DataArray(x1, coords=[(self.x1_name, x1)])
        self.x2: DataArray = DataArray(None)

        # TODO: add intersection with first walls to restrict possible coordinates
        self._machine_dims = machine_dimensions

        self.x: DataArray = DataArray(x_positions, coords=[(self.x1_name, self.x1)])
        self.y: DataArray = DataArray(y_positions, coords=[(self.x1_name, self.x1)])
        self.z: DataArray = DataArray(z_positions, coords=[(self.x1_name, self.x1)])
        self.R: DataArray = np.sqrt(self.x**2 + self.y**2)
        self.rho: DataArray
        self.theta: DataArray

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
        R = self.R.sel(channel=x1)
        z = self.z.sel(channel=x1)
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
        x = self.x.sel(channel=x1)
        y = self.y.sel(channel=x1)
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
        indices = np.where(np.isin(self.R, R) * np.isin(self.z, z))[0]
        x1 = self.x1.isel(channel=indices)
        x2 = DataArray(None)

        return x1, x2

    def map_profile_to_rho(
        self,
        profile_to_map: DataArray,
        t: LabeledArray = None,
        limit_to_sep=True,
        calc_rho=False,
    ) -> DataArray:
        """
        Map profile to measurement coordinates

        Parameters
        ----------
        profile_to_map
            DataArray of the profile to map
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
        profile = self.check_rho_and_profile(profile_to_map, t, calc_rho)

        coords = profile.coords
        along_los: DataArray
        if "R" in coords and "z" in coords:
            R_ = self.R
            z_ = self.z

            value_at_channels = profile.interp(R=R_, z=z_).T
        elif "rho_poloidal" in coords or "rho_toroidal" in coords:
            rho_ = self.rho
            if "theta" in coords:
                theta_ = self.theta
                value_at_channels = profile.interp(rho_poloidal=rho_, theta=theta_)
            else:
                value_at_channels = profile.interp(rho_poloidal=rho_)

            if limit_to_sep:
                value_at_channels = xr.where(
                    rho_ <= 1,
                    value_at_channels,
                    np.nan,
                )
        else:
            raise NotImplementedError("Coordinates not recognized...")

        drop_coords = [coord for coord in coords if coord != "t"]
        value_at_channels = value_at_channels.drop_vars(drop_coords)
        self.value_at_channels = value_at_channels
        self.profile = profile

        return value_at_channels

    def check_rho_and_profile(
        self, profile_to_map: DataArray, t: LabeledArray = None, calc_rho: bool = False
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
            self.convert_to_rho_theta(t=time)
        else:
            if not np.array_equal(self.rho.t, time):
                self.convert_to_rho_theta(t=time)

        # Check profile
        if not hasattr(profile_to_map, "t"):
            profile = profile_to_map.expand_dims({"t": time})  # type: ignore
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._abstract_equals(other)
