from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import xarray as xr
from xarray import apply_ufunc
from xarray import DataArray
from xarray import where

from indica.utilities import check_time_present
from .numpy_typing import FloatOrDataArray
from .numpy_typing import LabeledArray
from .numpy_typing import OnlyArray

_FLUX_TYPES = ["poloidal", "toroidal"]


class Equilibrium:
    """Class to hold and map equilibrium data.

    equilibrium_data
        Collection of equilibrium data
    R_shift
        Radial shift to test diagnostic mapping (positive for diagnostics)
        Either a float for all time slices or a DataArray with coord 't'
    z_shift
        Vertical z shift to test diagnostic mapping (positive for diagnostics)
        Either a float for all time slices or a DataArray with coord 't'

    TODO: should this class go in a sub-folder??
    """

    def __init__(
        self,
        equilibrium_data: Dict[str, DataArray],
        R_shift: FloatOrDataArray = 0.0,
        z_shift: FloatOrDataArray = 0.0,
    ):
        self.f: DataArray
        self.psi: DataArray
        self.psin: DataArray
        self.psi_boundary: DataArray
        self.psi_axis: DataArray
        self.ftor: DataArray
        self.rbnd: DataArray
        self.zbnd: DataArray
        self.rmag: DataArray
        self.zmag: DataArray

        self._data = equilibrium_data
        # Assign all equilibrium data as class attributes
        for k, v in equilibrium_data.items():
            setattr(self, k, v)

        # Substitute coordinates "psin" with "rhop"
        self.rhop = np.sqrt(self.psin)
        for k, v in equilibrium_data.items():
            if "psin" in v.dims:
                setattr(
                    self,
                    k,
                    v.assign_coords(rhop=("psin", self.rhop.data))
                    .swap_dims({"psin": "rhop"})
                    .drop_vars("psin"),
                )

        # Calculate volume and area if ajac and vjac are given
        if hasattr(self, "vjac") and hasattr(self, "ajac"):
            dpsin = self.psin[1] - self.psin[0]
            self.volume = (self.vjac * dpsin).cumsum("rhop")
            self.area = (self.ajac * dpsin).cumsum("rhop")

        # Calculate upper and lower boundaries
        self.zx_up = self.zbnd.min("index")
        self.zx_low = self.zbnd.max("index")

        # Calculate rho-toroidal and 2D rho matrix
        self.rhot = np.sqrt(
            (self.ftor - self.ftor.sel(rhop=0.0))
            / (self.ftor.sel(rhop=1.0) - self.ftor.sel(rhop=0.0))
        )
        self.rhop = np.sqrt(
            (self.psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)
        )

        # TODO: shift of equilibrium is a bad idea, but useful...
        #   - psi (R, z) is restricted to new limits to avoid NaNs
        #   - volume is not changed so is inconsistent!
        if isinstance(R_shift, float):
            R_offset = xr.full_like(self.t, R_shift)
        else:
            R_offset = R_shift.interp(t=self.t, kwargs={"fill_value": 0})
        if isinstance(z_shift, float):
            z_offset = xr.full_like(self.t, z_shift)
        else:
            z_offset = z_shift.interp(t=self.t, kwargs={"fill_value": 0})

        self.R_offset = xr.where(np.isfinite(R_offset), R_offset, 0)
        self.z_offset = xr.where(np.isfinite(z_offset), z_offset, 0)
        if np.any(np.abs(R_offset) > 0) or np.any(np.abs(z_offset) > 0):
            R_new = self.psi.R + self.R_offset
            z_new = self.psi.z + self.z_offset
            R_range = slice(R_new.min("R"), R_new.max("R"))
            z_range = slice(z_new.min("z"), z_new.max("z"))
            self.psi = self.psi.interp(R=R_new, z=z_new).sel(R=R_range, z=z_range)
            self.rhop = self.rhop.interp(R=R_new, z=z_new).sel(R=R_range, z=z_range)
            self.rmag -= self.R_offset
            self.zmag -= self.z_offset
            self.rbnd -= self.R_offset
            self.zbnd -= self.z_offset
            self.zx_low -= self.z_offset
            if hasattr(self, "rmji"):
                self.rmji -= self.R_offset
            if hasattr(self, "rmjo"):
                self.rmjo -= self.R_offset

        if np.any(np.isnan(self.rhop)):
            self.rhop = xr.where(self.rhop > 0, self.rhop, 0.0)
        self.t = self.rhop.t
        self.Rmin = min(self.rhop.coords["R"])
        self.Rmax = max(self.rhop.coords["R"])
        self.zmin = min(self.rhop.coords["z"])
        self.zmax = max(self.rhop.coords["z"])
        self.corner_angles = [
            np.arctan2(self.zmin - self.zmag, self.Rmax - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmax - self.zmag, self.Rmax - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmax - self.zmag, self.Rmin - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmin - self.zmag, self.Rmin - self.rmag) % (2 * np.pi),
        ]

    def Bfield(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        full_Rz: bool = False,
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray, LabeledArray]:
        """
        Magnetic field components at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """

        if t is not None:
            check_time_present(t, self.t)
            psi = self.psi.interp(t=t, method="nearest", assume_sorted=True)
            f = self.f.interp(t=t, method="nearest", assume_sorted=True)
            _rhop, _, _ = self.flux_coords(R, z, t)
        else:
            t = self.rhop.coords["t"]
            psi = self.psi
            f = self.f
            _rhop, _, _ = self.flux_coords(R, z)

        dpsi_dR = psi.differentiate("R").interp(R=R, z=z)
        dpsi_dz = psi.differentiate("z").interp(
            R=R,
            z=z,
        )

        b_R = -(np.float64(1.0) / R) * dpsi_dz  # type: ignore
        b_R.name = "Radial magnetic field"
        b_R = b_R.T

        b_z = (np.float64(1.0) / R) * dpsi_dR  # type: ignore
        b_z.name = "Vertical Magnetic Field (T)"
        b_z = b_z.T
        _rhop = where(_rhop > np.float64(0.0), _rhop, np.float64(-1.0) * _rhop)

        f = f.interp(rhop=_rhop)
        f.name = self.f.name
        b_T = f / R
        b_T.name = "Toroidal Magnetic Field (T)"

        if full_Rz:
            _b_T = b_T.interp(R=self.rmag, z=self.zmag) * self.rmag / self.rhop.R
            _b_T = _b_T.drop(["z", "rhop"]).expand_dims(dim={"z": self.rhop.z})
            b_T = _b_T.interp(R=R, z=z)

        return b_R, b_z, b_T, t

    def Btot(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        full_Rz: bool = False,
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Total magnetic field at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t, full_Rz=full_Rz)
        b_Tot = np.sqrt(
            b_R ** np.float64(2.0) + b_z ** np.float64(2.0) + b_T ** np.float64(2.0)
        )
        b_Tot.name = "Total Magnetic Field (T)"

        return b_Tot, t

    def Br(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Radial magnetic field at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)

        return b_R, t

    def Bz(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Vertical magnetic field at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)

        return b_z, t

    def Bt(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Toroidal magnetic field at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)
        return b_T, t

    def Bp(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Poloidal magnetic field at the desired time and location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)
        b_Pol = np.sqrt(b_R ** np.float64(2.0) + b_z ** np.float64(2.0))
        b_Pol.name = "Poloidal Magnetic Field (T)"
        return b_Pol, t

    def R_lfs(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        LFS major radius position of the given flux surface

        rho - Normalized flux coordinate values.
        t - Times (s).
        kind - Type of flux coordinate: "toroidal", "poloidal"
        """
        if t is None:
            rmjo = self.rmjo
            t = self.rmjo.coords["t"]
        else:
            check_time_present(t, self.t)
            rmjo = self.rmjo.interp(t=t, method="nearest")
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        R = rmjo.indica.interp2d(rhop=rhop, method="cubic")

        return R, t

    def R_hfs(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        HFS major radius position of the given flux surface

        rho - Normalized flux coordinate values.
        t - Times (s).
        kind - Type of flux coordinate: "toroidal", "poloidal"
        """
        if t is None:
            rmji = self.rmji
            t = self.rmji.coords["t"]
        else:
            check_time_present(t, self.t)
            rmji = self.rmji.interp(t=t, method="nearest")
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        R = rmji.indica.interp2d(rhop=rhop, method="cubic")

        return R, t

    def minor_radius(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """
        Minor radius of the given flux surface at a desired poloidal angle

        rho - Normalized flux coordinate values.
        theta - Poloidal angle.
        t - Times (s).
        kind - Type of flux coordinate: "toroidal", "poloidal"
        """
        ngrid = 100
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        theta = theta % (2 * np.pi)
        if t is not None:
            check_time_present(t, self.t)
            corner_angles = [
                angle.interp(t=t, method="nearest") for angle in self.corner_angles
            ]
            R0 = self.rmag.interp(t=t, method="nearest")
            z0 = self.zmag.interp(t=t, method="nearest")
            reference_rhos = self.rhop.interp(t=t, method="nearest")
            reference_rhos.name = self.rhop.name
        else:
            corner_angles = self.corner_angles
            R0 = self.rmag
            z0 = self.zmag
            reference_rhos = self.rhop
            t = self.rhop.coords["t"]
        minor_rad_max = apply_ufunc(
            lambda angle, corner1, corner2, corner3, corner4, R0, z0: (self.Rmax - R0)
            / np.cos(angle)
            if angle > corner1 or angle <= corner2
            else (self.zmax - z0) / np.sin(angle)
            if corner2 < angle <= corner3
            else (self.Rmin - R0) / np.cos(angle)
            if corner3 < angle <= corner4
            else (self.zmin - z0) / np.sin(angle),
            theta,
            corner_angles[0],
            corner_angles[1],
            corner_angles[2],
            corner_angles[3],
            R0,
            z0,
            vectorize=True,
        )
        minor_rads = DataArray(
            np.linspace(0.0, minor_rad_max, ngrid),
            dims=("r",) + minor_rad_max.dims,
            coords=minor_rad_max.coords,
        )
        R_grid = R0 + minor_rads * np.cos(theta)
        z_grid = z0 + minor_rads * np.sin(theta)
        fluxes_samples = reference_rhos.indica.interp2d(
            R=R_grid,
            z=z_grid,
            zero_coords={"R": R0, "z": z0},
            method="cubic",
            assume_sorted=True,
        ).rename("rhop")
        fluxes_samples.loc[{"r": 0}] = 0.0

        indices = fluxes_samples.indica.invert_root(rhop, "r", 0.0, method="cubic")
        return (
            minor_rads.indica.interp2d(r=indices, method="cubic", assume_sorted=True),
            t,
        )

    def flux_coords(
        self,
        R: OnlyArray,
        z: OnlyArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, DataArray, LabeledArray]:
        """
        Normalised flux coordinate and angle at a given location in space.

        R - Major radius position (m).
        z - The vertical position (m).
        t - Times (s).
        kind - Type of flux coordinate: "toroidal", "poloidal"
        """

        if t is None:
            rhop = self.rhop
            R_ax = self.rmag
            z_ax = self.zmag
            t = self.rhop.coords["t"]
            z_x_point_low = self.zx_low
            z_x_point_up = self.zx_up
        else:
            check_time_present(t, self.t)
            rhop = self.rhop.interp(t=t, method="nearest")
            R_ax = self.rmag.interp(t=t, method="nearest")
            z_ax = self.zmag.interp(t=t, method="nearest")
            z_x_point_low = self.zx_low.interp(t=t, method="nearest")
            z_x_point_up = self.zx_up.interp(t=t, method="nearest")

        # TODO: rho and theta dimensions not in the same order...
        rhop = rhop.interp(R=R, z=z)
        theta = np.arctan2(
            z - z_ax,
            R - R_ax,
        )

        # Correct for any interpolation errors resulting in negative fluxes
        rhop = xr.where((rhop < 0.0) * (rhop > -1e-12), 0.0, rhop)

        if kind != "poloidal":
            rhop, t = self.convert_flux_coords(rhop, t, "poloidal", kind)

        # Set rho to be negative in the private flux region
        rhop = xr.where(
            (rhop < 1.0) * (z < z_x_point_low) * (z < z_x_point_up), -rhop, rhop
        )

        # Convert to desired normalised flux coordinate
        rho, _ = self.convert_flux_coords(rhop, t, from_kind="poloidal", to_kind=kind)

        return rho, theta, t

    def spatial_coords(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
        """
        (R, z) coordinates of a flux surface given its (rho, theta).

        rho - Normalized flux coordinate values.
        theta - Poloidal angle.
        t - Times (s).
        kind - Type of flux coordinate: "toroidal", "poloidal"
        """
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        minor_rad, t = self.minor_radius(rhop, theta, t, kind)
        R0 = self.rmag.interp(t=t, method="nearest")
        z0 = self.zmag.interp(t=t, method="nearest")
        R = R0 + minor_rad * np.cos(theta)
        z = z0 + minor_rad * np.sin(theta)
        return R, z, t

    def convert_flux_coords(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        from_kind: Optional[str] = "poloidal",
        to_kind: Optional[str] = "toroidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """
        Convert between normalized flux coordinates coordinate systems.

        rho - Normalized flux coordinate values.
        t - Times (s).
        from_kind - Input flux coordinate: "toroidal", "poloidal"
        to_kind - Output flux coordinate: "poloidal", "toroidal"
        """
        if from_kind == to_kind:
            return rho, t

        supported = ["poloidal", "toroidal"]
        if from_kind not in supported or to_kind not in supported:
            raise ValueError("kind must be either poloidal or toroidal")

        if t is not None:
            check_time_present(t, self.t)
            conversion = self.rhot.interp(t=t, method="nearest")
        else:
            conversion = self.rhot
            t = self.rhot.t

        if to_kind == "toroidal":
            _rho = conversion.indica.interp2d(
                rhop=np.abs(rho), method="cubic", assume_sorted=True
            )
        elif to_kind == "poloidal":
            _rho = conversion.indica.invert_interp(np.abs(rho), "rhop", method="cubic")

        return _rho, t

    def cross_sectional_area(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """
        Cross-sectional area of desired flux surface

        rho - Normalized flux coordinate values.
        t - Times (s).
        kind - Normalised flux coordinate type: "toroidal", "poloidal"
        """
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        if t is None:
            t = self.rhop.coords["t"]
        else:
            check_time_present(t, self.t)
        area = self.area.interp(rhop=rhop).interp(t=t)
        return area, t

    def enclosed_volume(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """
        Volume of desired flux surface

        rho - Normalized flux coordinate values.
        t - Times (s).
        kind - Normalised flux coordinate type: "toroidal", "poloidal"
        """
        rhop, _ = self.convert_flux_coords(rho, t, from_kind=kind, to_kind="poloidal")
        if t is None:
            t = self.rhop.coords["t"]
        else:
            check_time_present(t, self.t)
        volume = self.area.interp(rhop=rhop).interp(t=t)
        return volume, t

    def write_to_geqdsk(self):
        # TODO: Implement writing to geqdsk
        raise NotImplementedError("Method not yet implemented")
