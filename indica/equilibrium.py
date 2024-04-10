"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import cast
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import prov.model as prov
import xarray as xr
from xarray import apply_ufunc
from xarray import DataArray
from xarray import where
from xarray import zeros_like

from indica.converters.time import get_tlabels_dt
from indica.utilities import check_time_present
from . import session
from .numpy_typing import LabeledArray

_FLUX_TYPES = ["poloidal", "toroidal"]


class Equilibrium:
    """Class to hold and interpolate equilibrium data.

    At instantiation it will require calibration to select an offset
    along the major radius. Electron temperature data is provided for
    this purpose. Once calibrated, the electron temperature at
    normalised flux surface rho = 1 should be about 100eV.

    Parameters
    ----------
    equilibrium_data : Dict[str, DataArray]
        A collection of equilibrium data rea in using
        :py:meth:`~indica.readers.DataReader.get_equilibrium`. TODO: List full set
        of required quantities.
    R_shift : float
        How much to shift the equilibrium inwards (or the remapped diagnostic outwards)
        on the major radius.
        TODO: this and z_shift should be time-dependent...
    z_shift : float
        How much to shift the equilibrium downwards (or the remapped diagnostic upwards)
        in the vertical coordinate.
    sess : session.Session
        An object representing the session being run. Contains information
        such as provenance data.
    offset_picker: OffsetPicker
        A callback which determines by how much to offset the equilibrium data
        along the major radius. Allows the user to select this interactively.

    """

    def __init__(
        self,
        equilibrium_data: Dict[str, DataArray],
        R_shift: float = 0.0,
        z_shift: float = 0.0,
        sess: session.Session = session.global_session,
    ):

        self._session = sess
        self.f = equilibrium_data["f"]
        self.faxs = equilibrium_data["faxs"]
        self.fbnd = equilibrium_data["fbnd"]
        self.ftor = equilibrium_data["ftor"]
        self.rmag = equilibrium_data["rmag"]
        self.rbnd = equilibrium_data["rbnd"]
        self.zmag = equilibrium_data["zmag"]
        self.zbnd = equilibrium_data["zbnd"]
        self.zx = self.zbnd.min("arbitrary_index")
        self.rhotor = np.sqrt(
            (self.ftor - self.ftor.sel(rho_poloidal=0.0))
            / (self.ftor.sel(rho_poloidal=1.0) - self.ftor.sel(rho_poloidal=0.0))
        )
        self.psi = equilibrium_data["psi"]

        # Including workaround in case faxs or fbnd had messy data
        rho: DataArray = np.sqrt((self.psi - self.faxs) / (self.fbnd - self.faxs))
        if np.any(np.isnan(rho.interp(R=self.rmag, z=self.zmag))):
            self.faxs = self.psi.interp(R=self.rmag, z=self.zmag).drop(["R", "z"])
            self.fbnd = self.psi.interp(R=self.rbnd, z=self.zbnd).mean(
                "arbitrary_index"
            )
            rho = np.sqrt((self.psi - self.faxs) / (self.fbnd - self.faxs))
        if np.any(np.isnan(rho)):
            rho = xr.where(rho > 0, rho, 0.0)

        self.rho = rho
        self.t = self.rho.t
        if "vjac" in equilibrium_data and "ajac" in equilibrium_data:
            psin = (equilibrium_data["vjac"].rho_poloidal) ** 2
            dpsin = psin[1] - psin[0]
            self.volume = (equilibrium_data["vjac"] * dpsin).cumsum("rho_poloidal")
            self.area = (equilibrium_data["ajac"] * dpsin).cumsum("rho_poloidal")
        elif "volume" in equilibrium_data and "area" in equilibrium_data:
            self.volume = equilibrium_data["volume"]
            self.area = equilibrium_data["area"]
        else:
            raise ValueError("No volume or area information")
        if "rmji" and "rmjo" in equilibrium_data:
            self.rmji = equilibrium_data["rmji"]
            self.rmjo = equilibrium_data["rmjo"]
        self.R_offset = R_shift
        self.z_offset = z_shift

        self.Rmin = min(self.rho.coords["R"])
        self.Rmax = max(self.rho.coords["R"])
        self.zmin = min(self.rho.coords["z"])
        self.zmax = max(self.rho.coords["z"])
        self.corner_angles = [
            np.arctan2(self.zmin - self.zmag, self.Rmax - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmax - self.zmag, self.Rmax - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmax - self.zmag, self.Rmin - self.rmag) % (2 * np.pi),
            np.arctan2(self.zmin - self.zmag, self.Rmin - self.rmag) % (2 * np.pi),
        ]

        self.prov_id = session.hash_vals(
            **equilibrium_data, R_offset=self.R_offset, z_offset=self.z_offset
        )
        self.provenance = sess.prov.entity(
            self.prov_id,
            {
                prov.PROV_TYPE: "Equilibrium",
                "R_offset": self.R_offset,
                "z_offset": self.z_offset,
            },
        )
        sess.prov.generation(
            self.provenance, sess.session, time=datetime.datetime.now()
        )
        sess.prov.attribution(self.provenance, sess.agent)
        for val in equilibrium_data.values():
            if "provenance" in val.attrs:
                self.provenance.wasDerivedFrom(val.attrs["provenance"])

    def Bfield(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        full_Rz: bool = False,
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray, LabeledArray]:
        """Magnetic field components at this location in space.

        TODO: B_T approximated as following 1/R for any z to fill whole (R,z) space

        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertical position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.

        Returns
        -------
        Br, Bz, Bt
            Magnetic field components at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        _R, _z = prepare_coords(
            R + np.full_like(R, self.R_offset), z + np.full_like(z, self.z_offset)
        )
        if t is not None:
            check_time_present(t, self.t)
            psi = self.psi.interp(t=t, method="nearest", assume_sorted=True)
            f = self.f.interp(t=t, method="nearest", assume_sorted=True)
            rho_, theta_, _ = self.flux_coords(_R, _z, t)
        else:
            t = self.rho.coords["t"]
            psi = self.psi
            f = self.f
            rho_, theta_, _ = self.flux_coords(_R, _z)

        dpsi_dR = psi.differentiate("R").interp(R=_R, z=_z)
        dpsi_dz = psi.differentiate("z").interp(
            R=_R,
            z=_z,
        )
        b_R = -(np.float64(1.0) / _R) * dpsi_dz  # type: ignore
        b_R.name = "Radial magnetic field"
        b_R = b_R.T
        b_z = (np.float64(1.0) / _R) * dpsi_dR  # type: ignore
        b_z.name = "Vertical Magnetic Field (T)"
        b_z = b_z.T
        rho_ = where(
            rho_ > np.float64(0.0), rho_, np.float64(-1.0) * rho_  # type: ignore
        )

        f = f.interp(rho_poloidal=rho_)
        f.name = self.f.name
        b_T = f / _R
        b_T.name = "Toroidal Magnetic Field (T)"

        if full_Rz:
            _b_T = b_T.interp(R=self.rmag, z=self.zmag) * self.rmag / self.rho.R
            _b_T = _b_T.drop(["z", "rho_poloidal"]).expand_dims(dim={"z": self.rho.z})
            b_T = _b_T.interp(R=_R, z=_z)

        return b_R, b_z, b_T, t

    def Btot(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        full_Rz: bool = False,
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Total magnetic field strength at this location in space.

        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertial position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.

        Returns
        -------
        Btot
            Total magnetic field strength at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)
        b_Tot = np.sqrt(
            b_R ** np.float64(2.0) + b_z ** np.float64(2.0) + b_T ** np.float64(2.0)
        )
        b_Tot.name = "Total Magnetic Field (T)"

        return b_Tot, t

    def Br(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Radial magnetic field strength at this location in space.
        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertial position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength.
        Returns
        -------
        Br
            Radial magnetic field strength at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)

        return b_R, t

    def Bz(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Vertical magnetic field strength at this location in space.
        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertical position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength.
        Returns
        -------
        Bz
            Vertical magnetic field strength at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)

        return b_z, t

    def Bt(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Toroidal magnetic field strength at this location in space.

        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertical position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.

        Returns
        -------
        Bt
            Toroidal magnetic field strength at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        b_R, b_z, b_T, t = self.Bfield(R, z, t)

        return b_T, t

    def Bp(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Poloidal magnetic field strength at this location in space.

        Parameters
        ----------
        R
            Major radius position at which to get magnetic field strength.
        z
            The vertical position at which to get the magnetic field strength.
        t
            Times at which to get the magnetic field strength. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.

        Returns
        -------
        Bp
            Poloidal magnetic field strength at the given location and time.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
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
        """Major radius position of the given flux surface on the Low Flux
         Side of the magnetic axis.

        Parameters
        ----------
        rho
            Flux values for the locations.
        t
            Times at which to get the major radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.


        Returns
        -------
        R_lfs
            Major radius on the LFS for the given flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        if t is None:
            rmjo = self.rmjo
            t = self.rmjo.coords["t"]
            rho, _ = self.convert_flux_coords(rho, t, kind, "poloidal")
            R = rmjo.indica.interp2d(rho_poloidal=rho, method="cubic") - self.R_offset
        else:
            check_time_present(t, self.t)
            rmjo = self.rmjo.interp(t=t, method="nearest")
            rho, _ = self.convert_flux_coords(rho, t, kind, "poloidal")
            R = rmjo.interp(rho_poloidal=rho, method="cubic") - self.R_offset

        return R, t

    def R_hfs(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Major radius position of the given flux surface on the High Flux
         Side of the magnetic axis.

        Parameters
        ----------
        rho
            Flux values for the locations.
        t
            Times at which to get the major radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.


        Returns
        -------
        R_hfs
            Major radius on the HFS for the given flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.

        """
        if t is None:
            rmji = self.rmji
            t = self.rmji.coords["t"]
        else:
            check_time_present(t, self.t)
            rmji = self.rmji.interp(t=t, method="nearest")
        rho, _ = self.convert_flux_coords(rho, t, kind, "poloidal")
        try:
            R = rmji.interp(rho_poloidal=rho, method="cubic") - self.R_offset
        except ValueError:
            R = rmji.indica.interp2d(rho_poloidal=rho, method="cubic") - self.R_offset

        return R, t

    def minor_radius(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """Minor radius at the given locations in the tokamak.

        Parameters
        ----------
        rho
            Flux surfaces on which the locations fall.
        theta
            Poloidal positions on which the locations fall.
        t
            Times at which to get the minor radius. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        minor_radius
            Minor radius of the locations.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        ngrid = 100
        rho, _ = self.convert_flux_coords(rho, t, kind, "poloidal")
        theta = theta % (2 * np.pi)
        if t is not None:
            check_time_present(t, self.t)
            corner_angles = [
                angle.interp(t=t, method="nearest") for angle in self.corner_angles
            ]
            R0 = self.rmag.interp(t=t, method="nearest")
            z0 = self.zmag.interp(t=t, method="nearest")
            reference_rhos = self.rho.interp(t=t, method="nearest")
            reference_rhos.name = self.rho.name
        else:
            corner_angles = self.corner_angles
            R0 = self.rmag
            z0 = self.zmag
            reference_rhos = self.rho
            t = self.rho.coords["t"]
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
        ).rename("rho_" + kind)
        fluxes_samples.loc[{"r": 0}] = 0.0

        indices = fluxes_samples.indica.invert_root(rho, "r", 0.0, method="cubic")
        return (
            minor_rads.indica.interp2d(r=indices, method="cubic", assume_sorted=True),
            t,
        )

    def flux_coords(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, DataArray, LabeledArray]:
        """Convert to the flux surface coordinate system.

        Parameters
        ----------
        R
            Major radius positions.
        z
            Vertical positions.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        rho
            Flux surface for each position.
        theta
            Poloidal angle along flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        _R, _z = prepare_coords(
            R + np.full_like(R, self.R_offset), z + np.full_like(z, self.z_offset)
        )
        if t is not None:
            check_time_present(t, self.t)
            rho = self.rho.interp(t=t, method="nearest")
            R_ax = self.rmag.interp(t=t, method="nearest")
            z_ax = self.zmag.interp(t=t, method="nearest")
            z_x_point = self.zx.interp(t=t, method="nearest")
        else:
            rho = self.rho
            R_ax = self.rmag
            z_ax = self.zmag
            t = self.rho.coords["t"]
            z_x_point = self.zx

        # TODO: rho and theta dimensions not in the same order...
        rho = rho.interp(R=_R, z=_z)
        theta = np.arctan2(
            _z - z_ax,
            _R - R_ax,
        )

        # Correct for any interpolation errors resulting in negative fluxes
        rho = xr.where((rho < 0.0) * (rho > -1e-12), 0.0, rho)

        if kind != "poloidal":
            rho, t = self.convert_flux_coords(rho, t, "poloidal", kind)

        # Set rho to be negative in the private flux region
        rho = xr.where((rho < 1.0) * (z < z_x_point), -rho, rho)

        return rho, theta, t

    def spatial_coords(
        self,
        rho: LabeledArray,
        theta: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
        """Convert to the spatial coordinate system.

        Parameters
        ----------
        rho
            Flux surface coordinate.
        theta
            Angular position.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        R
            Major radius positions.
        z
            Vertical positions.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        minor_rad, t = self.minor_radius(rho, theta, t, kind)
        R0 = self.rmag.interp(t=t, method="nearest")
        z0 = self.zmag.interp(t=t, method="nearest")
        R = R0 - self.R_offset + minor_rad * np.cos(theta)
        z = z0 - self.z_offset + minor_rad * np.sin(theta)
        return R, z, t

    def convert_flux_coords(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        from_kind: Optional[str] = "poloidal",
        to_kind: Optional[str] = "toroidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Convert between different coordinate systems.

        Parameters
        ----------
        rho
            Input flux surface coordinate.
        t
            Times for conversions. Defaults to the time range specified when
            equilibrium object was instantiated and frequency the equilibrium
            data was calculated at.
        from_kind
            The type of flux surfaces used for the input coordinates. May be
            "toroidal", "poloidal", plus optional extras depending on
            implementation.
        to_kind
            The type of flux surfaces on which to calculate the output
            coordinates. May be "toroidal", "poloidal", plus optional extras
            depending on implementation.

        Returns
        -------
        rho
            New flux surface for each position.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        if from_kind not in _FLUX_TYPES:
            raise ValueError(f"Unrecognised input flux kind, '{from_kind}'.")
        if to_kind not in _FLUX_TYPES:
            raise ValueError(f"Unrecognised output flux kind, '{to_kind}'.")
        if from_kind == to_kind:
            if t is None:
                t = self.rhotor.coords["t"]
            return rho, t
        if t is not None:
            check_time_present(t, self.t)
            conversion = self.rhotor.interp(t=t, method="nearest")
        else:
            conversion = self.rhotor
            t = self.rhotor.coords["t"]
        if to_kind == "toroidal":
            flux = conversion.indica.interp2d(
                rho_poloidal=np.abs(rho), method="cubic", assume_sorted=True
            )
        elif to_kind == "poloidal":
            flux = conversion.indica.invert_interp(
                np.abs(rho), "rho_poloidal", method="cubic"
            )
        return flux, t

    def cross_sectional_area(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """Calculates the cross-sectional area inside the flux surface rho and at
        given time t.

        Parameters
        ----------
        rho
            Values of rho at which to calculate the cross-sectional area.
        t
            Values of time at which to calculate the cross-sectional area.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        area
            Cross-sectional areas calculated at rho and t.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """

        if t is None:
            t = self.rho.coords["t"]

        check_time_present(t, self.t)
        if kind == "toroidal":
            _rho, _ = self.convert_flux_coords(
                rho, t, from_kind="toroidal", to_kind="poloidal"
            )
        elif kind == "poloidal":
            _rho = rho
        else:
            raise ValueError("kind must be either poloidal or toroidal")

        result = self.area.interp(rho_poloidal=_rho).interp(t=t)

        return (
            result,
            cast(LabeledArray, t),
        )

    def enclosed_volume(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, LabeledArray]:
        """Returns the volume enclosed by the specified flux surface.

        Parameters
        ----------
        rho
            Flux surfaces to get the enclosed volumes for.
        t
            Times at which to get the enclosed volume. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        vol
            Volumes of space enclosed by the flux surfaces.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        if t is None:
            t = self.rho.coords["t"]

        check_time_present(t, self.t)
        _rho: LabeledArray
        if kind == "toroidal":
            _rho, _ = self.convert_flux_coords(
                rho, t, from_kind="toroidal", to_kind="poloidal"
            )
        elif kind == "poloidal":
            _rho = rho
        else:
            raise ValueError("kind must be either poloidal or toroidal")

        result = self.volume.interp(rho_poloidal=_rho).interp(t=t)

        return (
            result,
            cast(LabeledArray, t),
        )

    def write_to_geqdsk(self):
        # TODO: Implement writing to geqdsk
        raise NotImplementedError("Method not yet implemented")


def prepare_coords(R: LabeledArray, z: LabeledArray) -> Tuple[DataArray, DataArray]:

    if type(R) != DataArray or type(z) != DataArray:
        coords: list = []
        for idim, npts in enumerate(np.shape(R)):
            coords.append((f"dim{idim}", np.arange(npts)))
        _R = DataArray(np.array(R), coords=coords)
        _z = DataArray(np.array(z), coords=coords)
    else:
        _R = R
        _z = z

    return _R, _z


DEFAULT_PARAMS = {
    "poloidal_a": 0.5,
    "poloidal_b": 1.0,
    "poloidal_n": 1,
    "poloidal_alpha": 0.01,
    "toroidal_a": 0.7,
    "toroidal_b": 1.4,
    "toroidal_n": 1,
    "toroidal_alpha": -0.00005,
    "Btot_a": 1.0,
    "Btot_b": 1.0,
    "Btot_alpha": 0.001,
}


def smooth_funcs(domain=(0.0, 1.0), max_val=None):
    if not max_val:
        max_val = 0.01
    min_val = -max_val
    nterms = 6
    coeffs = np.linspace(min_val, max_val, nterms)

    def f(x):
        x = (x - domain[0]) / (domain[1] - domain[0])
        term = 1
        y = zeros_like(x) if isinstance(x, DataArray) else np.zeros_like(x)
        for coeff in coeffs:
            y += coeff * term
            term *= x
        return y

    return f


def fake_equilibrium(
    tstart: float = 0,
    tend: float = 0.1,
    dt: float = 0.01,
    machine_dims=None,
    machine: str = "st40",
):
    equilibrium_data = fake_equilibrium_data(
        tstart=tstart, tend=tend, dt=dt, machine_dims=machine_dims, machine=machine
    )
    return Equilibrium(equilibrium_data)


def fake_equilibrium_data(
    tstart: float = 0,
    tend: float = 0.1,
    dt: float = 0.01,
    machine_dims=None,
    machine: str = "st40",
):
    def monotonic_series(
        start: float,
        stop: float,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype: bool = None,
    ):
        return np.linspace(
            start, stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype
        )

    if machine_dims is None:
        machine_dims = ((0.15, 0.85), (-0.75, 0.75))

    get_tlabels_dt(tstart, tend, dt)
    time = np.arange(tstart, tend + dt, dt)

    # ntime = time.size
    Btot_factor = None

    result = {}
    nspace = 100

    tfuncs = smooth_funcs((tstart, tend), 0.01)
    r_centre = (machine_dims[0][0] + machine_dims[0][1]) / 2
    z_centre = (machine_dims[1][0] + machine_dims[1][1]) / 2

    raw_result: dict = {}
    attrs: dict = {}

    result["rmag"] = DataArray(
        r_centre + tfuncs(time), coords=[("t", time)], name="rmag", attrs=attrs
    )
    result["rmag"].attrs["datatype"] = ("major_rad", "mag_axis")

    result["zmag"] = DataArray(
        z_centre + tfuncs(time), coords=[("t", time)], name="zmag", attrs=attrs
    )
    result["zmag"].attrs["datatype"] = ("z", "mag_axis")

    fmin = 0.1
    result["faxs"] = DataArray(
        fmin + np.abs(tfuncs(time)),
        {"t": time, "R": result["rmag"], "z": result["zmag"]},
        ["t"],
        name="faxs",
        attrs=attrs,
    )
    result["faxs"].attrs["datatype"] = ("magnetic_flux", "mag_axis")

    a_coeff = DataArray(
        np.vectorize(lambda x: 0.8 * x)(
            np.minimum(
                np.abs(machine_dims[0][0] - result["rmag"]),
                np.abs(machine_dims[0][1] - result["rmag"]),
            ),
        ),
        coords=[("t", time)],
    )

    if Btot_factor is None:
        b_coeff = DataArray(
            np.vectorize(lambda x: 0.8 * x)(
                np.minimum(
                    np.abs(machine_dims[1][0] - result["zmag"].data),
                    np.abs(machine_dims[1][1] - result["zmag"].data),
                ),
            ),
            coords=[("t", time)],
        )
        n_exp = 0.5
        fmax = 5.0
        result["fbnd"] = DataArray(
            fmax - np.abs(tfuncs(time)),
            coords=[("t", time)],
            name="fbnd",
            attrs=attrs,
        )
    else:
        b_coeff = a_coeff
        n_exp = 1
        fdiff_max = Btot_factor * a_coeff
        result["fbnd"] = DataArray(
            np.vectorize(lambda axs, diff: axs + 0.03 * diff)(
                result["faxs"], fdiff_max.values
            ),
            coords=[("t", time)],
            name="fbnd",
            attrs=attrs,
        )

    result["fbnd"].attrs["datatype"] = ("magnetic_flux", "separtrix")

    thetas = DataArray(
        np.linspace(0.0, 2 * np.pi, nspace, endpoint=False), dims=["arbitrary_index"]
    )

    r = np.linspace(machine_dims[0][0], machine_dims[0][1], nspace)
    z = np.linspace(machine_dims[1][0], machine_dims[1][1], nspace)
    rgrid = DataArray(r, coords=[("R", r)])
    zgrid = DataArray(z, coords=[("z", z)])
    psin = (
        (-result["zmag"] + zgrid) ** 2 / b_coeff**2
        + (-result["rmag"] + rgrid) ** 2 / a_coeff**2
    ) ** (0.5 / n_exp)

    psi = psin * (result["fbnd"] - result["faxs"]) + result["faxs"]
    psi.name = "psi"
    psi.attrs = attrs
    psi.attrs["datatype"] = ("magnetic_flux", "plasma")
    result["psi"] = psi

    psin_coords = np.linspace(0.0, 1.0, nspace)
    rho1d = np.sqrt(psin_coords)
    psin_data = DataArray(psin_coords, coords=[("rho_poloidal", rho1d)])
    result["psin"] = psin_data

    ftor_min = 0.1
    ftor_max = 5.0
    result["ftor"] = DataArray(
        np.outer(1 + tfuncs(time), monotonic_series(ftor_min, ftor_max, nspace)),
        coords=[("t", time), ("rho_poloidal", rho1d)],
        name="ftor",
        attrs=attrs,
    )
    result["ftor"].attrs["datatype"] = ("toroidal_flux", "plasma")

    # It should be noted during this calculation that the full extent of theta
    # isn't represented in the resultant rbnd and zbnd values.
    # This is because for rbnd: 1/sqrt(tan(x)^2) and for zbnd: 1/sqrt(tan(x)^-2)
    # are periodic functions which span a fixed 0 to +inf range on the y-axis
    # between 0 and 2pi, with f(x) = f(x+pi) and f(x) = f(pi-x)
    result["rbnd"] = (
        result["rmag"]
        + a_coeff * b_coeff / np.sqrt(a_coeff**2 * np.tan(thetas) ** 2 + b_coeff**2)
    ).assign_attrs(**attrs)
    result["rbnd"].name = "rbnd"
    result["rbnd"].attrs["datatype"] = ("major_rad", "separatrix")

    result["zbnd"] = (
        result["zmag"]
        + a_coeff
        * b_coeff
        / np.sqrt(a_coeff**2 + b_coeff**2 * np.tan(thetas) ** -2)
    ).assign_attrs(**attrs)
    result["zbnd"].name = "zbnd"
    result["zbnd"].attrs["datatype"] = ("z", "separatrix")

    # Indices of thetas for,
    # 90 <= thetas < 180
    # 180 <= thetas < 270
    # 270 <= thetas < 360
    arcs = {
        "90to180": np.flatnonzero((0.5 * np.pi <= thetas) & (thetas < 1.0 * np.pi)),
        "180to270": np.flatnonzero((1.0 * np.pi <= thetas) & (thetas < 1.5 * np.pi)),
        "270to360": np.flatnonzero((1.5 * np.pi <= thetas) & (thetas < 2.0 * np.pi)),
    }

    # Transforms rbnd appropriately to represent the values when
    # 90 <= theta < 180 and 180 <= theta < 270
    result["rbnd"][:, arcs["90to180"]] = (
        -result["rbnd"][:, arcs["90to180"]] + 2 * result["rmag"]
    )
    result["rbnd"][:, arcs["180to270"]] = (
        -result["rbnd"][:, arcs["180to270"]] + 2 * result["rmag"]
    )

    # Transforms zbnd appropriately to represent the values when
    # 180 <= theta < 270 and 270 <= theta < 360
    result["zbnd"][:, arcs["180to270"]] = (
        -result["zbnd"][:, arcs["180to270"]] + 2 * result["zmag"]
    )
    result["zbnd"][:, arcs["270to360"]] = (
        -result["zbnd"][:, arcs["270to360"]] + 2 * result["zmag"]
    )

    if Btot_factor is None:
        f_min = 0.1
        f_max = 5.0
        time_vals = tfuncs(time)
        space_vals = monotonic_series(f_min, f_max, nspace)
        f_raw = np.outer(abs(1 + time_vals), space_vals)
    else:
        f_raw = np.outer(
            np.sqrt(
                Btot_factor**2
                - (raw_result["fbnd"] - raw_result["faxs"]) ** 2 / a_coeff**2
            ),
            np.ones_like(rho1d),
        )
        f_raw[:, 0] = Btot_factor

    result["f"] = DataArray(
        f_raw, coords=[("t", time), ("rho_poloidal", rho1d)], name="f", attrs=attrs
    )
    result["f"].attrs["datatype"] = ("f_value", "plasma")

    # TODO: RMJO and RMJI not calculated correctly...
    result["rmjo"] = (result["rmag"] + a_coeff * psin_data**n_exp).assign_attrs(
        **attrs
    )
    result["rmjo"].name = "rmjo"
    result["rmjo"].attrs["datatype"] = ("major_rad", "lfs")
    result["rmjo"].coords["z"] = result["zmag"]
    result["rmji"] = (result["rmag"] - a_coeff * psin_data**n_exp).assign_attrs(
        **attrs
    )
    result["rmji"].name = "rmji"
    result["rmji"].attrs["datatype"] = ("major_rad", "hfs")
    result["rmji"].coords["z"] = result["zmag"]

    result["vjac"] = (
        4
        * n_exp
        * np.pi**2
        * result["rmag"]
        * a_coeff
        * b_coeff
        * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["vjac"].name = "vjac"
    result["vjac"].attrs["datatype"] = ("volume_jacobian", "plasma")

    result["ajac"] = (
        2 * n_exp * np.pi * a_coeff * b_coeff * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["ajac"].name = "ajac"
    result["ajac"].attrs["datatype"] = ("area_jacobian", "plasma")

    return result
