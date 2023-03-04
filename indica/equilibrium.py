"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import cast
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import apply_ufunc
from xarray import DataArray
from xarray import where

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
        self.rhotor = np.sqrt(
            (self.ftor - self.ftor.sel(rho_poloidal=0.0))
            / (self.ftor.sel(rho_poloidal=1.0) - self.ftor.sel(rho_poloidal=0.0))
        )
        self.psi = equilibrium_data["psi"]
        self.rho = np.sqrt((self.psi - self.faxs) / (self.fbnd - self.faxs))
        if "vjac" in equilibrium_data and "ajac" in equilibrium_data:
            self.psin = equilibrium_data["psin"]
            dpsin = self.psin[1] - self.psin[0]
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
        self.rmag = equilibrium_data["rmag"]
        self.rbnd = equilibrium_data["rbnd"]
        self.zmag = equilibrium_data["zmag"]
        self.zbnd = equilibrium_data["zbnd"]
        self.zx = self.zbnd.min("arbitrary_index")
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
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray, LabeledArray]:
        """Magnetic field components at this location in space.

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
            Magnetic field components at the given location and times.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        _R = convert_to_dataarray(R, ("R", R))
        _z = convert_to_dataarray(z, ("z", z))
        if t is not None:
            psi = self.psi.interp(t=t, method="nearest", assume_sorted=True)
            f = self.f.interp(t=t, method="nearest", assume_sorted=True)
            rho_, theta_, _ = self.flux_coords(_R, _z, t)
        else:
            t = self.rho.coords["t"]
            psi = self.psi
            f = self.f
            rho_, theta_, _ = self.flux_coords(_R, _z)

        dpsi_dR = psi.differentiate("R").indica.interp2d(
            R=_R, z=_z, method="cubic", assume_sorted=True,
        )
        dpsi_dz = psi.differentiate("z").indica.interp2d(
            R=R, z=z, method="cubic", assume_sorted=True,
        )
        b_R = -(np.float64(1.0) / _R) * dpsi_dz  # type: ignore
        b_R.name = "Radial magnetic field"
        b_z = (np.float64(1.0) / _R) * dpsi_dR  # type: ignore
        b_z.name = "Vertical Magnetic Field (T)"
        rho_ = where(
            rho_ > np.float64(0.0), rho_, np.float64(-1.0) * rho_  # type: ignore
        )
        f = f.indica.interp2d(rho_poloidal=rho_, method="cubic", assume_sorted=True,)
        f.name = self.f.name
        b_T = f / _R
        b_T.name = "Toroidal Magnetic Field (T)"

        return b_R, b_z, b_T, t

    def Btot(
        self, R: LabeledArray, z: LabeledArray, t: Optional[LabeledArray] = None
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
            Total magnetic field strength at the given location and times.
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
            Radial magnetic field strength at the given location and times.
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
            Vertical magnetic field strength at the given location and times.
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
            Toroidal magnetic field strength at the given location and times.
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
            Poloidal magnetic field strength at the given location and times.
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

        _R = convert_to_dataarray(R, ("R", R))
        _z = convert_to_dataarray(z, ("z", z))
        if t is not None:
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

        rho_interp = rho.indica.interp2d(
            R=_R + self.R_offset,
            z=_z + self.z_offset,
            zero_coords={"R": R_ax, "z": z_ax},
            method="cubic",
            assume_sorted=True,
        )
        # Correct for any interpolation errors resulting in negative fluxes
        rho_interp = where(
            np.logical_and(rho_interp < 0.0, rho_interp > -1e-12), 0.0, rho_interp
        )
        theta = np.arctan2(
            _z + cast(np.ndarray, self.z_offset) - z_ax,
            _R + cast(np.ndarray, self.R_offset) - R_ax,
        )
        if len(np.shape(theta)) > 1:
            theta = theta.transpose()

        if kind != "poloidal":
            rho_interp, t = self.convert_flux_coords(rho_interp, t, "poloidal", kind)

        # Set rho to be negative in the private flux region
        rho_interp = where(
            np.logical_and(rho_interp < 1.0, z < z_x_point), -rho_interp, rho_interp
        )

        return rho_interp, theta, t

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


def convert_to_dataarray(value, coords) -> DataArray:
    if type(value) != DataArray:
        return DataArray(value, [coords])
    else:
        return value
