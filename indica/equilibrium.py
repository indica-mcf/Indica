"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import Any
from typing import cast
from typing import Dict
from typing import Hashable
from typing import Optional
from typing import Tuple

import numpy as np
import prov.model as prov
from scipy.integrate import trapz
from xarray import apply_ufunc
from xarray import concat
from xarray import DataArray
from xarray import where

from . import session
from .abstract_equilibrium import AbstractEquilibrium
from .numpy_typing import LabeledArray
from .offset import interactive_offset_choice
from .offset import OffsetPicker
from .operators import SplineFit
from .utilities import coord_array

_FLUX_TYPES = ["poloidal", "toroidal"]


class Equilibrium(AbstractEquilibrium):
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
    T_e : Optional[DataArray]
        Electron temperature data (from HRTS on JET). If present, used to compute
        an offset of the equilibrium along the major radius.
    R_shift : float
        How much to shift the equilibrium profile inwards on the major radius.
        Ignored if `T_e` is also passed as an argument.
    z_shift : flaot
        How much to shift the equilibrium profile downwards in the vertical
        coordinate.
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
        T_e: Optional[DataArray] = None,
        R_shift: float = 0.0,
        z_shift: float = 0.0,
        sess: session.Session = session.global_session,
        offset_picker: OffsetPicker = interactive_offset_choice,
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
        self.rmji = equilibrium_data["rmji"]
        self.rmjo = equilibrium_data["rmjo"]
        self.psi = equilibrium_data["psi"]
        self.rho = np.sqrt((self.psi - self.faxs) / (self.fbnd - self.faxs))
        self.vjac = equilibrium_data["vjac"]
        self.rmag = equilibrium_data["rmag"]
        self.rbnd = equilibrium_data["rbnd"]
        self.zmag = equilibrium_data["zmag"]
        self.zbnd = equilibrium_data["zbnd"]
        self.zx = self.zbnd.min("arbitrary_index")
        if T_e is not None:
            offsets = coord_array(np.linspace(0.0, 0.04, 9), "offset")
            t = T_e.coords["t"].assign_attrs(datatype=("time", "plasma"))
            separatrix = DataArray(1.0, attrs={"datatype": ("norm_flux_pol", "plasma")})
            separatrix.coords["norm_flux_pol"] = 1.0
            Rmag = self.rmag.interp(t=T_e.coords["t"], method="nearest") - offsets
            zmag = self.zmag.interp(t=T_e.coords["t"], method="nearest") - z_shift
            rhos = concat(
                [
                    self.rho.interp(t=t, method="nearest").indica.interp2d(
                        R=T_e.coords["index"] - offset,
                        z=T_e.coords["index_z_offset"] - z_shift,
                        zero_coords={
                            "R": Rmag.sel(offset=offset),
                            "z": zmag,
                        },
                        method="cubic",
                        assume_sorted=True,
                    )
                    for offset in offsets
                ],
                offsets,
            )

            thetas = np.arctan2(
                T_e.coords["index_z_offset"] + z_shift - zmag,
                T_e.coords["index"] + offsets - Rmag,
            )
            T_e_with_rho = T_e.expand_dims(
                cast(Dict[Hashable, Any], {"offset": offsets})
            ).assign_coords(rho_poloidal=rhos, theta=thetas)
            fitter = SplineFit(lower_bound=0.0, sess=sess)

            T_e_sep = concat(
                [
                    fitter(separatrix, t, T_e_with_rho.sel(offset=offset))[0]
                    for offset in offsets
                ],
                offsets,
            )

            square_residuals = (T_e_sep - 100.0) ** 2
            best_fits = square_residuals.offset[square_residuals.argmin(dim="offset")]
            overall_best = best_fits.mean()
            fluxes = rhos.sel(offset=overall_best, method="nearest")
            offset = float(fluxes.offset)
            offset, accept = offset_picker(offset, T_e, fluxes, best_fits)
            while not accept:
                fluxes = self.rho.interp(
                    t=T_e.coords["t"], method="nearest"
                ).indica.interp2d(
                    R=T_e.coords["index"] - offset,
                    z=T_e.coords["index_z_offset"] - z_shift,
                    method="cubic",
                    assume_sorted=True,
                )
                offset, accept = offset_picker(offset, T_e, fluxes, best_fits)

            self.R_offset = offset
        else:
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
        if (T_e is not None) and ("provenance" in T_e.attrs):
            self.provenance.wasDerivedFrom(T_e.attrs["provenance"])

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
        if t is not None:
            interp1d_method = "linear"
            if (isinstance(t, DataArray) or isinstance(t, np.ndarray)) and (
                t.shape[0] > 1
            ):
                if t.shape[0] > 3:
                    interp1d_method = "cubic"

            psi = self.psi.interp(t=t, method=interp1d_method, assume_sorted=True)

            f = self.f.interp(t=t, method=interp1d_method, assume_sorted=True)

            rho_, theta_, _ = self.flux_coords(R, z, t)
        else:
            t = self.rho.coords["t"]
            psi = self.psi
            f = self.f
            rho_, theta_, _ = self.flux_coords(R, z)

        dpsi_dR = psi.differentiate("R").indica.interp2d(
            R=R,
            z=z,
            method="cubic",
            assume_sorted=True,
        )
        dpsi_dz = psi.differentiate("z").indica.interp2d(
            R=R,
            z=z,
            method="cubic",
            assume_sorted=True,
        )

        # Components of poloidal field
        b_R = -(np.float64(1.0) / R) * dpsi_dz  # type: ignore
        b_z = (np.float64(1.0) / R) * dpsi_dR  # type: ignore
        b_Pol = np.sqrt(b_R ** np.float64(2.0) + b_z ** np.float64(2.0))

        # Need this as the current flux_coords function
        # returns some negative values for rho
        rho_ = where(
            rho_ > np.float64(0.0), rho_, np.float64(-1.0) * rho_  # type: ignore
        )

        f = f.indica.interp2d(
            rho_poloidal=rho_,
            method="cubic",
            assume_sorted=True,
        )
        f.name = self.f.name

        # Toroidal field
        b_T = f / R

        b_Tot = np.sqrt(b_Pol ** np.float64(2.0) + b_T ** np.float64(2.0))

        b_Tot.name = "Total Magnetic Field (T)"

        return b_Tot, t

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

    def cross_sectional_area(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        ntheta: int = 12,
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
        ntheta
            Number subdivisions of 2 * pi to integrate over for the cross-
            sectional area.
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

        if np.isscalar(rho):
            rho = np.array([rho])

        if np.isscalar(t):
            t = np.array([t])

        theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
        # Reassignment to a different type is not recognised by mypy.
        theta = DataArray(  # type: ignore
            data=theta,
            coords={"theta": theta},
            dims=[
                "theta",
            ],
        )

        minor_radii = np.empty(ntheta, dtype=DataArray)

        for i, itheta in enumerate(theta):
            minor_radii[i], _ = self.minor_radius(rho, itheta, t, kind)

        # This sets minor_radii to zero in the rho positions (in the minor_radii array)
        # where the rho value is below the precision possible through
        # the trapz() function.
        zero_check = np.where(rho < 1e-18)[0]
        if zero_check.size > 0:
            for i in zero_check:
                for k, itheta in enumerate(theta):
                    minor_radii[k][:, i] = np.zeros(minor_radii[k][:, i].shape)

        minor_radii = minor_radii**2
        minor_radii = minor_radii * 0.5

        area = trapz(minor_radii, theta, axis=0)

        result = DataArray(
            data=area,
            coords=[("t", np.array(t)), ("rho_poloidal", np.array(rho))],
            dims=["t", "rho_poloidal"],
        )

        return (
            result,
            cast(LabeledArray, t),
        )

    def enclosed_volume(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[DataArray, DataArray, LabeledArray]:
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

        interp1d_method = "linear"
        if (isinstance(t, DataArray) or isinstance(t, np.ndarray)) and (t.shape[0] > 1):
            if t.shape[0] > 3:
                interp1d_method = "cubic"

        major_radius_axis = self.rmag.interp(
            t=t,
            method=interp1d_method,
            assume_sorted=True,
        )

        # Cross-sectional area calculated by integrating:
        # 0.5 * minor_radius(theta) ** 2 with respect to theta from 0 to 2 * np.pi
        area_arr, _ = self.cross_sectional_area(rho, t, kind=kind)

        # Vol = area * toroidal circumference measure at the magnetic axis
        vol_enclosed = area_arr * 2 * np.pi * major_radius_axis
        vol_enclosed.name = "Enclosed volumes (m^3)"

        return vol_enclosed, area_arr, t

    def invert_enclosed_volume(
        self,
        vol: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
        """Returns the value of the flux surface enclosing the specified volume.

        Parameters
        ----------
        vol
            Volumes of space enclosed by the flux surfaces.
        t
            Times at which to get the enclosed volume. Defaults to the
            time range specified when equilibrium object was instantiated and
            frequency the equilibrium data was calculated at.
        kind
            The type of flux surface to use. May be "toroidal", "poloidal",
            plus optional extras depending on implementation.

        Returns
        -------
        rho
            Flux surfaces for the enclosed volumes.
        t
            If ``t`` was not specified as an argument, return the time the
            results are given for. Otherwise return the argument.
        """
        raise NotImplementedError(
            "{} does not implement an 'invert_enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

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
            method="linear",
            assume_sorted=True,
        ).rename("rho_" + kind)
        fluxes_samples.loc[{"r": 0}] = 0.0

        indices = fluxes_samples.indica.invert_root(rho, "r", 0.0, method="linear")
        return (
            minor_rads.indica.interp2d(r=indices, method="linear", assume_sorted=True),
            t,
        )

    def flux_coords(
        self,
        R: LabeledArray,
        z: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray, LabeledArray]:
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
        if t is not None:
            rho = self.rho.interp(t=t, method="nearest")
            R_ax = self.rmag.interp(t=t, method="nearest")
            z_ax = self.zmag.interp(t=t, method="nearest")
            z_x_point = self.zx.interp(t=t, method="nearest")
            t = t
        else:
            rho = self.rho
            R_ax = self.rmag
            z_ax = self.zmag
            t = self.rho.coords["t"]
            z_x_point = self.zx
        rho_interp = rho.indica.interp2d(
            R=R + self.R_offset,
            z=z + self.z_offset,
            zero_coords={"R": R_ax, "z": z_ax},
            method="cubic",
            assume_sorted=True,
        )
        # Correct for any interpolation errors resulting in negative fluxes
        rho_interp = where(
            np.logical_and(rho_interp < 0.0, rho_interp > -1e-12), 0.0, rho_interp
        )
        theta = np.arctan2(
            z + cast(np.ndarray, self.z_offset) - z_ax,
            R + cast(np.ndarray, self.R_offset) - R_ax,
        )
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
