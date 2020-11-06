"""Contains an abstract base class for reading equilibrium data for a pulse.
"""

import datetime
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import prov.model as prov
from xarray import apply_ufunc
from xarray import concat
from xarray import DataArray
from xarray import where

from .numpy_typing import LabeledArray
from .offset import interactive_offset_choice
from .offset import OffsetPicker
from .session import global_session
from .session import hash_vals
from .session import Session


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
    T_e : Optional[DataArray]
        Electron temperature data (from HRTS on JET).
    sess : Session
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
        sess: Session = global_session,
        offset_picker: OffsetPicker = interactive_offset_choice,
    ):
        # def find_separatrix_index(data):
        #     coords = data.coords[data.dims[0]]
        #     bracket = (coords[0], coords[-1])
        #     interp = interp1d(
        #         coords, data.data, "cubic", copy=False, assume_sorted=True
        #     )
        #     return root_scalar(lambda x: interp(x) - 1.0, bracket).root

        self._session = sess
        self.f = equilibrium_data["f"]
        self.faxs = equilibrium_data["faxs"]
        self.fbnd = equilibrium_data["fbnd"]
        ftor = equilibrium_data["ftor"]
        self.ftor = np.sqrt(
            (ftor - ftor.sel(rho_poloidal=0.0))
            / (ftor.sel(rho_poloidal=1.0) - ftor.sel(rho_poloidal=0.0))
        )
        self.rmji = equilibrium_data["rmji"]
        self.rmjo = equilibrium_data["rmjo"]
        self.psi = equilibrium_data["psi"]
        self.rho = np.sqrt((self.psi - self.faxs) / (self.fbnd - self.faxs))
        print("------------------------------------------")
        print(self.rho)
        self.vjac = equilibrium_data["vjac"]
        self.rmag = equilibrium_data["rmag"]
        self.rbnd = equilibrium_data["rbnd"]
        self.zmag = equilibrium_data["zmag"]
        self.zbnd = equilibrium_data["zbnd"]
        if T_e is not None:
            Te_Rz = T_e.indica.with_Rz_coords()
            offsets = DataArray(np.linspace(0.0, 0.04, 9), dims="offset", name="offset")
            index = T_e.dims[1]
            # TODO: Consider how to do 2D cubic interpolation
            rhos = concat(
                [
                    self.rho.interp(
                        t=T_e.coords["t"], method="nearest"
                    ).indica.interp2d(
                        R=Te_Rz.coords["R"] - offset,
                        z=Te_Rz.coords["z"],
                        zero_coords={
                            "R": self.rmag.interp(t=T_e.coords["t"], method="nearest"),
                            "z": self.zmag.interp(t=T_e.coords["t"], method="nearest"),
                        },
                        method="cubic",
                    )
                    for offset in offsets
                ],
                offsets,
            )
            # separatrix_indices = apply_ufunc(
            #     find_separatrix_index,
            #     rhos,
            #     input_core_dims=[[index]],
            #     exclude_dims=set((index,))
            # )
            separatrix_indices = rhos.indica.invert_root(
                1.0, index, int(T_e.coords[index][-1]), method="cubic"
            )

            square_residuals = (
                T_e.dropna(index).indica.interp2d(
                    {index: separatrix_indices}, method="cubic"
                )
                - 100.0
            ) ** 2
            best_fits = square_residuals.argmin(dim="offset")
            overall_best = best_fits.mean()
            fluxes = rhos.sel(offset=overall_best, method="nearest")
            offset = float(fluxes.offset)
            offset, accept = offset_picker(offset, T_e, fluxes, best_fits)
            while not accept:
                fluxes = self.rho.interp(
                    t=T_e.coords["t"], method="nearest"
                ).indica.interp2d(
                    R=Te_Rz.coords["R"] - offset, z=Te_Rz.coords["z"], method="cubic"
                )
                offset, accept = offset_picker(offset, T_e, fluxes, best_fits)
            self.R_offset = offset
        else:
            self.R_offset = 0.0

        self.Rmin = min(self.rho.coords["R"])
        self.Rmax = max(self.rho.coords["R"])
        self.zmin = min(self.rho.coords["z"])
        self.zmax = max(self.rho.coords["z"])
        self.corner_angles = [
            np.arctan2(self.zmin - self.zmag, self.Rmax - self.rmag),
            np.arctan2(self.zmax - self.zmag, self.Rmax - self.rmag),
            np.arctan2(self.zmax - self.zmag, self.Rmin - self.rmag),
            np.arctan2(self.zmin - self.zmag, self.Rmin - self.rmag),
        ]

        self.prov_id = hash_vals(**equilibrium_data)
        self.provenance = sess.prov.entity(
            self.prov_id, {prov.PROV_TYPE: "Equilibrium"},
        )
        sess.prov.generation(
            self.provenance, sess.session, time=datetime.datetime.now()
        )
        sess.prov.attribution(self.provenance, sess.agent)
        # TODO: Add PROV dependencies to ``equilibrium_data``

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
        raise NotImplementedError(
            "{} does not implement an 'Btot' method.".format(self.__class__.__name__)
        )

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
        else:
            rmjo = self.rmjo.interp(t=t, method="nearest")
        rho, _ = self.convert_flux_coords(rho, t, kind, "poloidal")
        R = rmjo.indica.interp2d(rho_poloidal=rho, method="cubic") + self.R_offset
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
        R = rmji.indica.interp2d(rho_poloidal=rho, method="cubic") + self.R_offset
        return R, t

    def enclosed_volume(
        self,
        rho: LabeledArray,
        t: Optional[LabeledArray] = None,
        kind: str = "poloidal",
    ) -> Tuple[LabeledArray, LabeledArray]:
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
        raise NotImplementedError(
            "{} does not implement an 'enclosed_volume' "
            "method.".format(self.__class__.__name__)
        )

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
    ) -> Tuple[LabeledArray, LabeledArray]:
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
            R0 = self.rmag.interp(t=t, method="nearest") + self.R_offset
            z0 = self.zmag.interp(t=t, method="nearest")
            reference_rhos = self.rho.interp(t=t, method="nearest")
            reference_rhos.name = self.rho.name
        else:
            corner_angles = self.corner_angles
            R0 = self.rmag + self.R_offset
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
            np.linspace(0.0, minor_rad_max, ngrid), dims=("r",) + minor_rad_max.dims
        )
        R_grid = R0 + minor_rads * np.cos(theta) - self.R_offset
        z_grid = z0 + minor_rads * np.sin(theta)
        fluxes_samples = reference_rhos.indica.interp2d(
            R=R_grid, z=z_grid, zero_coords={"R": R0, "z": z0}, method="cubic"
        ).rename("rho_" + kind)
        fluxes_samples.loc[{"r": 0}] = 0.0
        indices = fluxes_samples.indica.invert_interp(rho, "r", method="cubic")
        return minor_rads.indica.interp2d(r=indices, method="cubic"), t

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
            t = t
        else:
            rho = self.rho
            R_ax = self.rmag
            z_ax = self.zmag
            t = self.rho.coords["t"]
        rho_interp = rho.indica.interp2d(
            R=R - self.R_offset, z=z, zero_coords={"R": R_ax, "z": z_ax}, method="cubic"
        )
        # Correct for any interpolation errors resulting in negative fluxes
        rho_interp = where(
            np.logical_and(rho_interp < 0.0, rho_interp > -1e-12), 0.0, rho_interp
        )
        theta = np.arctan2(z - z_ax, R - self.R_offset - R_ax)
        if kind != "poloidal":
            rho_interp, t = self.convert_flux_coords(rho_interp, t, "poloidal", kind)
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
        R = R0 + self.R_offset + minor_rad * np.cos(theta)
        z = z0 + minor_rad * np.sin(theta)
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
            return rho, t
        if t is not None:
            conversion = self.ftor.interp(t=t, method="nearest")
        else:
            conversion = self.ftor
            t = self.ftor.coords["t"]
        if to_kind == "toroidal":
            flux = conversion.indica.interp2d(rho_poloidal=rho, method="cubic")
        elif to_kind == "poloidal":
            flux = conversion.indica.invert_interp(rho, "rho_poloidal", method="cubic")
        return flux, t
