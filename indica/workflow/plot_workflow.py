from typing import Union

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.operators.extrapolate_impurity_density import recover_threshold_rho
from .base_workflow import BaseWorkflow


class PlotWorkflow:
    """
    Methods for plotting common comparisons of workflow outputs
    """

    def __init__(
        self,
        workflow: BaseWorkflow,
        default_time: Union[int, float] = None,
        plot_interactive: bool = True,
    ) -> None:
        self.workflow = workflow
        if default_time is None:
            default_time = float(self.workflow.sxr_emissivity.coords["t"].values[0])
        self.default_time = default_time
        if plot_interactive is True:
            plt.ion()

        self.rho_deriv, self.theta_deriv = self.workflow.sxr_emissivity.attrs[
            "transform"
        ].convert_from_Rz(self.workflow.R, self.workflow.z, self.workflow.t)
        self.rmag = self.workflow.sxr_emissivity.attrs["transform"].equilibrium.rmag
        self.zmag = self.workflow.sxr_emissivity.attrs["transform"].equilibrium.zmag

    def plot_density_2D(
        self, element: str, time: Union[int, float] = None, *args, **kwargs
    ):
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()
        self.workflow.ion_densities.sel({"element": element}, drop=True).interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel({"t": time}, method="nearest").plot(x="R", ax=ax, *args, **kwargs)
        return fig, ax

    def plot_density_midplane(
        self,
        threshold_rho: float = None,
        time: Union[int, float] = None,
        *args,
        **kwargs
    ):
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()
        self.workflow.ion_densities.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel(
            {"t": time, "z": self.zmag.sel({"t": time}, method="nearest")},
            method="nearest",
        ).plot.line(
            x="R", ax=ax, *args, **kwargs
        )
        if threshold_rho is not None:
            threshold_R_outer, _ = self.workflow.sxr_emissivity.attrs[
                "transform"
            ].convert_to_Rz(threshold_rho, 0, self.workflow.t)
            threshold_R_inner, _ = self.workflow.sxr_emissivity.attrs[
                "transform"
            ].convert_to_Rz(threshold_rho, np.pi, self.workflow.t)
            ax.axvline(threshold_R_outer.sel({"t": time}, method="nearest"))
            ax.axvline(threshold_R_inner.sel({"t": time}, method="nearest"))
        return fig, ax

    def plot_bolometry_emission(
        self,
        bolometry_diagnostic: DataArray,
        time: Union[int, float] = None,
        *args,
        **kwargs
    ):
        kwargs.pop("label", None)
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()
        bolometry_diagnostic.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Bolometry diagnostic", *args, **kwargs
        )
        self.workflow.derived_bolometry.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Derived bolometry", *args, **kwargs
        )
        ax.legend()
        return fig, ax

    def plot_sxr_los_fit(self, time):
        fig, ax = plt.subplots()
        self.workflow.additional_data["invert_sxr"]["camera_results"][0].camera.sel(
            t=time, method="nearest"
        ).plot(ax=ax, x="sxr_v_rho_poloidal", label="camera")
        self.workflow.additional_data["invert_sxr"]["camera_results"][
            0
        ].back_integral.sel(t=time, method="nearest").plot(
            ax=ax, x="sxr_v_rho_poloidal", label="back_integral"
        )
        ax.legend()
        return fig, ax

    def plot_emissivity_midplane_R(
        self,
        threshold_rho: float = None,
        time: Union[int, float] = None,
        *args,
        **kwargs
    ):
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()
        self.workflow.sxr_emissivity.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel(
            {"t": time, "z": self.zmag.sel({"t": time}, method="nearest")},
            method="nearest",
        ).plot.line(
            x="R", ax=ax, *args, **kwargs
        )
        if threshold_rho is not None:
            threshold_R_outer, _ = self.workflow.sxr_emissivity.attrs[
                "transform"
            ].convert_to_Rz(threshold_rho, 0, self.workflow.t)
            threshold_R_inner, _ = self.workflow.sxr_emissivity.attrs[
                "transform"
            ].convert_to_Rz(threshold_rho, np.pi, self.workflow.t)
            ax.axvline(threshold_R_outer.sel({"t": time}, method="nearest"))
            ax.axvline(threshold_R_inner.sel({"t": time}, method="nearest"))
        return fig, ax

    def plot_emissivity_midplane_rho(
        self, time: Union[int, float] = None, *args, **kwargs
    ):
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()

        emiss_lfs = (
            self.workflow.sxr_emissivity.sel(theta=0, method="nearest")
            .copy()
            .transpose()
        )
        emiss_hfs = self.workflow.sxr_emissivity.sel(
            theta=np.pi, method="nearest"
        ).copy()
        emiss_hfs = emiss_hfs.assign_coords(
            {"rho_poloidal": -emiss_hfs.rho_poloidal}
        ).transpose()

        emiss_all = xr.concat([emiss_hfs[::-1], emiss_lfs], dim="rho_poloidal")

        emiss_all.sel(t=time, method="nearest").plot(ax=ax)

        threshold_rho = recover_threshold_rho(
            1.5e3, self.workflow.electron_temperature
        ).sel(t=time, method="nearest")
        ax.axvline(threshold_rho)
        ax.axvline(-threshold_rho)

        return fig, ax
