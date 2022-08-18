from typing import Dict
from typing import Union

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.operators.extrapolate_impurity_density import asymmetry_from_rho_theta
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
        comparison_source: Dict[str, Union[int, str]] = None,
        plot_interactive: bool = True,
    ) -> None:
        self.workflow = workflow
        if default_time is None:
            default_time = float(self.workflow.sxr_emissivity.coords["t"].values[0])
        self.default_time = default_time
        if comparison_source is None:
            comparison_source = {}
        self.comparison_source = comparison_source
        if plot_interactive is True:
            plt.ion()

        self.rho_deriv, self.theta_deriv = self.workflow.sxr_emissivity.attrs[
            "transform"
        ].convert_from_Rz(self.workflow.R, self.workflow.z, self.workflow.t)
        self.rmag = self.workflow.sxr_emissivity.attrs["transform"].equilibrium.rmag
        self.zmag = self.workflow.sxr_emissivity.attrs["transform"].equilibrium.zmag

    def get_comparison_data(
        self, pulse: int, uid: str, instrument: str, quantity: str, **kwargs
    ) -> DataArray:
        raise NotImplementedError("No comparison fetcher")

    def plot_density_2D(
        self, element: str = None, time: Union[int, float] = None, **kwargs
    ):
        if element is None:
            element = self.workflow.high_z
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)
        self.workflow.ion_densities.sel({"element": element}, drop=True).interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel({"t": time}, method="nearest").plot(x="R", ax=ax, **kwargs)
        return fig, ax

    def plot_density_midplane(
        self, threshold_rho: float = None, time: Union[int, float] = None, **kwargs
    ):
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)
        self.workflow.ion_densities.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel(
            {"t": time, "z": self.zmag.sel({"t": time}, method="nearest")},
            method="nearest",
        ).plot.line(
            x="R", ax=ax, **kwargs
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
        self, bolometry_diagnostic: DataArray, time: Union[int, float] = None, **kwargs
    ):
        if bolometry_diagnostic.attrs.get("transform", None) is None:
            return
        kwargs.pop("label", None)
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)
        bolometry_diagnostic.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Bolometry diagnostic", **kwargs
        )
        self.workflow.derived_bolometry.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Derived bolometry", **kwargs
        )
        ax.legend()
        return fig, ax

    def plot_sxr_los_fit(self, time: Union[int, float] = None, **kwargs):
        if time is None:
            time = self.default_time
        additional_data = getattr(self.workflow, "additional_data", None)
        if additional_data is None:
            return
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)
        additional_data["invert_sxr"]["camera_results"][0].camera.sel(
            t=time, method="nearest"
        ).plot(ax=ax, x="sxr_v_rho_poloidal", label="camera", **kwargs)
        additional_data["invert_sxr"]["camera_results"][0].back_integral.sel(
            t=time, method="nearest"
        ).plot(ax=ax, x="sxr_v_rho_poloidal", label="back_integral", **kwargs)
        ax.legend()
        return fig, ax

    def plot_emissivity_midplane_R(
        self, threshold_rho: float = None, time: Union[int, float] = None, **kwargs
    ):
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)
        self.workflow.sxr_emissivity.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel(
            {"t": time, "z": self.zmag.sel({"t": time}, method="nearest")},
            method="nearest",
        ).plot.line(
            x="R", ax=ax, **kwargs
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

    def plot_emissivity_midplane_rho(self, time: Union[int, float] = None, **kwargs):
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        fig, ax = plt.subplots(figsize=figsize)

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

    def plot_asymmetry_high_z(self, time: Union[int, float] = None, **kwargs):
        def asymmetry_parameter_to_modifier(asymmetry_parameter, R):
            return np.exp(
                asymmetry_parameter
                * (
                    R.sel(theta=np.pi, method="nearest", drop=True) ** 2
                    - R.sel(theta=0, method="nearest", drop=True) ** 2
                )
            )

        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        threshold_rho = recover_threshold_rho(1.5e3, self.workflow.electron_temperature)
        flux_surface = getattr(self.workflow, "flux_surface", None)
        if flux_surface is None:
            raise UserWarning("No flux surface object in workflow")
        R_deriv, z_deriv = flux_surface.convert_to_Rz(
            self.workflow.rho, self.workflow.theta, self.workflow.t
        )
        asymmetry_modifier_n_high_z = asymmetry_parameter_to_modifier(
            asymmetry_from_rho_theta(
                self.workflow.n_high_z.copy(),
                flux_surface,
                threshold_rho,
                self.workflow.t,
            ),
            R_deriv,
        )
        asymmetry_n_high_z = (1 - asymmetry_modifier_n_high_z) / (
            1 + asymmetry_modifier_n_high_z
        )
        asymmetry_n_high_z_2 = self.workflow.derived_asymmetry_high_z
        asymmetry_modifier_cxrs_hfs = asymmetry_parameter_to_modifier(
            self.workflow.asymmetry_parameter_high_z.copy(), R_deriv
        )
        asymmetry_cxrs = (1 - asymmetry_modifier_cxrs_hfs) / (
            1 + asymmetry_modifier_cxrs_hfs
        )
        asymmetry_modifier_cxrs_hfs_midz = asymmetry_parameter_to_modifier(
            self.workflow.asymmetry_parameter_other_z.copy(), R_deriv
        )
        asymmetry_cxrs_midz = (1 - asymmetry_modifier_cxrs_hfs_midz) / (
            1 + asymmetry_modifier_cxrs_hfs_midz
        )
        sxr_R_deriv, _ = flux_surface.convert_to_Rz(
            self.workflow.sxr_fitted_asymmetry_parameter.rho_poloidal,
            self.workflow.theta,
            self.workflow.t,
        )
        asymmetry_parameter_emiss = asymmetry_parameter_to_modifier(
            self.workflow.sxr_fitted_asymmetry_parameter, sxr_R_deriv
        )
        asymmetry_emiss = (1 - asymmetry_parameter_emiss) / (
            1 + asymmetry_parameter_emiss
        )

        fig, ax = plt.subplots(figsize=figsize)
        asymmetry_n_high_z.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Asymmetry from n_high_z"
        )
        asymmetry_n_high_z_2.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Asymmetry from n_high_z (2)"
        )
        asymmetry_cxrs.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Asymmetry from ANGF (W)"
        )
        asymmetry_cxrs_midz.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Asymmetry from ANGF (Ni)"
        )
        asymmetry_emiss.sel({"t": time}, method="nearest").plot(
            ax=ax, label="Asymmetry from SXR emissivitiy"
        )
        ax.legend()
        return fig, ax
