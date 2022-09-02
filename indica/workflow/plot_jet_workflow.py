from pathlib import Path
from typing import Dict
from typing import Union

from matplotlib import pyplot as plt
import numpy as np
from sal.client import SALClient
from xarray import DataArray

from .jet_workflow import JetWorkflow
from .plot_workflow import PlotWorkflow

SAL = SALClient("https://sal.jet.uk")

DEFAULT_COMPARISON_SOURCE: Dict[str, str] = {
    "instrument": "wsxp",
    "uid": "msertoli",
    "sxr_los_r": "sxlr",
    "sxr_los_z": "sxlz",
    "sxr_los_r_shift": "ersh",
    "sxr_los_z_shift": "ezsh",
    "n_high_z": "hzmp",
    "n_low_z": "lzav",
    "n_other_z": "ozmp",
    "back_calc_bolometry": "blbc",
    "back_calc_sxr": "sxbc",
}

DEFAULT_WALL_COORDS_FILE = Path(__file__).parent / "wall_coords_jet.txt"


class PlotJetWorkflow(PlotWorkflow):
    def __init__(
        self,
        workflow: JetWorkflow,
        default_time: Union[int, float] = None,
        comparison_source: Dict[str, Union[int, str]] = None,
        plot_interactive: bool = True,
    ):
        if not isinstance(workflow, JetWorkflow):
            raise UserWarning(
                "PlotJetWorkflow relies on JetWorkflow specific properties"
            )
        super().__init__(
            workflow=workflow,
            default_time=default_time,
            comparison_source=comparison_source,
            plot_interactive=plot_interactive,
        )
        self.workflow: JetWorkflow
        if self.comparison_source.get("pulse") is None:
            self.comparison_source["pulse"] = self.workflow.pulse
        for key, val in DEFAULT_COMPARISON_SOURCE.items():
            if self.comparison_source.get(key) is None:
                self.comparison_source[key] = val

    def __call__(self, **kwargs) -> None:
        """
        Plot everything
        """
        attr: str
        for attr in self.__dir__():
            if attr == "plot_density_midplane":
                self.plot_density_midplane(  # type: ignore
                    threshold_rho=self.workflow.additional_data["extrapolate_n_high_z"][
                        "threshold_rho"
                    ],
                    yscale="log",
                    **{key: val for key, val in kwargs.items() if key != "yscale"},
                )
            elif attr == "plot_density_2D":
                self.plot_density_2D(  # type: ignore
                    element=kwargs.get("element", "w"),
                    **{key: val for key, val in kwargs.items() if key != "element"},
                )
            elif attr == "plot_bolometry_emission":
                self.plot_bolometry_emission(  # type: ignore
                    bolometry_diagnostic=self.workflow.diagnostics["bolo"]["kb5v"],
                    **kwargs,
                )
            elif attr.startswith("plot_"):
                getattr(self, attr)(**kwargs)

    def get_comparison_data(
        self, pulse: int, uid: str, instrument: str, quantity: str, **kwargs
    ) -> DataArray:
        signal = SAL.get(
            "/pulse/{}/ppf/signal/{}/{}/{}".format(pulse, uid, instrument, quantity)
        )
        coords, dims = [], []
        for dimension in signal.dimensions:
            if dimension.temporal:
                dimname = "t"
            else:
                dimname = "x"
            coords.append((dimname, dimension.data))
            dims.append(dimname)
        return DataArray(
            data=signal.data,
            coords=coords,
            dims=dims,
        )

    def plot_los(self, time: Union[int, float] = None, **kwargs):
        if time is None:
            time = self.default_time
        figsize = kwargs.pop("figsize", None)
        levels = kwargs.pop("levels", 20)
        colors = kwargs.pop("colors", "black")
        equil_name = kwargs.pop("equilibrium", "efit_equilibrium")
        wall_coords_file = Path(
            kwargs.pop("wall_coords_file", DEFAULT_WALL_COORDS_FILE)
        ).expanduser()
        equilibrium = getattr(self.workflow, equil_name, None)
        if equilibrium is None:
            raise UserWarning(
                "Equilibrium object {} does not exist in workflow".format(equil_name)
            )
        emissivity = self.workflow.sxr_emissivity.copy()
        diagnostics = getattr(self.workflow, "diagnostics", None)
        if diagnostics is None:
            raise UserWarning("No diagnostic data in workflow")
        los_zip = zip(
            diagnostics["sxr"]["v"].transform.x_start.values,
            diagnostics["sxr"]["v"].transform.x_end.values,
            diagnostics["sxr"]["v"].transform.z_start.values,
            diagnostics["sxr"]["v"].transform.z_end.values,
        )

        ppf_los_r_shift = self.get_comparison_data(
            int(self.comparison_source["pulse"]),
            str(self.comparison_source["uid"]),
            str(self.comparison_source["instrument"]),
            str(self.comparison_source["sxr_los_r_shift"]),
        )
        ppf_sxr_los_r = (
            self.get_comparison_data(
                int(self.comparison_source["pulse"]),
                str(self.comparison_source["uid"]),
                str(self.comparison_source["instrument"]),
                str(self.comparison_source["sxr_los_r"]),
            )
            + ppf_los_r_shift[0]
        )
        ppf_los_z_shift = self.get_comparison_data(
            int(self.comparison_source["pulse"]),
            str(self.comparison_source["uid"]),
            str(self.comparison_source["instrument"]),
            str(self.comparison_source["sxr_los_z_shift"]),
        )
        ppf_sxr_los_z = (
            self.get_comparison_data(
                int(self.comparison_source["pulse"]),
                str(self.comparison_source["uid"]),
                str(self.comparison_source["instrument"]),
                str(self.comparison_source["sxr_los_z"]),
            )
            + ppf_los_z_shift[0]
        )
        ppf_los_zip = zip(
            ppf_sxr_los_r.isel(t=0).values,
            ppf_sxr_los_r.isel(t=-1).values,
            ppf_sxr_los_z.isel(t=0).values,
            ppf_sxr_los_z.isel(t=-1).values,
        )
        ppf_los_shift = np.sqrt(ppf_los_r_shift[0] ** 2 + ppf_los_z_shift[0] ** 2)

        wall_data = np.genfromtxt(wall_coords_file)
        machine_dims = diagnostics["sxr"]["v"].transform._machine_dims

        fig, ax = plt.subplots(figsize=figsize)
        equilibrium.psi.sel({"t": time}, method="nearest").plot.contour(
            ax=ax, x="R", levels=levels, colors=colors
        )
        emissivity.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel({"t": time}, method="nearest").plot(ax=ax, x="R")
        for coord in los_zip:
            line1 = ax.plot(coord[:2], coord[2:], color="red", linestyle="dashed")
        for coord in ppf_los_zip:
            line2 = ax.plot(coord[:2], coord[2:], color="green", linestyle="dashed")
        ax.plot(wall_data[:, 0], wall_data[:, 1], color="black", linewidth=2.5)
        ax.set_xlim(*machine_dims[0])
        ax.set_ylim(*machine_dims[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("z (m)")
        line1[0].set_label("SXR LOS")
        line2[0].set_label("PPF SXR LOS (shift: {:4.2f}m)".format(ppf_los_shift.values))
        ax.legend()
        return fig, ax

    def plot_sxr_los_fit(self, time: Union[int, float] = None, **kwargs):
        fig, ax = super().plot_sxr_los_fit(time=time, **kwargs)
        self.get_comparison_data(
            pulse=int(self.comparison_source["pulse"]),
            uid=str(self.comparison_source["uid"]),
            instrument=str(self.comparison_source["instrument"]),
            quantity=str(self.comparison_source["back_calc_sxr"]),
        ).sel(t=time, method="nearest").plot(
            ax=ax, x="x", label="PPF comparison", **kwargs
        )
        ax.legend()
        return fig, ax

    def plot_asymmetry_high_z(self, time: Union[int, float] = None, **kwargs):
        fig, ax = super().plot_asymmetry_high_z(time=time, **kwargs)
        hzmp = self.get_comparison_data(
            pulse=int(self.comparison_source["pulse"]),
            uid=str(self.comparison_source["uid"]),
            instrument=str(self.comparison_source["instrument"]),
            quantity=str(self.comparison_source["n_high_z"]),
        )
        if hzmp is not None:
            hzmp_lfs = hzmp.where(hzmp.x > 0.0, drop=True).interp(
                {"x": self.workflow.rho}
            )
            hzmp_hfs = hzmp.where(hzmp.x < 0.0, drop=True)
            hzmp_hfs["x"] = -hzmp_hfs["x"]
            hzmp_hfs = hzmp_hfs.interp({"x": self.workflow.rho})
            asymmetry_hzmp = (hzmp_lfs - hzmp_hfs) / (hzmp_lfs + hzmp_hfs)
            asymmetry_hzmp.sel({"t": time}, method="nearest").plot(
                ax=ax, label="PPF comparison data"
            )
        ax.legend()
        return fig, ax
