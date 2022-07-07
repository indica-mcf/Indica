from typing import Union

from matplotlib import pyplot as plt

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

    def plot_midplane(self, time: Union[int, float] = None, *args, **kwargs):
        if time is None:
            time = self.default_time
        fig, ax = plt.subplots()
        self.workflow.ion_densities.interp(
            {"rho_poloidal": self.rho_deriv, "theta": self.theta_deriv}
        ).sel({"t": time, "z": 0.3}, method="nearest").plot.line(
            x="R", ax=ax, *args, **kwargs
        )
