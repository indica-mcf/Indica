import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.transect import TransectCoordinates
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


class ThomsonScattering(DiagnosticModel):
    """
    Object representing a Thomson scattering diagnostic
    """

    transform: TransectCoordinates

    def __init__(
        self,
        name: str,
        instrument_method="get_thomson_scattering",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "ne":
                quantity = quant
                self.bckc[quantity] = self.Ne_at_channels
            elif quant == "te":
                quantity = quant
                self.bckc[quantity] = self.Te_at_channels
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            error = xr.full_like(self.bckc[quantity], 0.0)
            stdev = xr.full_like(self.bckc[quantity], 0.0)
            self.bckc[quantity].attrs = {
                "datatype": datatype,
                "transform": self.transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
            }

    def __call__(
        self,
        Ne: DataArray = None,
        Te: DataArray = None,
        t: LabeledArray = None,
        calc_rho: bool = False,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Te
            Electron temperature profile (dims = "rho", "t")

        Returns
        -------
        Dictionary of back-calculated quantities (identical to abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Ne = self.plasma.electron_density.interp(t=t)
            Te = self.plasma.electron_temperature.interp(t=t)
        else:
            if Ne is None or Te is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Ne = Ne
        self.Te = Te

        Ne_at_channels = self.transform.map_to_rho(
            Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        Te_at_channels = self.transform.map_to_rho(
            Te,
            t=self.t,
            calc_rho=calc_rho,
        )

        self.Ne_at_channels = Ne_at_channels
        self.Te_at_channels = Te_at_channels

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    diagnostic_name: str = "ts",
    plasma=None,
    plot=False,
):

    # TODO: LOS sometimes crossing bad EFIT reconstruction and crashing...

    if plasma is None:
        plasma = example_plasma()

    # Create new interferometers diagnostics
    nchannels = 11
    x_positions = np.linspace(0.2, 0.8, nchannels)
    y_positions = np.linspace(0.0, 0.0, nchannels)
    z_positions = np.linspace(0.0, 0.0, nchannels)

    transform = TransectCoordinates(
        x_positions,
        y_positions,
        z_positions,
        diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
    )
    transform.set_equilibrium(plasma.equilibrium)
    model = ThomsonScattering(
        diagnostic_name,
    )
    model.set_transform(transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        levels = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.equilibrium.rho.sel(t=t, method="nearest").plot.contour(
                levels=levels, alpha=0.5, colors=[cols_time[i]] * len(levels)
            )
        plt.scatter(
            model.transform.R,
            model.transform.z,
            label="Channels",
            marker="*",
            color="k",
        )
        plt.xlim(0, 1.0)
        plt.ylim(-0.6, 0.6)
        plt.axis("scaled")
        plt.legend()

        # Plot LOS mapping on equilibrium
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.transform.R,
                model.transform.rho.sel(t=t, method="nearest"),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                marker="o",
            )
        plt.xlabel("Position of measurement on flux space")
        plt.ylabel("Rho-poloidal")
        plt.legend()

        # Plot back-calculated profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_density.sel(t=t).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Ne = bckc["ne"].sel(t=t, method="nearest")
            plt.scatter(Ne.rho_poloidal, Ne, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured electron density (m^-3)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_temperature.sel(t=t).plot(
                color=cols_time[i],
                label=f"t={t:1.2f} s",
                alpha=0.7,
            )
            Te = bckc["te"].sel(t=t, method="nearest")
            plt.scatter(Te.rho_poloidal, Te, color=cols_time[i], marker="o", alpha=0.7)
        plt.xlabel("Channel")
        plt.ylabel("Measured electron temperature (eV)")
        plt.legend()

    return plasma, model, bckc
