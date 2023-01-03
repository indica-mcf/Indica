import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


class Interferometry(DiagnosticModel):
    """
    Object representing an interferometer diagnostics
    """

    Ne: DataArray
    los_integral_ne: DataArray

    def __init__(
        self,
        name: str,
        instrument_method="get_interferometry",
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
                self.bckc[quantity] = self.los_integral_ne
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "datatype": datatype,
                    "transform": self.los_transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

    def __call__(self, Ne: DataArray = None, t: LabeledArray = None, calc_rho=False):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile
        t

        Returns
        -------

        """
        # TODO: decide whether to select nearest time-points or interpolate in time!!
        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Ne = self.plasma.electron_density.interp(t=t)
        else:
            if Ne is None:
                raise ValueError("Give inputs or assign plasma class!")
        self.t = t
        self.Ne = Ne

        los_integral_ne = self.los_transform.integrate_on_los(
            Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        self.los_integral_ne = los_integral_ne

        self._build_bckc_dictionary()

        return self.bckc


def example_run(plasma=None, plot=False):
    if plasma is None:
        plasma = example_plasma()

    # Create new interferometers diagnostics
    diagnostic_name = "smmh1"
    los_start = np.array([[0.8, 0, 0], [0.8, 0, -0.1], [0.8, 0, -0.2]])
    los_end = np.array([[0.17, 0, 0], [0.17, 0, -0.25], [0.17, 0, -0.2]])
    origin = los_start
    direction = los_end - los_start
    model = Interferometry(
        diagnostic_name,
    )
    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=2,
    )
    los_transform.set_equilibrium(plasma.equilibrium)
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]

        plt.figure()
        plasma.equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
            levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        )
        channels = model.los_transform.x1
        cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))
        for chan in channels:
            plt.plot(
                model.los_transform.R[chan],
                model.los_transform.z[chan],
                linewidth=3,
                color=cols[chan],
                alpha=0.7,
                label=f"CH{chan}",
            )

        plt.xlim(0, 1.0)
        plt.ylim(-0.6, 0.6)
        plt.axis("scaled")
        plt.legend()

        # Plot LOS mapping on equilibrium
        plt.figure()
        for chan in channels:
            model.los_transform.rho[chan].sel(t=tplot, method="nearest").plot(
                color=cols[chan],
                label=f"CH{chan}",
            )
        plt.xlabel("Path along the LOS")
        plt.ylabel("Rho-poloidal")
        plt.legend()

        # Plot back-calculated values
        plt.figure()
        for chan in channels:
            bckc["ne"].sel(channel=chan).plot(label=f"CH{chan}", color=cols[chan])
        plt.xlabel("Time (s)")
        plt.ylabel("Ne LOS-integrals (m^-2)")
        plt.legend()

        # Plot the profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.Ne.rho_poloidal,
                model.Ne.sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("Ne (m^-3)")
        plt.legend()

    return plasma, model, bckc
