from indica.models.abstractdiagnostic import DiagnosticModel
from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

from indica.readers import ST40Reader
from indica.models.plasma import example_run as example_plasma
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
import matplotlib.cm as cm

import xarray as xr
from xarray import DataArray
import matplotlib.pylab as plt
import numpy as np


class Interferometry(DiagnosticModel):
    """
    Object representing an interferometer diagnostics
    """

    def __init__(
        self, name: str, instrument_method="get_interferometry",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

        self.Ne = None
        self.los_integral_ne = None

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
                    "transform": self.transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

    def __call__(self, Ne: DataArray = None, t: LabeledArray = None, **kwargs):
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
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.sel(t=t)
        else:
            if Ne is None:
                raise ValueError("Give inputs or assign plasma class!")
        self.Ne = Ne
        # if len(np.shape(t)) == 0:
        #     t = np.array([t])

        x1 = self.transform.x1
        x2 = self.transform.x2
        # los_integral_ne = self.transform.integrate_on_los(Ne, x1, x2, t=t,)
        R = self.transform.R
        dR = R[0][0] - R[0][-1]
        los_integral_ne = Ne.mean()*dR


        self.los_integral_ne = los_integral_ne
        self.t = los_integral_ne.t

        self._build_bckc_dictionary()

        return self.bckc


def example_run():
    plasma = example_plasma()

    # Read equilibrium data and initialize Equilibrium and Flux-surface transform objects
    pulse = 9229
    it = int(len(plasma.t) / 2)
    tplot = plasma.t[it]
    reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)

    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # Create new interferometers diagnostics
    diagnostic_name = "smmh1"
    los_start = np.array([[0.8, 0, 0], [0.8, 0, -0.1], [0.8, 0, -0.2]])
    los_end = np.array([[0.17, 0, 0], [0.17, 0, -0.25], [0.17, 0, -0.2]])
    origin = los_start
    direction = los_end - los_start
    model = Interferometry(diagnostic_name,)
    transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    model.set_transform(transform)
    model.set_flux_transform(plasma.flux_transform)
    model.set_plasma(plasma)
    bckc = model()
    # bckc = model(Ne=plasma.electron_density, t=plasma.t)

    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    )
    channels = model.transform.x1
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))
    for chan in channels:
        plt.plot(
            model.transform.R[chan],
            model.transform.z[chan],
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
        model.transform.rho[chan].sel(t=tplot, method="nearest").plot(
            color=cols[chan], label=f"CH{chan}",
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
    plt.show(block=True)

    return plasma, model, bckc


if __name__ == "__main__":

    example_run()