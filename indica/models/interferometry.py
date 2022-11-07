from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

import xarray as xr
from xarray import DataArray
import matplotlib.pylab as plt
import numpy as np
from typing import Tuple


class Interferometry:
    """
    Object representing an interferometer diagnostics
    """

    def __init__(
        self,
        name: str,
        origin: LabeledArray,
        direction: LabeledArray,
        dl: float = 0.005,
        passes: int = 2,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        instrument_method="get_interferometry",
    ):

        self.name = name
        self.instrument_method = instrument_method

        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

        self.los_transform = LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            name=name,
            dl=dl,
            machine_dimensions=machine_dimensions,
            passes=passes,
        )

        self.bckc = {}
        self.Ne = None
        self.los_integral_ne = None

    def set_los_transform(self, transform: LineOfSightTransform):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform
        self.bckc = {}

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)
        self.bckc = {}

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

    def __call__(self, Ne: DataArray, t: LabeledArray = None):
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

        self.Ne = Ne

        x1 = self.los_transform.x1
        x2 = self.los_transform.x2
        los_integral_ne = self.los_transform.integrate_on_los(Ne, x1, x2, t=t,)

        self.los_integral_ne = los_integral_ne
        self.t = los_integral_ne.t

        self._build_bckc_dictionary()

        return self.bckc


def example_run():
    from indica.readers import ST40Reader
    from hda.models.plasma import example_plasma
    from indica.equilibrium import Equilibrium
    from indica.converters import FluxSurfaceCoordinates
    import matplotlib.cm as cm

    plasma = example_plasma()
    plasma.build_atomic_data()

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
    model = Interferometry(
        diagnostic_name,
        origin,
        direction,
        passes=2,
        machine_dimensions=plasma.machine_dimensions,
    )
    model.set_flux_transform(plasma.flux_transform)
    bckc = model(plasma.electron_density, t=plasma.t,)

    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
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
    for i, t in enumerate(plasma.t):
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
