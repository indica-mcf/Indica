from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray

import xarray as xr
from xarray import DataArray
import matplotlib.pylab as plt
import numpy as np
import inspect


class Interferometer:
    """
    Object representing an interferometer diagnostics
    """

    def __init__(self, name: str):
        self.name = name

    def set_los_transform(self, transform: LineOfSightTransform, passes: int = 2):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform
        self.passes = passes

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)

    def map_to_los(self, Ne: DataArray, t: LabeledArray = None):
        """
        Map interferometer measurements along line of sight

        Parameters
        ----------
        Ne
            1D profile of the electron density
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        along_los = {}
        along_los["ne"] = self.los_transform.map_to_los(Ne, t=t)
        self.along_los = along_los

        return along_los

    def integrate_on_los(self, Ne: DataArray, t: LabeledArray = None):
        """
        Calculate the integral of the interferometer measurement along the line of sight

        Parameters
        ----------
        Ne
            1D profile of the electron density
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        los_integral = {}
        x1 = self.los_transform.x1
        x2 = self.los_transform.x2
        los_integral["ne"] = self.los_transform.integrate_on_los(
            Ne, x1, x2, t=t, passes=self.passes
        )
        self.los_integral = los_integral

        return los_integral

    def line_integrated_phase_shift(self, Ne: DataArray, t: LabeledArray = None):
        """
        Full forward model requires the calculation of the measured phase shift
        that is currently not saved to ST40 database...
        """
        raise NotImplementedError("Calculation of phase shift still not implemented")

    def build_bckc_dictionary(self):
        if not hasattr(self, "bckc"):
            bckc = {}

        bckc["ne"] = self.los_integral["ne"]
        error = xr.full_like(bckc["ne"], 0.0)
        stdev = xr.full_like(bckc["ne"], 0.0)
        bckc["ne"].attrs = {
            "datatype": ("density", "electrons"),
            "transform": self.los_transform,
            "error": error,
            "stdev": stdev,
            "provenance": str(self),
        }

        self.bckc = bckc
        return bckc

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
        self.integrate_on_los(Ne, t=t)
        bckc = self.build_bckc_dictionary()
        return bckc


def example_interferometer():
    from indica.readers import ST40Reader
    from hda.models.plasma import example_plasma
    from indica.equilibrium import Equilibrium
    from indica.converters import FluxSurfaceCoordinates
    import matplotlib.cm as cm

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

    # Assign transforms to plasma object
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # Create new interferometers diagnostics
    diagnostic_name = "smmh1"
    smm = Interferometer(name=diagnostic_name)
    los_start = np.array([[0.8, 0, 0], [0.8, 0, -0.1], [0.8, 0, -0.2]])
    los_end = np.array([[0.17, 0, 0], [0.17, 0, -0.25], [0.17, 0, -0.2]])
    origin = los_start
    direction = los_end - los_start
    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )

    los_transform.set_flux_transform(plasma.flux_transform)
    los_transform.convert_to_rho(los_transform.x1, los_transform.x2, t=plasma.t)
    smm.set_los_transform(los_transform)

    # Calculate the forward model of the measurement
    smm(plasma.electron_density)

    # Plot lines of sight on the poloidal plane
    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    )
    for channel in los_transform.x1:
        plt.plot(
            smm.los_transform.R[channel],
            smm.los_transform.z[channel],
            linewidth=3,
            alpha=0.7,
            label=f"SMM chan {channel}",
        )
    plt.xlim(0, 1.0)
    plt.ylim(-0.6, 0.6)
    plt.axis("scaled")
    plt.legend()

    # Plot the electron density profiles
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    plt.figure()
    for i, t in enumerate(plasma.t):
        plt.plot(
            plasma.rho,
            plasma.electron_density.sel(t=t),
            color=cols[i],
            label=f"t={t:1.2f} s",
        )
    plt.xlabel("rho")
    plt.ylabel("Ne")
    plt.legend()

    # Plot the ratios of the LOS-integrals from the different lines of sight
    plt.figure()
    reference = [0, 1]
    for channel in los_transform.x1:
        for ref in reference:
            if channel == ref or channel < ref:
                continue
            (smm.bckc["ne"].sel(channel=channel) / smm.bckc["ne"].sel(channel=ref)).plot(
                label=f"SMM chan {channel}/{ref}"
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Ne LOS-integral ratios")
    plt.legend()

    return plasma, smm
