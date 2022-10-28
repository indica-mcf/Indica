from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray

import xarray as xr
import matplotlib.pylab as plt
import numpy as np


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

    def map_to_los(self, Ne: LabeledArray, t: LabeledArray = None):
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
        return along_los

    def integrate_on_los(self, Ne: LabeledArray, t: LabeledArray = None):
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
        along_los = {}
        los_integral = {}
        los_integral["ne"], along_los["ne"] = self.los_transform.integrate_on_los(
            Ne, t=t, passes=self.passes
        )

        return los_integral, along_los

    def line_integrated_phase_shift(self, Ne: LabeledArray, t: LabeledArray = None):
        raise NotImplementedError("Calculation of phase shift still not implemented")


def example_interferometer():
    from indica.readers import ST40Reader
    from hda.models.plasma import example_plasma
    from indica.equilibrium import Equilibrium
    from indica.converters import FluxSurfaceCoordinates, LineOfSightTransform

    plasma = example_plasma()

    # Read equilibrium data and initialize Equilibrium and Flux-surface transform objects
    pulse = 9229
    tplot = plasma.t[3]
    reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)

    # Assign transforms to plasma object
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # create new interferometer and assign transforms for remapping
    diagnostic_name = "smmh1"
    smmh1 = Interferometer(name=diagnostic_name)
    los_start = np.array([0.8, 0, 0])
    los_end = np.array([0.17, 0, 0])
    origin = tuple(los_start)
    direction = tuple(los_end - los_start)
    los_transform = LineOfSightTransform(
        origin_x=origin[0],
        origin_y=origin[1],
        origin_z=origin[2],
        direction_x=direction[0],
        direction_y=direction[1],
        direction_z=direction[2],
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh1.set_los_transform(los_transform)

    diagnostic_name = "smmh2"
    smmh2 = Interferometer(name=diagnostic_name)
    los_start = np.array([0.8, 0, -0.1])
    los_end = np.array([0.17, 0, -0.25])
    origin = tuple(los_start)
    direction = tuple(los_end - los_start)
    los_transform = LineOfSightTransform(
        origin_x=origin[0],
        origin_y=origin[1],
        origin_z=origin[2],
        direction_x=direction[0],
        direction_y=direction[1],
        direction_z=direction[2],
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh2.set_los_transform(los_transform)

    diagnostic_name = "smmh3"
    smmh3 = Interferometer(name=diagnostic_name)
    los_start = np.array([0.8, 0, -0.2])
    los_end = np.array([0.17, 0, -0.2])
    origin = tuple(los_start)
    direction = tuple(los_end - los_start)
    los_transform = LineOfSightTransform(
        origin_x=origin[0],
        origin_y=origin[1],
        origin_z=origin[2],
        direction_x=direction[0],
        direction_y=direction[1],
        direction_z=direction[2],
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh3.set_los_transform(los_transform)

    ne_smmh1 = []
    ne_smmh2 = []
    ne_smmh3 = []
    for t in plasma.t:
        integral, _ = smmh1.integrate_on_los(plasma.electron_density.sel(t=t), t)
        ne_smmh1.append(integral["ne"])
        integral, _ = smmh2.integrate_on_los(plasma.electron_density.sel(t=t), t)
        ne_smmh2.append(integral["ne"])
        integral, _ = smmh3.integrate_on_los(plasma.electron_density.sel(t=t), t)
        ne_smmh3.append(integral["ne"])
    ne_smmh1 = xr.concat(ne_smmh1, "t").assign_coords(t=plasma.t)
    ne_smmh2 = xr.concat(ne_smmh2, "t").assign_coords(t=plasma.t)
    ne_smmh3 = xr.concat(ne_smmh3, "t").assign_coords(t=plasma.t)

    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    )
    plt.plot(
        smmh1.los_transform.R,
        smmh1.los_transform.z,
        linewidth=3,
        alpha=0.7,
        label="SMMH1",
    )
    plt.plot(
        smmh2.los_transform.R,
        smmh2.los_transform.z,
        linewidth=3,
        alpha=0.7,
        label="SMMH2",
        linestyle="dashed",
    )
    plt.plot(
        smmh3.los_transform.R,
        smmh3.los_transform.z,
        linewidth=3,
        alpha=0.7,
        label="SMMH3",
        linestyle="dotted",
    )
    plt.xlim(0, 1.0)
    plt.ylim(-0.6, 0.6)
    plt.axis("scaled")
    plt.legend()

    plt.figure()
    (ne_smmh1/ne_smmh2).plot(label="SMMH1/SMMH2")
    (ne_smmh1/ne_smmh3).plot(label="SMMH1/SMMH3", linestyle="dashed")
    (ne_smmh2/ne_smmh3).plot(label="SMMH2/SMMH3", linestyle="dotted")
    plt.xlabel("Time (s)")
    plt.ylabel("Ne LOS-integral ratios")
    plt.legend()

    import matplotlib.cm as cm
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    plt.figure()
    for i, t in enumerate(plasma.t):
        plt.plot(plasma.rho, plasma.electron_density.sel(t=t), color=cols[i], label=f"t={t:1.2f} s")
    plt.xlabel("rho")
    plt.ylabel("Ne")
    plt.legend()

    return plasma, smmh1, smmh2
