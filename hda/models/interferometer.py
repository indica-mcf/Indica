from indica.converters.line_of_sight import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray

import matplotlib.pylab as plt
import numpy as np

class Interferometer:
    """
    Object representing an interferometer diagnostics
    """

    def __init__(self, name: str):
        self.name = name

    def set_los_transform(self, transform: LineOfSightTransform, passes:int=2):
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
        los_integral["ne"], along_los["ne"] = self.los_transform.integrate_on_los(Ne, t=t, passes=self.passes)

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
        origin=origin,
        direction=direction,
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh1.set_los_transform(los_transform)

    diagnostic_name = "smmh2"
    smmh2 = Interferometer(name=diagnostic_name)
    los_start = np.array([0.8, 0, 0])
    los_end = np.array([0.17, 0, -0.25])
    origin = tuple(los_start)
    direction = tuple(los_end - los_start)
    los_transform = LineOfSightTransform(
        origin=origin,
        direction=direction,
        name=diagnostic_name,
        dl=0.006,
        machine_dimensions=plasma.machine_dimensions,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=plasma.t)
    smmh2.set_los_transform(los_transform)

    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    plt.plot(smmh1.los_transform.R, smmh1.los_transform.z, linewidth=3, alpha=0.7, label="SMMH1")
    plt.plot(smmh2.los_transform.R, smmh2.los_transform.z, linewidth=3, alpha=0.7, label="SMMH2", linestyle="dashed")
    plt.xlim(0, 1.0)
    plt.ylim(-0.6, 0.6)
    plt.axis("scaled")
    plt.legend()

    # for t in plasma.t

    return plasma, smmh1, smmh2
