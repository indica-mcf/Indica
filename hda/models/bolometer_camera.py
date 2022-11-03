from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.converters import FluxSurfaceCoordinates
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

import xarray as xr
from xarray import DataArray
import matplotlib.pylab as plt
import numpy as np
from typing import Tuple


class Bolometer:
    """
    Object representing a bolometer camera diagnostic
    """

    def __init__(
        self,
        name: str,
        origin: LabeledArray,
        direction: LabeledArray,
        dl: float = 0.005,
        passes: int = 1,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        instrument_method="get_radiation",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.bckc = {}
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

    def set_los_transform(self, transform: LineOfSightTransform):
        """
        Set line of sight transform of diagnostic

        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        """
        self.los_transform = transform
        self.bckc = {}
        self.los_integral_radiation = None


    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)
        self.bckc = {}
        self.los_integral_radiation = None

    def _build_bckc_dictionary(self):
        bckc = {}

        available_quantities = AVAILABLE_QUANTITIES[self.instrument_method]

        quantity = "brightness"
        data_type = ("brightness", "total")
        bckc[quantity] = self.los_integral_radiation
        for quant in bckc.keys():
            if quant not in available_quantities:
                raise ValueError(
                    f"Quantity {quantity} not in AVAILABLE_QUANTITIES "
                    f"({list(available_quantities)})"
                )
            error = xr.full_like(bckc[quantity], 0.0)
            stdev = xr.full_like(bckc[quantity], 0.0)
            bckc[quantity].attrs = {
                "datatype": data_type,
                "transform": self.los_transform,
                "error": error,
                "stdev": stdev,
                "provenance": str(self),
            }

        self.bckc = bckc
        return bckc

    def __call__(
        self,
        Ne: DataArray,
        Nimp: DataArray,
        fractional_abundance: dict,
        cooling_factor: DataArray,
        t: LabeledArray = None,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Nimp
            Impurity density profiles (dims = "rho", "t", "element")
        fractional_abundance
            Fractional abundance dictionary of DataArrays of each element to be included
        cooling_factor
            Cooling factor dictionary of DataArrays of each element to be included
        t
            Time (s) for remapping on equilibrium reconstruction

        Returns
        -------
        Dictionary of back-calculated quantities (identical structure returned by abstractreader.py)

        """
        x1 = self.los_transform.x1
        x2 = self.los_transform.x2
        elements = Nimp.element.values
        fz, lz, = fractional_abundance, cooling_factor
        emission = []
        for ielem, elem in enumerate(elements):
            emission.append(lz[elem].sum("ion_charges") * Nimp.sel(element=elem) * Ne)
        emission = xr.concat(emission, "element")
        los_integral = self.los_transform.integrate_on_los(
            emission.sum("element"), x1, x2, t=t,
        )

        self.emission = emission
        self.los_integral_radiation = los_integral

        bckc = self._build_bckc_dictionary()

        return bckc


def example_bolometer():
    from indica.readers import ST40Reader
    from hda.models.plasma import example_plasma
    from indica.equilibrium import Equilibrium
    from indica.converters import FluxSurfaceCoordinates
    import matplotlib.cm as cm

    # TODO: solve issue of LOS sometimes crossing bad EFIT reconstruction outside of the separatrix

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

    # Assign transforms to plasma object
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # Create new interferometers diagnostics
    diagnostic_name = "bolo_Rz"
    nchannels = 11
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.43, -0.43, nchannels)
    los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start
    bolo = Bolometer(
        diagnostic_name, origin, direction, machine_dimensions=plasma.machine_dimensions
    )
    bolo.set_flux_transform(plasma.flux_transform)
    bckc = bolo(
        plasma.electron_density,
        plasma.impurity_density,
        plasma.fz,
        plasma.lz_tot,
        t=plasma.t,
    )

    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    )
    channels = bolo.los_transform.x1
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))
    for chan in channels:
        plt.plot(
            bolo.los_transform.R[chan],
            bolo.los_transform.z[chan],
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
        bolo.los_transform.rho[chan].sel(t=tplot, method="nearest").plot(
            color=cols[chan], label=f"CH{chan}",
        )
    plt.xlabel("Path along the LOS")
    plt.ylabel("Rho-poloidal")
    plt.legend()

    # Plot back-calculated values
    plt.figure()
    for chan in channels:
        bckc["brightness"].sel(channel=chan).plot(
            label=f"CH{chan}", color=cols[chan]
        )
    plt.xlabel("Time (s)")
    plt.ylabel("BOLO LOS-integrals (W/m^2)")
    plt.legend()

    # Plot the radiation profiles
    cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    plt.figure()
    for i, t in enumerate(plasma.t):
        plt.plot(
            bolo.emission.rho_poloidal,
            bolo.emission.sum("element").sel(t=t),
            color=cols_time[i],
            label=f"t={t:1.2f} s",
        )
    plt.xlabel("rho")
    plt.ylabel("Local radiated power (W/m^3)")
    plt.legend()

    return plasma, bolo
