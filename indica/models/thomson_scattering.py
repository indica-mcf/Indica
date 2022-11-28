from indica.converters.transect import TransectCoordinates
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

from indica.readers import ST40Reader
from indica.models.plasma import example_run as example_plasma
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates

import matplotlib.cm as cm

import xarray as xr
from xarray import DataArray
import matplotlib.pylab as plt
import numpy as np


class ThomsonScattering(DiagnosticModel):
    """
    Object representing a Thomson scattering diagnostic
    """

    def __init__(
        self, name: str, instrument_method="get_thomson_scattering",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]
        self.transform: TransectCoordinates = None

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
        self, Ne: DataArray = None, Te: DataArray = None, t: LabeledArray = None,
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
        Dictionary of back-calculated quantities (identical structure returned by abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Ne = self.plasma.electron_density.sel(t=t)
            Te = self.plasma.electron_temperature.sel(t=t)
        else:
            if Ne is None or Nimp is None or Lz is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.Ne = Ne
        self.Te = Te

        x1 = self.transform.x1
        x2 = self.transform.x2

        if len(np.shape(t)) == 0:
            t = np.array([t])

        Ne_at_channels = self.transform.map_to_channels(Ne, x1, x2, t=t,)
        Te_at_channels = self.transform.map_to_channels(Te, x1, x2, t=t,)

        self.Ne_at_channels = Ne_at_channels
        self.Te_at_channels = Te_at_channels
        self.t = Ne_at_channels.t

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    diagnostic_name: str = "ts", plasma=None, plot=False,
):

    # TODO: solve issue of LOS sometimes crossing bad EFIT reconstruction outside of the separatrix

    if plasma is None:
        plasma = example_plasma()

        # Read equilibrium data and initialize Equilibrium and Flux-surface transform objects
        pulse = 9229
        it = int(len(plasma.t) / 2)
        reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)

        equilibrium_data = reader.get("", "efit", 0)
        equilibrium = Equilibrium(equilibrium_data)
        flux_transform = FluxSurfaceCoordinates("poloidal")
        flux_transform.set_equilibrium(equilibrium)

        # Assign transforms to plasma object
        plasma.set_equilibrium(equilibrium)
        plasma.set_flux_transform(flux_transform)

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
    model = ThomsonScattering(diagnostic_name,)
    model.set_transform(transform)
    model.set_flux_transform(plasma.flux_transform)
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
            label=f"Channels",
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
            bckc["ne"].sel(t=t, method="nearest").plot(
                label=f"t={t:1.2f} s", color=cols_time[i], marker="o", alpha=0.7
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured electron density (m^-3)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            bckc["te"].sel(t=t, method="nearest").plot(
                label=f"t={t:1.2f} s", color=cols_time[i], marker="o", alpha=0.7
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured electron temperature (eV)")
        plt.legend()

        # Plot the plasma profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_density.sel(t=t).plot(
                color=cols_time[i], label=f"t={t:1.2f} s", alpha=0.7,
            )
        plt.xlabel("rho")
        plt.ylabel("Electron density (eV)")
        plt.legend()

        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plasma.electron_temperature.sel(t=t).plot(
                color=cols_time[i], label=f"t={t:1.2f} s", alpha=0.7,
            )
        plt.xlabel("rho")
        plt.ylabel("Electron temperature (eV)")
        plt.legend()

    return plasma, model, bckc


def xy_camera_views(
    diagnostic_name="bolo_xy", option: int = 0, side: int = 0, plasma=None
):
    from copy import deepcopy

    def get_geometry(option: int, side: int):
        # Option 0 = SXR as is
        # Option 1 = 0.4 - 0.8
        # Option 2 = SXR modified, wide slit
        # Option 3 = SXR modified, thin slit
        # Option 4 = 12 channels, thin slit
        geometry = {}
        nsensors = [8, 8, 8, 8, 12]

        sensor_size = [5.08, 5.08, 5.08, 5.08, 5.08]
        sensor_distance = [5.08, 2.54, 5.08, 5.08, 5.08]
        pinhole_width = [1.0, 1.5, 3.5, 1.0, 1.0]

        sensor_center = [
            [365.26, -1295.21, 0],
            [444.31, -1212.53, 0],
            [444.0, -1258.62, 0],
            [444.0, -1258.62, 0],
            [454.57, -1283.49, 0],
        ]
        pinhole_center = [
            [369.54, -1225.34, 0],
            [451.46, -1162.12, 0],
            [448.1, -1188.84, 0],
            [448.1, -1188.84, 0],
            [461.51, -1159.99, 0],
        ]

        x_shifts = (
            +(np.arange(nsensors[option]) - (nsensors[option] - 1) / 2.0)
            * sensor_distance[option]
        )
        sensor_locations = []
        for x in x_shifts:
            xyz = deepcopy(sensor_center[option])
            xyz[0] += x
            sensor_locations.append(xyz)
        geometry["sensor_location"] = np.array(sensor_locations) * 1.0e-3

        pinhole_location = pinhole_center[option]
        pinhole_location[0] += pinhole_width[option] / 2.0 * side
        geometry["pinhole_location"] = (
            np.array([pinhole_location] * nsensors[option]) * 1.0e-3
        )

        return geometry

        # import pandas as pd
        #
        # _file = "/home/marco.sertoli/data/xy_bolometer_senario_parameters.csv"
        # df = pd.read_csv(_file)
        #
        # value = df["Pinhole Location (x,y,z) [mm]"][option]
        # geometry["pinhole_location"] = np.array([np.array(
        #     [float(v) * 1.0e-3 for v in value.split(",")]
        # )]*nsensors[option])
        #
        # sensor_locations = []
        # for chan in range(8):
        #     value = df[f"Sensor {chan+1} Location (x,y,z) [mm]"][option]
        #     sensor_locations.append([float(v) * 1.0e-3 for v in value.split(",")])
        # geometry["sensor_location"] = np.array(sensor_locations)
        #
        # value = df["Pinhole Size [mm]: width"][option]
        # geometry["pinhole_size"] = float(value) * 1.e3
        #
        # value = df["Sensor Size w"][option]
        # geometry["sensor_size"] = float(value) * 1.e3
        #
        # value = df["Solid Angle (sr) "][option]
        # geometry["solid_angle"] = float(value)
        #
        # return geometry

    geometry = get_geometry(option, side)

    sensors = geometry["sensor_location"]
    pinhole = geometry["pinhole_location"]
    origin = pinhole
    direction = pinhole - sensors

    _start = origin
    _end = origin + 10 * direction
    direction = _end - _start

    return example_run("bolo_xy", origin=origin, direction=direction, plasma=plasma)


def viewing_cone(option: int = 2, plasma=None):
    plasma, model0, bckc0 = xy_camera_views(option=option, side=0, plasma=plasma)
    _, model1, bckc1 = xy_camera_views(option=option, side=1, plasma=plasma)
    _, model2, bckc2 = xy_camera_views(option=option, side=-1, plasma=plasma)

    cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    tind_plot = [0, int(len(plasma.t) / 2.0), len(plasma.t) - 1]

    # Plot the radiation profiles
    plt.figure()
    for i, t in enumerate(plasma.t.values):
        plt.plot(
            model0.emission.rho_poloidal,
            model0.emission.sum("element").sel(t=t),
            color=cols_time[i],
            label=f"t={t:1.2f} s",
        )
    plt.xlabel("rho")
    plt.ylabel("Local radiated power (W/m^3)")
    plt.legend()

    # Plot the radiation profiles on minor radius
    plt.figure()
    for i, t in enumerate(plasma.t.values):
        R_lfs = plasma.equilibrium.rmjo.sel(t=t, method="nearest").interp(
            rho_poloidal=model0.emission.rho_poloidal
        )
        R_hfs = plasma.equilibrium.rmji.sel(t=t, method="nearest").interp(
            rho_poloidal=model0.emission.rho_poloidal
        )
        min_r = (R_lfs - R_hfs) / 2.0
        plt.plot(
            min_r,
            model0.emission.sum("element").sel(t=t),
            color=cols_time[i],
            label=f"t={t:1.2f} s",
        )
    plt.xlabel("Minor radius (m)")
    plt.ylabel("Local radiated power (W/m^3)")
    plt.legend()

    # Plot forward model
    plt.figure()
    for i in tind_plot:
        impact = model0.transform.impact_xyz.value
        plt.plot(
            impact,
            bckc0["brightness"].sel(t=plasma.t[i], method="nearest"),
            label=f"t={plasma.t[i].values:1.2f} s",
            color=cols_time[i],
        )
        y0 = bckc1["brightness"].sel(t=plasma.t[i], method="nearest")
        y1 = bckc2["brightness"].sel(t=plasma.t[i], method="nearest")
        plt.fill_between(impact, y0, y1, color=cols_time[i], alpha=0.5)
    plt.xlabel("Impact parameter (m)")
    plt.ylabel("Measured brightness (W/m^2)")
    plt.title("Centre (dashed) and side of cones (shaded)")
    plt.legend()

    model0.transform.plot_los(tplot=np.mean(plasma.t))
    for chan in model0.transform.x1:
        plt.plot(
            model1.transform.x[chan],
            model1.transform.y[chan],
            color="gray",
            linestyle="dashed",
        )
        plt.plot(
            model2.transform.x[chan],
            model2.transform.y[chan],
            color="gray",
            linestyle="dotted",
        )

    return plasma, model0, bckc0
