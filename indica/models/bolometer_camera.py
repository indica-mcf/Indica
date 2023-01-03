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


class Bolometer(DiagnosticModel):
    """
    Object representing a bolometer camera diagnostic
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_radiation",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "brightness":
                quantity = quant
                self.bckc[quantity] = self.los_integral_radiation
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

    def __call__(
        self,
        Ne: DataArray = None,
        Nimp: DataArray = None,
        Lz: dict = None,
        t: LabeledArray = None,
        calc_rho=False,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Nimp
            Impurity density profiles (dims = "rho", "t", "element")
        Lz
            Cooling factor dictionary of DataArrays of each element to be included
        t
            Time (s) for remapping on equilibrium reconstruction

        Returns
        -------
        Dictionary of back-calculated quantities (as abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Ne = self.plasma.electron_density.interp(t=t)
            _Lz = self.plasma.lz_tot
            Lz = {}
            for elem in _Lz.keys():
                Lz[elem] = _Lz[elem].interp(t=t)
            Nimp = self.plasma.impurity_density.interp(t=t)
        else:
            if Ne is None or Nimp is None or Lz is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Ne = Ne
        self.Nimp = Nimp
        self.Lz = Lz

        elements = Nimp.element.values

        _emission = []
        for ielem, elem in enumerate(elements):
            _emission.append(
                self.Lz[elem].sum("ion_charges") * self.Nimp.sel(element=elem) * self.Ne
            )
        emission = xr.concat(_emission, "element")
        los_integral = self.los_transform.integrate_on_los(
            emission.sum("element"),
            t=t,
            calc_rho=calc_rho,
        )

        self.emission = emission
        self.los_integral_radiation = los_integral

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    diagnostic_name: str = "bolo_Rz",
    origin: LabeledArray = None,
    direction: LabeledArray = None,
    plasma=None,
    plot=False,
):

    if plasma is None:
        plasma = example_plasma()

    # Create new interferometers diagnostics
    if origin is None or direction is None:
        nchannels = 11
        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.17
        los_end[:, 1] = 0.0
        los_end[:, 2] = np.linspace(0.53, -0.53, nchannels)
        los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
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
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    los_transform.set_equilibrium(plasma.equilibrium)
    model = Bolometer(
        diagnostic_name,
    )
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

        # Plot back-calculated profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            bckc["brightness"].sel(t=t, method="nearest").plot(
                label=f"t={t:1.2f} s", color=cols_time[i]
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured brightness (W/m^2)")
        plt.legend()

        # Plot the radiation profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.emission.rho_poloidal,
                model.emission.sum("element").sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("Local radiated power (W/m^3)")
        plt.legend()

    return plasma, model, bckc


def xy_camera_views(
    diagnostic_name="bolo_xy",
    option: int = 0,
    side: int = 0,
    plasma=None,
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

        # sensor_size = [5.08, 5.08, 5.08, 5.08, 5.08]
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
        impact = model0.los_transform.impact_xyz.value
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

    model0.los_transform.plot_los(tplot=np.mean(plasma.t))
    for chan in model0.los_transform.x1:
        plt.plot(
            model1.los_transform.x[chan],
            model1.los_transform.y[chan],
            color="gray",
            linestyle="dashed",
        )
        plt.plot(
            model2.los_transform.x[chan],
            model2.los_transform.y[chan],
            color="gray",
            linestyle="dotted",
        )

    return plasma, model0, bckc0
