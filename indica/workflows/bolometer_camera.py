import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from indica.models.plasma import example_run as example_plasma
from copy import deepcopy
from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.bolometer_camera import Bolometer


def rz_geometry() -> dict:
    # los_end = np.full((nchannels, 3), 0.0)
    # los_end[:, 0] = 0.17
    # los_end[:, 1] = 0.0
    # los_end[:, 2] = np.linspace(0.6, -0.6, nchannels)
    # los_start = np.array([[1.0, 0, 0]] * los_end.shape[0])
    # origin = los_start
    # direction = los_end - los_start

    geometry = {
        "wide": {
            "nsensors": 12,
            "sensor_width": 5.08,
            "sensor_distance": 5.08,
            "pinhole_width": 1.0,
            "sensor_center": [365.26, -1295.21, 0],
            "pinhole_center": [369.54, -1225.34, 0],
        },
        "last": {
            "nsensors": 12,
            "sensor_width": 5.08,
            "sensor_distance": 5.08,
            "pinhole_width": 2.25,
            "sensor_location": [
                [421.64, -1288.14, 0],
                [426.72, -1288.34, 0],
                [432.79, -1288.54, 0],
                [436.87, -1288.74, 0],
                [442.42, -1288.96, 0],
                [447.50, -1289.16, 0],
                [452.58, -1289.36, 0],
                [457.65, -1289.56, 0],
                [465.71, -1289.88, 0],
                [470.78, -1290.08, 0],
                [475.86, -1290.27, 0],
                [480.93, -1290.47, 0],
            ],
            "pinhole_center": [455.52, -1181.54, 0],
        },
    }
    return geometry


def xy_geometry() -> dict:
    geometry = {
        "wide": {
            "nsensors": 8,
            "sensor_width": 5.08,
            "sensor_distance": 5.08,
            "pinhole_width": 3.0,
            "sensor_center": [365.26, -1295.21, 0],
            "pinhole_center": [374.54, -1225.34, 0],
        },
        "last": {
            "nsensors": 12,
            "sensor_width": 5.08,
            "sensor_distance": 5.08,
            "pinhole_width": 3,
            "sensor_location": [
                [421.64, -1288.14, 0],
                [426.72, -1288.34, 0],
                [432.79, -1288.54, 0],
                [436.87, -1288.74, 0],
                [442.42, -1288.96, 0],
                [447.50, -1289.16, 0],
                [452.58, -1289.36, 0],
                [457.65, -1289.56, 0],
                [465.71, -1289.88, 0],
                [470.78, -1290.08, 0],
                [475.86, -1290.27, 0],
                [480.93, -1290.47, 0],
            ],
            "pinhole_center": [455.52, -1181.54, 0],
        },
    }
    return geometry


def get_geometry(view: str = "xy", option: str = "12_chans", side: int = 0):
    """
    Currently accounting for infinitely small detector and finite pinhole size
    """
    geo: dict
    if view == "xy":
        geo = xy_geometry()[option]
    elif view == "rz":
        geo = rz_geometry()[option]
    else:
        raise ValueError(f"Camera direction {view} not supported")

    if "sensor_location" not in geo.keys():
        sensor_locations = []
        x_shifts = list(
            +(np.arange(geo["nsensors"]) - (geo["nsensors"] - 1) / 2.0)
            * geo["sensor_distance"]
        )
        for x in x_shifts:
            xyz = deepcopy(geo["sensor_center"])
            xyz[0] += x
            sensor_locations.append(xyz)
        geo["sensor_location"] = np.array(sensor_locations)

    if "pinhole_location" not in geo.keys():
        pinhole_location = geo["pinhole_center"]
        pinhole_location[0] += geo["pinhole_width"] / 2.0 * side
        geo["pinhole_location"] = np.array([pinhole_location] * geo["nsensors"])

    geo["pinhole_location"] *= np.full_like(geo["pinhole_location"], 1.0e-3)
    geo["sensor_location"] *= np.full_like(geo["sensor_location"], 1.0e-3)

    return geo


def camera_views(
    view="xy", option: str = "last", side: int = 0,
):

    geometry = get_geometry(view=view, option=option, side=side)

    sensors = geometry["sensor_location"]
    pinhole = geometry["pinhole_location"]
    origin = pinhole
    direction = pinhole - sensors

    _start = origin
    _end = origin + 10 * direction
    direction = _end - _start

    return origin, direction


def viewing_cone(
    view="xy", option: str = "last", plasma=None, pulse: int = 9229,
):
    if plasma is None:
        print("Initializing plasma")
        plasma = example_plasma(pulse=pulse)
        _ = plasma.lz_tot
        _ = plasma.ion_density

    Ne = plasma.electron_density
    _Lz = plasma._lz_tot
    Lz = {}
    for elem in _Lz.keys():
        Lz[elem] = _Lz[elem]
    Nion = plasma._ion_density
    t = plasma.t

    model = []
    bckc = []
    sides = [0, 1, -1]
    for side in sides:
        origin, direction = camera_views(view=view, option=option, side=side)
        model0 = Bolometer(f"bolo_{view}_{option}",)
        los_transform = LineOfSightTransform(
            origin[:, 0],
            origin[:, 1],
            origin[:, 2],
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            machine_dimensions=plasma.machine_dimensions,
            passes=1,
        )
        los_transform.set_equilibrium(plasma.equilibrium)
        model0.set_los_transform(los_transform)
        model.append(model0)
        print(f"Running model for {view}_{option}, side {side}")
        bckc.append(model0(Ne=Ne, Nion=Nion, Lz=Lz, t=t,))

    cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
    tind_plot = [0, int(len(plasma.t) / 2.0), len(plasma.t) - 1]

    # General model plots
    model[0].plot()

    # Cone width
    cols = model[0].los_transform.plot_los()
    for chan in model[0].los_transform.x1:
        xf = np.concatenate(
            [
                model[1].los_transform.x.sel(channel=chan),
                model[2].los_transform.x.sel(channel=chan),
            ]
        )
        yf = np.concatenate(
            [
                model[1].los_transform.y.sel(channel=chan),
                model[2].los_transform.y.sel(channel=chan),
            ]
        )
        plt.fill(xf, yf, color=cols[chan], alpha=0.5)

    # Plot forward model
    plt.figure()
    for i in tind_plot:
        x = bckc[0]["brightness"].channel
        y0 = bckc[0]["brightness"].sel(t=plasma.t[i], method="nearest")
        y1 = bckc[1]["brightness"].sel(t=plasma.t[i], method="nearest")
        y2 = bckc[2]["brightness"].sel(t=plasma.t[i], method="nearest")
        plt.plot(
            x, y0, label=f"t={plasma.t[i].values:1.2f} s", color=cols_time[i],
        )
        plt.fill_between(x, y1, y2, color=cols_time[i], alpha=0.8)
    plt.xlabel("Channel")
    plt.ylabel("Measured brightness (W/m^2)")
    plt.title("Centre (dashed) and side of cones (shaded)")
    plt.legend()

    imp1 = model[1].los_transform.impact_parameter
    imp2 = model[2].los_transform.impact_parameter

    # cone_width = np.sqrt(
    #     (imp1["x"] - imp2["x"]) ** 2
    #     + (imp1["y"] - imp2["y"]) ** 2
    #     + (imp1["z"] - imp2["z"]) ** 2
    # )
    #
    # print(f"Cone width = {cone_width}")

    return plasma, model[0], bckc[0]
