import math
from pathlib import Path

import cycler
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.converters import LinesOfSightTransform
from indica.readers import PPFReader
from indica.readers import surf_los


def los_info(use_equil=False):

    """Read data, initialize coordinate object, convert coordinates and
    save all to final DataArray
    """

    # Data
    nt_data = 10
    data = fac_data(nt=nt_data)
    lines_of_sight = data.attrs["transform"]

    # R and z along all the LOS
    nlos = np.size(lines_of_sight.R_start)
    r, z, t = lines_of_sight._convert_to_Rz(
        lines_of_sight.default_x1, lines_of_sight.default_x2, lines_of_sight.default_t
    )
    coords = [
        ("los_index", np.arange(nlos)),
        ("x2", lines_of_sight.default_x2),
    ]
    r = DataArray(r, coords)
    z = DataArray(z, coords)

    # Rhos and theta along LOS on the equilibrium time axis
    r_ax, z_ax = fac_axis(nt_data)
    rhos, theta = fac_rhos(r, z, r_ax, z_ax, nt=nt_data)

    # Make dataarrays for broadcasting
    coords = [
        ("t", data.coords["t"]),
        ("los_index", np.arange(nlos)),
        ("x2", lines_of_sight.default_x2),
    ]
    rhos = DataArray(rhos, coords)
    theta = DataArray(theta, coords)
    r_ax = DataArray(r_ax, [("t", data.coords["t"])])
    z_ax = DataArray(z_ax, [("t", data.coords["t"])])

    # Calculate impact parameter, R and z of min_rho
    min_rho = rhos.min("x2")
    x2_min_rho = rhos.coords["x2"][rhos.argmin("x2")]
    r_min = r.interp(x2=x2_min_rho).transpose()
    z_min = z.interp(x2=x2_min_rho).transpose()

    # Find pinhole position (don't take for granted that it is at x2=0)
    pinhole_x2 = (r.std("los_index") + z.std("los_index")).argmin()
    pinhole_pos = (r[:, pinhole_x2].mean().values, z[:, pinhole_x2].mean().values)

    # Pinhole theta calculated with respect to machine center
    machine_center = (
        np.mean(lines_of_sight._machine_dims[0]),
        np.mean(lines_of_sight._machine_dims[1]),
    )
    pinhole_theta = np.arctan2(
        (pinhole_pos[1] - machine_center[1]), (pinhole_pos[0] - machine_center[0])
    )

    # Sign of rho_min
    # If 45 deg < abs(pinhole_theta) < 135 deg then it's a vertical camera
    if np.abs(pinhole_theta) > math.pi / 4 and np.abs(pinhole_theta) < math.pi * 3 / 4:
        diff = r_min - r_ax
    else:
        diff = z_min - z_ax
    sgn = (diff) / np.abs(diff)

    min_rho *= sgn

    # Plot data to see if it makes sense
    # LOSs on R, z plane
    ilos = 15
    it = 5
    colors = mpl.cm.plasma(np.linspace(0, 1, nt_data))

    plt.figure()
    plt.plot(r.transpose(), z.transpose(), "k", zorder=1)
    (ihfs,) = np.where(sgn[it, :] <= 0)
    (ilfs,) = np.where(sgn[it, :] > 0)
    plt.plot(
        r[ilfs, :].values.transpose(), z[ilfs, :].values.transpose(), "k", zorder=1
    )
    plt.plot(
        r[ihfs, :].values.transpose(),
        z[ihfs, :].values.transpose(),
        "r",
        zorder=1,
        label="HFS los",
    )
    for il in range(nlos):
        plt.scatter(
            r_min[it, il],
            z_min[it, il],
            marker="x",
            color=colors[it],
            zorder=2,
            label="Min-distance",
        )

    mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)
    for i in range(len(colors)):
        plt.scatter(
            r_ax[i],
            z_ax[i],
            edgecolor=colors[i],
            facecolor=colors[i],
            zorder=2,
            label="Axis",
        )
    plt.axis("equal")
    plt.xlabel("R (m)")
    plt.ylabel("z (m)")

    # Plot rhos/min_rho time evolution for one los
    plt.figure()
    colors = mpl.cm.plasma(np.linspace(0, 1, nt_data))
    mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)
    for it in range(nt_data):
        rh = data.attrs["rhos"][it, ilos, :]
        mr = data.attrs["min_rho"][it, ilos]

        p = plt.plot(rh)
        col = p[len(p) - 1].get_color()
        plt.hlines(mr, 0, len(rh), colors=col, linestyles="dashed")

    plt.xlabel("Points along LOS")
    plt.ylabel("Rhos")
    plt.title("Rhos (solid) & impact (dash) for LOS index %s" % ilos)

    return data


def fac_data(nt=10):
    """Fake data since can't connect to server
    nt = number of time-points
    """
    # LOS info
    filepath = Path(surf_los.__file__).parent / "surf_los.dat"
    pulse = 94900
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, "SXR/V"
    )

    # Number of active lines-of-sight
    nlos = np.size(rstart)

    # Coordinate transform object
    transform = LinesOfSightTransform(rstart, zstart, Tstart, rend, zend, Tend)

    # Fake data with some noise (not needed...)
    times = np.linspace(40, 41, nt)
    data = []
    for i in range(nlos):
        data.append(np.linspace(1, 10 * (i + 1), nt))
    data = np.array(data) * (1.0 + (np.random.random_sample(np.shape(data)) - 0.5) / 10)
    data = data.transpose()

    # Coordinate list
    diagnostic_coord = "_".join(["sxr", "t", "coords"])
    coords = [
        ("t", times),
        (diagnostic_coord, np.arange(nlos)),
    ]

    # Metadata
    meta = {
        "datatype": ("luminous_flux", None),
        "transform": transform,
    }

    # Return DataArray
    return DataArray(
        data,
        coords,
        attrs=meta,
    )


def fac_axis(nt):
    """Linearly evolving values axis position for specified number of
    time-points to test changes in inpact parameter
    """
    r_ax = np.linspace(2.8, 3.2, nt)
    z_ax = np.linspace(-0.5, 0.5, nt)

    return r_ax, z_ax


def fac_rhos(r, z, r_ax, z_ax, nt=10):
    """Invent rhos (missing connection with SAL and anyway not sure
    how to do deal with the equilibrium object at the moment...)
    """
    rhos = []
    theta = []
    for rx, zx in zip(r_ax, z_ax):
        min_r = np.sqrt((r - rx) ** 2 + (z - zx) ** 2)
        th = np.arctan2((z - zx), (r - rx))
        rhos.append(np.array([mr / np.max(min_r) for mr in min_r]))
        theta.append(th)

    return np.array(rhos), np.array(theta)


def get_hrts_data():

    reader = PPFReader(90279, 45.0, 50.0)
    data = reader.get_thomson_scattering("jetppf", "hrts", 0, {"ne", "te"})

    return data


def get_sxr_data():

    reader = PPFReader(90279, 45.0, 50.0)
    data = reader.get_radiation("jetppf", "sxr", 0, {"h", "v", "t"})

    return data


def get_efit_data():

    reader = PPFReader(90279, 45.0, 50.0)
    data = reader.get("jetppf", "efit", 0)

    return data
