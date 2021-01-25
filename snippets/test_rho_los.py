import getpass

import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.readers import PPFReader


def los_info(use_equil=False):
    def initialize_reader(shot=90279, t0=45, t1=46):

        reader = PPFReader(shot, t0, t1)
        if reader.requires_authentication:
            user = input("JET username: ")
            password = getpass.getpass("JET password: ")
            assert reader.authenticate(user, password)

        return reader

    # Read data
    reader = initialize_reader()
    data_hrts = reader.get_thomson_scattering("jetppf", "hrts", 0, {"ne", "te"})
    data_equil = reader.get("jetppf", "efit", 0)
    data_sxr = reader.get_radiation("jetppf", "sxr", 0, {"h", "v", "t"})

    # data_hrts = from_pickle("90279_hrts")
    # data_equil = from_pickle("90279_equil")
    # data_sxr = from_pickle("90279_sxr")

    equil = Equilibrium(data_equil)

    time = data_hrts["te"].coords["t"]

    lines_of_sight = data_sxr["v"].attrs["transform"]

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

    # Convert (R, z) to (rho, theta) for each LOS at specified times
    rhos, thetas, t = equil.flux_coords(r, z, time)

    # Plot contour plot flux-surfaces, LOS fan and rho values along specified LOSs
    ilos = -1
    while ilos < nlos:
        ilos += 1
        X, Y = np.meshgrid(equil.rho.coords["R"], equil.rho.coords["z"])
        Z = equil.rho.sel(t=equil.rho.coords["t"].min()).values
        plt.close("all")

        fig1, ax1 = plt.subplots()
        CS = ax1.contour(X, Y, Z)
        ax1.clabel(CS, inline=1, fontsize=10)
        plt.gca().set_aspect("equal", adjustable="box")
        ax1.plot(r.transpose(), z.transpose(), "k", alpha=0.5)
        ax1.plot(r[ilos, :], z[ilos, :], "b")
        ax1.plot(r[-ilos - 1, :], z[-ilos - 1, :], "r")

        fig2, ax2 = plt.subplots()
        ax2.plot(z[ilos, :], rhos[0, ilos, :], "b")
        ax2.plot(z[-ilos - 1, :], rhos[0, -ilos - 1, :], "r")
        plt.ylim(0, 1.2)
        plt.hlines(1, -1.5, 2.0, linestyles="dashed")

        input("Press enter to cycle to next LOS pair")
