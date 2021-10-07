"""
Functions to perform tests on XRCS parametrisation
to evaluate central temperatures from measured values
"""

import numpy as np
from xarray import DataArray
import matplotlib.pylab as plt
from indica.equilibrium import Equilibrium
from indica.readers import ST40Reader
from indica.converters import FluxSurfaceCoordinates

plt.ion()

class sxr():
    def __init__(self):
        pulse = 9229
        tstart = 0.01
        tend = 0.1
        self.revision = 0
        self.reader = ST40Reader(pulse, tstart, tend)
        self.get()

    def __call__(self, *args, **kwargs):
        self.remap()
        self.plot()

    def get(self):
        self.sxr = self.reader.get("sxr", "diode_arrays", self.revision)

    def remap(self):
        self.efit = self.reader.get("", "efit", self.revision)
        self.equilibrium = Equilibrium(self.efit)

        self.flux_coords = FluxSurfaceCoordinates("poloidal")
        self.flux_coords.set_equilibrium(self.equilibrium)

        npts = 100
        for k, data in self.sxr.items():
            trans = data.attrs["transform"]
            x1 = data.coords[trans.x1_name]
            x2_arr = np.linspace(0, 1, npts)
            x2 = DataArray(x2_arr, dims=trans.x2_name)
            dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
            data.attrs["x2"] = x2
            data.attrs["dl"] = dl
            data.attrs["x"], data.attrs["y"], data.attrs["z"] = trans.convert_to_xyz(x1, x2, 0)
            data.attrs["R"], data.attrs["z"] = trans.convert_to_Rz(x1, x2, 0)
            rho_equil, _ = self.flux_coords.convert_from_Rz(
                data.attrs["R"], data.attrs["z"]
            )
            rho = rho_equil.interp(t=data.t, method="linear")
            data.attrs["rho"] = rho
            self.sxr[k] = data

    def plot(self):
        import matplotlib.pylab as plt
        th = np.linspace(0, 2*np.pi)
        time = 0.08

        f4 = self.sxr["filter_4"]

        plt.figure()
        rmag = self.equilibrium.rmag.sel(t=time, method="nearest").values
        rmji = self.equilibrium.rmji.sel(t=time, rho_poloidal=1., method="nearest").values
        rmjo = self.equilibrium.rmjo.sel(t=time, rho_poloidal=1., method="nearest").values
        plt.plot(0.17 * np.cos(th), 0.17 * np.sin(th), color="black")
        plt.plot(rmag * np.cos(th), rmag * np.sin(th), label="Rmag", color="red", linestyle="dashed")
        plt.plot(rmji * np.cos(th), rmji * np.sin(th), color="red", label="Plasma")
        plt.plot(rmjo * np.cos(th), rmjo * np.sin(th), color="red")
        for i in range(f4.R.shape[0]):
            plt.plot(f4.x[i,:], f4.y[i, :])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)
        plt.hlines(0, -0.8, 0.8, color="black")
        plt.vlines(0, -0.8, 0.8, color="black")
        plt.title("SXR LOS in (x,y) plane")
        plt.legend()

        plt.figure()
        plt.vlines(rmag, -0.5, 0.5, label="Rmag", color="red", linestyle="dashed")
        for i in range(f4.R.shape[0]):
            plt.plot(f4.R[i,:], f4.z[i, :])
        plt.xlabel("R (m)")
        plt.ylabel("z (m)")
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        plt.vlines(0.17, -0.5, 0.5, label="Inner column", color="black")
        plt.title("SXR LOS in (R,z) plane")
        plt.legend()

        plt.figure()
        for i in range(f4.R.shape[0]):
            plt.plot(f4.rho.sel(t=time, method="nearest")[i, :])
        plt.xlabel("Position along LOS")
        plt.ylabel("rho_poloidal")
        plt.ylim(0, 1.1)
        plt.hlines(1, 0, len(f4.rho[0, 0, :]), linestyles="dashed", label="Separatrix")
        plt.title(f"SXR LOS at t = {int(time*1.e3)} ms")
        plt.legend()
