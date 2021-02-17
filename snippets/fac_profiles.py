import numpy as np
import matplotlib.pylab as plt
from copy import deepcopy
from xarray import DataArray

class main_plasma_profs:
    def __init__(self):
        """Make fake plasma profiles for electron density, electron and ion temperature, toroidal rotation

        Initialization already calculates default profiles and saves them as class attributes

        Results are saved as xarray.DataArray"""

        self.h_mode()

    def h_mode(self,
               te_0=3.5e3,
               ti_0=3.5e3,
               ne_0=6.e19,
               vrot_0=1.5e5):

        self.te = self.build_temperature(y_0=te_0, y_ped=ti_0/4., x_ped=0.95,
                                         datatype=("temperature", "electrons"))
        self.ti = self.build_temperature(y_0=ti_0, y_ped=ti_0/4., x_ped=0.95,
                                         datatype=("temperature", "ions"))
        self.ne = self.build_density(y_0=ne_0, y_ped=ne_0/2., x_ped=0.95,
                                     datatype=("density", "electrons"))
        self.vrot = self.build_rotation(y_0=vrot_0, y_ped=vrot_0/1.5, x_ped=0.8,
                                        w_edge=0.25,
                                        datatype=("rotation", "ions"))

    def l_mode(self,
               te_0=1.e3,
               ti_0=1.e3,
               ne_0=6.e19,
               vrot_0=0.5e5):

        self.te = self.build_temperature(y_0=te_0, y_ped=te_0/3., x_ped=0.85,
                                         datatype=("temperature", "electrons"))
        self.ti = self.build_temperature(y_0=ti_0, y_ped=ti_0/3., x_ped=0.85,
                                         datatype=("temperature", "ions"))
        self.ne = self.build_density(y_0=ne_0, y_ped=ne_0/5., x_ped=0.85,
                                     datatype=("density", "electrons"))
        self.vrot = self.build_rotation(y_0=vrot_0, y_ped=vrot_0/1.5, x_ped=0.8,
                                        w_edge=0.25,
                                        datatype=("rotation", "ions"))

    def build_temperature(self,
                          y_0=3.e3, y_ped=700., x_ped=0.95,
                          w_core=None, w_edge=None,
                          datatype=("temperature", "electrons"),
                          peaked=False) -> DataArray:

        return self.build_profile_gauss(y_0, y_ped, x_ped, datatype,
                                           w_core=w_core, w_edge=w_edge,
                                           peaked=peaked)

    def build_density(self,
                      y_0=6.e19, y_ped=3.e19, x_ped=0.95,
                      w_core=None, w_edge=None,
                      datatype=("density", "electrons"),
                      peaked=False) -> DataArray:

        return self.build_profile_gauss(y_0, y_ped, x_ped, datatype,
                                           w_core=w_core, w_edge=w_edge,
                                           peaked=peaked)

    def build_rotation(self,
                       y_0=1.5e5, y_ped=1.e5, x_ped=0.8,
                       w_core=None, w_edge=0.25,
                       datatype=("rotation", "ions"),
                       peaked=False) -> DataArray:

        return self.build_profile_gauss(y_0, y_ped, x_ped, datatype,
                                             w_core=w_core, w_edge=w_edge,
                                             peaked=peaked)

    def build_profile_gauss(self,
                            y_0,
                            y_ped,
                            x_ped,
                            datatype,
                            w_core=None,
                            w_edge=None,
                            peaked=False) -> DataArray:

        x_core = np.linspace(0, x_ped, 30)
        x_edge = np.linspace(x_ped, 1.05, 7)
        x = np.concatenate([ x_core[:-1], x_edge ])

        x_edge = x[np.where(x >= x_ped)[0]]
        x_core = x[np.where(x <= x_ped)[0]]

        if not w_edge:
            w_edge = np.abs(1.0 - x_ped)/2.
        if not w_core:
            w_core = 0.4
        if peaked:
            w_core = 0.3

        y_edge = gaussian(x_edge, y_ped, 0.0, x_ped, w_edge)
        y_core = gaussian(x_core, y_0, y_ped, 0.0, w_core)

        if (np.abs(y_core[-1] - y_edge[0])/y_edge[0]) > 1.e-2:
            y_core = ( (y_core - y_core[-1]) /
                       (y_core[0] - y_core[-1]) *
                       (y_core[0] - y_edge[0]) + y_edge[0]
                       )

        y = np.concatenate([
            y_core[:-1],
            y_edge
        ])

        coords = [("rho_poloidal", x)]
        dims = ["rho_poloidal"]
        attrs = {"datatype": datatype}
        name = datatype[0] + "_" + datatype[1]
        value = DataArray(
            y,
            coords,
            dims,
            attrs=attrs)
        value.name = name

        return value

    def plot(self):
        if hasattr(self, "te"):
            plt.figure()
            self.te.plot(marker="o", color="b", alpha=0.5)

        if hasattr(self, "ti"):
            plt.figure()
            self.ti.plot(marker="x", color="r", alpha=0.5)

        if hasattr(self, "ne"):
            plt.figure()
            self.ne.plot(marker="x", color="r", alpha=0.5)

        if hasattr(self, "vrot"):
            plt.figure()
            self.vrot.plot(marker="x", color="r", alpha=0.5)

def coord_array(coord_vals,
                coord_name):
    return DataArray(coord_vals, coords=[(coord_name, coord_vals)])

def gaussian(x, A, B, x_0, w):
    """Build Gaussian with known parameters
    """
    return ((A - B) * np.exp(-(x - x_0) ** 2 / (2 * w ** 2)) + B)
