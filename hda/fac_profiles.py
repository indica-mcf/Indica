import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray
import hda.physics as ph


class Plasma_profs:
    def __init__(self, rho):
        """
        Make fake plasma profiles for electron density, electron and ion
        temperature, toroidal rotation

        Initialization already calculates default profiles and saves
        them as class attributes

        Results are given as xarray.DataArray

        Parameters
        ----------
        rho
            Radial grid
        """
        self.rho = rho
        self.l_mode()

    def h_mode(
        self,
        te_0=3.5e3,
        ti_0=3.5e3,
        ne_0=6.0e19,
        vrot_0=1.5e5,
        ne_shape=0.8,
        te_shape=0.8,
    ):
        self.te = self.build_temperature(
            y_0=te_0,
            y_ped=ti_0 / 4.0,
            x_ped=0.95,
            w_core=te_shape,
            datatype=("temperature", "electron"),
        )
        self.ti = self.build_temperature(
            y_0=ti_0,
            y_ped=ti_0 / 4.0,
            x_ped=0.95,
            w_core=te_shape,
            datatype=("temperature", "ion"),
        )
        self.ne = self.build_density(
            y_0=ne_0,
            y_ped=ne_0 / 2.0,
            x_ped=0.95,
            w_core=ne_shape,
            datatype=("density", "electron"),
        )
        self.vrot = self.build_rotation(
            y_0=vrot_0,
            y_ped=vrot_0 / 1.5,
            x_ped=0.85,
            w_core=0.5,
            w_edge=0.25,
            datatype=("rotation", "ion"),
        )

    def l_mode(
        self,
        te_0=1.0e3,
        ti_0=1.0e3,
        ne_0=6.0e19,
        vrot_0=0.5e5,
        ne_shape=0.9,
        te_shape=0.9,
    ):
        self.te = self.build_temperature(
            y_0=te_0,
            y_ped=te_0 / 5.0,
            x_ped=0.9,
            w_core=te_shape,
            datatype=("temperature", "electron"),
        )
        self.ti = self.build_temperature(
            y_0=ti_0,
            y_ped=ti_0 / 5.0,
            x_ped=0.9,
            w_core=te_shape,
            datatype=("temperature", "ion"),
        )
        self.ne = self.build_density(
            y_0=ne_0,
            y_ped=ne_0 / 5.0,
            x_ped=0.9,
            w_core=ne_shape,
            datatype=("density", "electron"),
        )
        self.vrot = self.build_rotation(
            y_0=vrot_0,
            y_ped=vrot_0 / 1.5,
            x_ped=0.9,
            w_core=0.5,
            w_edge=0.25,
            datatype=("rotation", "ion"),
        )

    def build_temperature(
        self,
        x_0=0.0,
        y_0=3.0e3,
        y_ped=700.0,
        x_ped=0.85,
        w_core=None,
        w_edge=None,
        datatype=("temperature", "electron"),
        peaked=False,
    ) -> DataArray:

        return build_profile_gauss(
            self.rho,
            x_0,
            y_0,
            y_ped,
            x_ped,
            datatype=datatype,
            w_core=w_core,
            w_edge=w_edge,
            peaked=peaked,
        )

    def build_density(
        self,
        x_0=0.0,
        y_0=6.0e19,
        y_ped=3.0e19,
        x_ped=0.85,
        w_core=None,
        w_edge=None,
        datatype=("density", "electron"),
        peaked=False,
    ) -> DataArray:

        return build_profile_gauss(
            self.rho,
            x_0,
            y_0,
            y_ped,
            x_ped,
            datatype=datatype,
            w_core=w_core,
            w_edge=w_edge,
            peaked=peaked,
        )

    def build_rotation(
        self,
        x_0=0.0,
        y_0=1.5e5,
        y_ped=1.0e5,
        x_ped=0.85,
        w_core=0.5,
        w_edge=0.25,
        datatype=("rotation", "ion"),
        peaked=False,
    ) -> DataArray:

        return build_profile_gauss(
            self.rho,
            x_0,
            y_0,
            y_ped,
            x_ped,
            datatype=datatype,
            w_core=w_core,
            w_edge=w_edge,
            peaked=peaked,
        )

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


def build_profile_gauss(
    x,
    x_0,
    y_0,
    y_ped,
    x_ped,
    x_coord="rho_poloidal",
    datatype=None,
    w_core=None,
    w_edge=None,
    peaked=False,
) -> DataArray:

    x_ped_tmp = x[np.argmin(np.abs(x - x_ped))]
    x_edge = x[np.where(x >= x_ped_tmp)[0]]
    x_core = x[np.where(x <= x_ped_tmp)[0]]

    if not w_edge:
        w_edge = np.abs(1.0 - x_ped_tmp) / 2.0
    if not w_core:
        w_core = 0.4
    if peaked:
        w_core = 0.3

    y_edge = ph.gaussian(x_edge, y_ped, 1.0, x_ped - 0.01, w_edge)
    y_core = ph.gaussian(x_core, y_0, y_ped, x_0, w_core)

    # plt.plot(x_edge, y_edge)
    # plt.plot(x_core, y_core)

    if (np.abs(y_core[-1] - y_edge[0]) / y_edge[0]) > 1.e-2:
        if y_core[-1] < y_core[0]:
            # print("Standard")
            y_core = (y_core - y_core[-1]) / (y_core[0] - y_core[-1]) * (
                y_core[0] - y_edge[0]
            ) + y_edge[0]
        else:
            # print("Hollow")
            y_core = (y_core - y_core[-1]) + y_edge[0]

    y = np.concatenate([y_core[:-1], y_edge])

    # plt.plot(x, y, "*")

    coords = [(x_coord, x)]
    dims = [x_coord]
    value = DataArray(y, coords, dims)
    if datatype is not None:
        attrs = {"datatype": datatype}
        name = datatype[1] + "_" + datatype[0]
        value.name = name
        value.attrs = attrs

    return value


def coord_array(coord_vals, coord_name):
    return DataArray(coord_vals, coords=[(coord_name, coord_vals)])
