from matplotlib import pyplot as plt
from omfit_classes import omfit_eqdsk, omfit_gapy

from indica.configs.operators.aurora_config import AuroraConfig
from indica.operators.abstractoperator import Operator

import xarray as xr
import numpy as np
import aurora


"""
equilibrium and geqdsk interface
- set on initialisation / No interface just give the operator a geqdsk / interface should be somewhere else (Plasma adjacent??)
- can't do time dependent without having time in geqdsks

process outputs:
- format dict the same as F_z
"""


class FractionalAbundanceAurora(Operator):
    """
    Calculate fractional abundance for all ionisation charges of a given element using Aurora
    """

    def __init__(
        self,
        aurora_config: dict = AuroraConfig,
    ):
        self.aurora_config = aurora_config
        self.geqdsk = self.set_geqdsk()


    def set_geqdsk(self):
        # TODO: How to use Equilibrium??
        geqdsk = omfit_eqdsk.OMFITgeqdsk("/home/michael.gemmell/python/Aurora/examples/example.gfile")
        return geqdsk


    def set_kinetic_profiles(self, Te: xr.DataArray, Ne: xr.DataArray, Nh: xr.DataArray, element: str = "Ar"):
        assert self.aurora_config
        kp = self.aurora_config["kin_profs"]
        kp["Te"]["rhop"] = Te.rhop.values
        kp["ne"]["rhop"] = Ne.rhop.values
        kp["n0"]["rhop"] = Nh.rhop.values
        kp["Te"]["times"] = Te.t.values
        kp["ne"]["times"] = Ne.t.values
        kp["n0"]["times"] = Nh.t.values
        kp["Te"]["vals"] = Te.values
        kp["ne"]["vals"] = Ne.values * 1e-6  # m^-3 -> cm^-3
        kp["n0"]["vals"] = Nh.values * 1e-6
        kp['imp'] = element


    def translate_coeff_to_aurora_grid(self, D_z: xr.DataArray, V_z: xr.DataArray):
        # Cast the DataArrays to Aurora rhop grid whilst converting to cm^-2 and cm^-1 units
        _D_z = (D_z.interp(rhop=self.asim.rhop_grid, kwargs={"fill_value": "extrapolate"})).values * 1e4
        _V_z = (V_z.interp(rhop=self.asim.rhop_grid, kwargs={"fill_value": "extrapolate"})).values * 1e2
        return _D_z, _V_z


    def run_steady_state(self, D_z: np.ndarray, V_z: np.ndarray, plot: bool = False):
        raise NotImplementedError
        # return self.asim.run_aurora_steady(D_z, V_z, plot=plot)


    def run_time_evolution(self, D_z: np.ndarray, V_z: np.ndarray, plot: bool = False):
        return self.asim.run_aurora(D_z, V_z, plot=plot)


    def calc_fz(self, nz: np.ndarray, rhop: np.ndarray, time: np.ndarray, zimp: np.ndarray,
                  time_out: np.ndarray, rhop_out: np.ndarray) -> xr.DataArray:
        _Nimp = xr.DataArray(data=nz, coords={"rhop": rhop, "ion_charge": zimp, "t": time, }).transpose("t", "rhop", "ion_charge")
        Nimp = _Nimp.interp(t=time_out, rhop=rhop_out)
        F_z_t = Nimp / Nimp.sum("ion_charge")
        return F_z_t


    def plot_fractional_abundance(self, F_z_t):
        aurora.plot_tools.slider_plot(
                F_z_t.rhop,
                F_z_t.t,
                F_z_t.values.transpose(2, 1, 0),
                xlabel=r"$\rho$ [-]",
                ylabel="time [s]",
                zlabel=r"fractional abundance$ [-]",
                labels=map(str, range(F_z_t.ion_charge.values.shape[0])),
                plot_sum=True,
            )



    def __call__(
        self,
        Te: xr.DataArray,
        Ne: xr.DataArray,
        Nh: xr.DataArray,
        D_z: xr.DataArray,
        V_z: xr.DataArray,
        plot: bool = False,
        element: str = "Ar",
    ) -> xr.DataArray:

        assert self.geqdsk

        self.set_kinetic_profiles(Te, Ne, Nh, element=element)
        self.asim = aurora.aurora_sim(namelist=self.aurora_config, geqdsk=self.geqdsk)
        _D_z, _V_z = self.translate_coeff_to_aurora_grid(D_z, V_z)

        out = self.run_time_evolution(D_z=_D_z, V_z=_V_z, plot=plot)
        self.F_z_t = self.calc_fz(out["nz"], rhop=self.asim.rhop_grid,
                                    time=self.asim.time_grid, zimp=np.arange(self.asim.Z_imp + 1),
                                    time_out=Te.t.values, rhop_out=Te.rhop.values)
        if plot:
            self.plot_fractional_abundance(self.F_z_t)
        return self.F_z_t


if __name__ == "__main__":

    from indica.defaults.load_defaults import load_default_objects

    # equil = load_default_objects("st40", "equilibrium")
    plasma = load_default_objects("st40", "plasma")
    ne = plasma.electron_density
    Te = plasma.electron_temperature
    Nh = plasma.neutral_density
    D_z = xr.DataArray(data=2 * np.ones(50), coords={"rhop": np.linspace(0,1,50)})
    V_z = xr.DataArray(data=-2 * np.ones(50), coords={"rhop": np.linspace(0,1,50)})

    operator = FractionalAbundanceAurora(aurora_config=AuroraConfig)
    bckc = operator(Ne=ne, Te=Te, Nh=Nh, D_z=D_z, V_z=V_z, plot=True)
    plt.show(block=True)



