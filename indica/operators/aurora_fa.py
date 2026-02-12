from matplotlib import pyplot as plt
from omfit_classes import omfit_eqdsk, omfit_gapy

from indica.configs.operators.aurora_config import AuroraSteadyStateConfig, AuroraConfig
from indica.operators.abstractoperator import Operator
from indica.numpy_typing import LabeledArray


import xarray as xr
import numpy as np
import aurora



class FractionalAbundanceAurora(Operator):
    """
    Calculate fractional abundance for all ionisation charges of a given element using Aurora
    """

    def __init__(
        self,
        steady_state_flag: bool = True,
        aurora_config: AuroraConfig = None,
        steady_state_config: AuroraSteadyStateConfig = None,
        geqdsk = None,
    ):
        self.steady_state_flag = steady_state_flag
        self.aurora_config = aurora_config
        self.steady_state_config = steady_state_config
        self.geqdsk = geqdsk

        self.Ne = None
        self.Te = None
        self.Nh = None

        self.set_geqdsk()
        namelist = aurora.load_default_namelist()
        namelist['imp'] = 'Ar'
        namelist['source_type'] = 'const'
        namelist['source_rate'] = 2e20
        self.namelist = namelist


    def set_geqdsk(self):

        # This should not be here. Give the geqdsk at initialisation, It's not my problem where it comes from

        self.geqdsk = omfit_eqdsk.OMFITgeqdsk("/home/michael.gemmell/python/Aurora/examples/example.gfile")


    def set_profiles(self, Te, Ne, Nh):
        kp = self.namelist['kin_profs']
        kp["Te"]["rhop"] = Te.rhop.values
        kp["ne"]["rhop"] = Ne.rhop.values
        kp["Te"]["vals"] = Te.values
        kp["ne"]["vals"] = Ne.values


    def run_steady_state(self, D_z, V_z, plot: bool = False):
        return self.asim.run_aurora_steady(D_z, V_z, plot=plot)

    def run_time_evolution(self, D_z, V_z, plot: bool = False):
        return self.asim.run_aurora(D_z=D_z, V_z=V_z, plot=plot)

    def __call__(
        self,
        Te: xr.DataArray,
        Ne: xr.DataArray,
        Nh: xr.DataArray,
        D_z: xr.DataArray,
        V_z: xr.DataArray,
        plot: bool = False,
    ) -> xr.DataArray:
        """
        Executes all functions in correct order to calculate the fractional abundance.
        """

        """
        config handling:
        - pass kwargs and construct inside __init__?
        equilibrium and geqdsk interface
        - set on initialisation / No interface just give the operator a geqdsk / interface should be somewhere else (Plasma adjacent??)
        - can't do time dependent without having time in geqdsks
        profile handler
        - take profiles on rhop and set namespace (with unit conversions) and do internal conversion to aurora rhop/rvol grid
        - Some kind of checking of dimensions (time dependent vs steady state)
        
        process outputs:
        - format dict the same as F_z
        
        """

        assert self.geqdsk

        self.set_profiles(Te, Ne, Nh,)
        self.asim = aurora.aurora_sim(namelist=self.namelist, geqdsk=self.geqdsk)

        D_z = 1e4 * np.ones(len(self.asim.rvol_grid))  # cm^2/s
        V_z = -2e2 * np.ones(len(self.asim.rvol_grid))  # cm/s

        if self.steady_state_flag:
            out = self.run_steady_state(D_z=D_z, V_z=V_z, plot=plot)
        else:
            out = self.run_time_evolution(D_z=D_z, V_z=V_z, plot=plot)

        self.bckc = out

        return self.bckc



if __name__ == "__main__":


    from indica.defaults.load_defaults import load_default_objects

    plasma = load_default_objects("st40", "plasma")

    aurora_config = AuroraConfig
    steady_state_config = AuroraSteadyStateConfig()

    operator = FractionalAbundanceAurora(
        aurora_config=aurora_config,
        steady_state_config=steady_state_config,
                                         )

    # rhop = xr.DataArray(np.linspace(0, 1, 50), dims="rhop")
    # ne = xr.DataArray(1e14 * (1 - rhop.values) ** 2 + 1e13, coords={"rhop":rhop})
    # Te = xr.DataArray(2000 * (1 - rhop.values) ** 2 + 50, coords={"rhop":rhop})
    # Nh = 0

    ne = plasma.electron_density
    Te = plasma.electron_temperature
    Nh = plasma.neutral_density

    # D_z = 1e4 * np.ones(50)  # cm^2/s
    # V_z = -2e2 * np.ones(50)  # cm/s


    bckc = operator(Ne=ne, Te=Te, Nh=Nh, D_z=D_z, V_z=V_z, plot=True)
    plt.show(block=True)


