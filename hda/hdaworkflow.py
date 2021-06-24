from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pickle

from hda.hdadata import HDAdata
from hda.hdaplot import HDAplot

import xarray as xr
from xarray import DataArray
from scipy.optimize import least_squares

plt.ion()


class HDArun:
    def __init__(
        self,
        pulse=8256,
        tstart=0.01,
        tend=0.1,
        dt=0.01,
        elements=("h", "c", "ar"),
        ion_conc=(1, 0.03, 0.001),
        ne_shape=1,
        te_shape=0.8,
        regime="l_mode",
        ne_l="smmh1",
    ):
        """From the measured Te and Ti values, search for Te profile that best
        matches total pressure.

        Ti profile shape estimated from passive spectrometer measurements
        and spectral line emission(Te)

        Ne and plasma volume are taken as given, the former from
        interferometer <Ne> measurements and assuming a fixed profile shape,
        the latter from equilibrium reconstruction

        Treat each time-point independently...

        data = experimental data
        bckc = back-calculated values

        """
        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.elements = elements
        self.ion_conc = ion_conc
        self.ne_shape = ne_shape
        self.te_shape = te_shape
        self.regime = regime

        self.data = HDAdata(
            pulse=self.pulse,
            tstart=self.tstart,
            tend=self.tend,
            dt=self.dt,
            elements=self.elements,
            ion_conc=self.ion_conc,
        )

        self.data.build_data(
            ne_shape=self.ne_shape, te_shape=self.te_shape, regime=self.regime, ne_l=ne_l
        )
        self.data.simulate_spectrometers()
        self.data.match_xrcs()

    def __call__(self, use_c5=True, debug=False):

        # self.initialize_bckc()
        # self.recover_temperature(use_c5=use_c5, debug=debug)
        # self.recover_zeff()

        self.initialize_bckc()
        # self.match_xrcs()
        self.recover_density()
        # self.recover_zeff(optimize="density")

        # self.plot()

    def initialize_bckc(self):
        # Initialize back-calculated values
        self.bckc = deepcopy(self.data)

        # Fast ion pressure = 0
        self.bckc.fast_temp.values = xr.zeros_like(self.bckc.el_dens).values
        self.bckc.fast_dens.values = xr.zeros_like(self.bckc.el_dens).values

        # Impurity concentrations = 0, propagate new ion density across all measurements
        self.bckc.ion_conc.loc[dict(element=self.bckc.main_ion)] = 1.0
        self.bckc.ion_dens.loc[
            dict(element=self.bckc.main_ion)
        ] = self.bckc.el_dens.values
        for elem in self.bckc.impurities:
            self.bckc.ion_conc.loc[dict(element=elem)] = 0.0
            self.bckc.ion_dens.loc[dict(element=elem)] = np.zeros_like(
                self.bckc.el_dens.values
            )
        self.bckc.propagate_ion_dens()

    def plot(self, savefig=False, name="", correl="t", plot_spectr=False):
        data = self.data
        bckc = None
        if hasattr(self, "bckc"):
            bckc = self.bckc
        HDAplot(
            data,
            bckc,
            savefig=savefig,
            name=name,
            correl=correl,
            plot_spectr=plot_spectr,
        )

    def recover_temperature(self, use_c5=False, debug=False, niter=3):
        print("\n Re-calculating temperatures to match plasma energy \n")

        data = self.data
        bckc = self.bckc

        nt = len(data.time)

        te_xrcs = data.te_xrcs
        ti_xrcs = data.ti_xrcs
        he_like_bckc = bckc.spectrometers["he_like"]
        # if use_c5 and "passive_c5" in data.spectrometers.keys():
        #     x_ped = 0.85
        #     passive_c5_data = data.spectrometers["passive_c5"]
        #     passive_c5_bckc = bckc.spectrometers["passive_c5"]

        const = DataArray([1.0] * nt, coords=[("t", data.time)])

        # Default profile shapes and initial guess of temperatures
        el_temp = (
            self.bckc.profs.te / self.bckc.profs.te.sel(rho_poloidal=0) * te_xrcs
        ).transpose()
        ion_temp = (
            self.bckc.profs.ti / self.bckc.profs.ti.sel(rho_poloidal=0) * ti_xrcs
        ).transpose()

        for j in range(niter):
            print(f"Iteration {j+1} or {niter}")
            bckc.el_temp *= const

            te_0 = el_temp.sel(rho_poloidal=0)

            # Calculate Ti(0) from He-like spectrometer
            he_like_bckc.simulate_measurements(bckc.el_dens, el_temp, ion_temp)
            ti_0 = (el_temp * ti_xrcs / he_like_bckc.el_temp).sel(
                rho_poloidal=0, method="nearest"
            )
            ion_temp *= ti_0 / ion_temp.sel(rho_poloidal=0)

            # Calculate Ti(pedestal) from passive C5+ spectrometer
            # if use_c5 and "passive_c5" in data.spectrometers.keys():
            #     passive_c5_bckc.simulate_measurements(
            #         bckc.el_dens, bckc.el_temp, ion_temp
            #     )
            #     ti_ped = (
            #         bckc.el_temp * passive_c5_data.ion_temp / passive_c5_bckc.el_temp
            #     ).sel(rho_poloidal=x_ped, method="nearest")

            # Generate profiles with given central and pedestal values
            el_temp *= te_0 / el_temp.sel(rho_poloidal=0)

            if debug:
                plt.figure()
                plt.plot(data.rho, data.el_temp.transpose(), "k*", alpha=0.5)
                plt.plot(data.rho, bckc.el_temp.transpose(), "k", alpha=0.5)
                plt.plot(
                    data.rho,
                    data.ion_temp.sel(element=bckc.main_ion).transpose(),
                    "ro",
                    alpha=0.5,
                )
                plt.plot(
                    data.rho, ion_temp.transpose(), "r--", alpha=0.5,
                )
                input("Press key to continue")

            bckc.el_temp.values = el_temp
            for elem in bckc.elements:
                bckc.ion_temp.loc[dict(element=elem)] = ion_temp

            # Calculate diamagnetic energy and compare with experimental value
            bckc.calc_pressure()
            dwmhd = data.wmhd - bckc.wmhd  # missing pressure in estimated value
            const = 1 + dwmhd / bckc.wmhd

        if debug:
            plt.close("all")

        bckc.propagate_parameters()
        self.bckc = bckc
        return bckc

    def recover_density(self, debug=False, niter=7):
        print("\n Re-calculating density to match plasma energy \n")

        data = self.data
        bckc = self.bckc

        nt = len(data.time)

        dens = self.bckc.profs.ne / self.bckc.profs.ne.sel(rho_poloidal=0)
        # Initial guess of electron density & initialize temperatures
        for t in data.time:
            bckc.el_dens.loc[dict(t=t)] = (dens * 5.0e19).values

        const = DataArray([1.0] * nt, coords=[("t", data.time)])

        # Recover electron density
        for j in range(niter):
            print(f"Iteration {j+1} or {niter}")
            bckc.el_dens *= const
            bckc.calc_main_ion_dens(fast_dens=False)
            bckc.impose_flat_zeff()
            bckc.calc_zeff()

            if debug:
                plt.figure()
                plt.plot(data.rho, data.el_dens.transpose(), "k*", alpha=0.5)
                plt.plot(data.rho, bckc.el_dens.transpose(), "k", alpha=0.5)
                input("Press key to continue")

            # Calculate diamagnetic energy and compare with experimental value
            bckc.calc_pressure()
            dwmhd = data.wmhd - bckc.wmhd  # missing pressure in estimated value
            const.values = 1 + dwmhd / bckc.wmhd

        bckc.propagate_parameters()

        # Recalculate the interferometer measurement
        if hasattr(bckc, "nirh1"):
            bckc.nirh1.values = bckc.calc_ne_los_int("nirh1").values
        if hasattr(bckc, "smmh1"):
            bckc.smmh1.values = bckc.calc_ne_los_int("smmh1").values

        self.bckc = bckc
        if debug:
            plt.close("all")
        return bckc

    def recover_zeff(self, niter=3, optimize="temperature"):
        """

        Returns
        -------

        """

        def residuals(zeff):
            zeff_xr = xr.DataArray(zeff, coords=[("t", bckc.time)])
            bckc.zeff.loc[dict(element=elem_zeff)] = zeff_xr - bckc.zeff.sel(
                element=bckc.main_ion
            )

            ion_dens_tmp = (bckc.zeff.sum("element") - 1) / (
                bckc.meanz.sel(element=elem_zeff) ** 2 / bckc.el_dens
            )
            bckc.ion_dens.loc[dict(element=elem_zeff)] = ion_dens_tmp

            bckc.propagate_ion_dens()

            resid = data.vloop - bckc.vloop

            return resid

        data = self.data
        bckc = self.bckc

        # Initial estimate using Zeff = 1 and all impurity densities to 0
        bckc.ion_conc[self.bckc.main_ion] = 1.0
        for elem in bckc.impurities:
            bckc.ion_conc[elem] = 0.0
            bckc.ion_dens.loc[dict(element=elem)] = (
                bckc.ion_dens.sel(element=elem) * bckc.ion_conc[elem]
            )
        bckc.propagate_ion_dens()

        # Assume Zeff is all due to first light impurity
        elem_zeff = bckc.impurities[0]
        zeff = xr.ones_like(bckc.time).values * 1.5
        bounds = (
            xr.full_like(bckc.time, 1.0).values,
            xr.full_like(bckc.time, 4.0).values,
        )

        for j in range(niter):
            fit = least_squares(residuals, zeff, bounds=bounds, verbose=2)
            if fit.status == -1:
                raise RuntimeError(
                    "Improper input to `least_squares` function when trying to "
                    "fit emissivity to radiation data."
                )
            elif fit.status == 0:
                warnings.warn(
                    f"Attempt to fit emissivity to radiation data at time t={t} "
                    "reached maximum number of function evaluations.",
                    RuntimeWarning,
                )

            zeff_xr = xr.DataArray(fit.x, coords=[("t", bckc.time)])
            bckc.zeff.loc[dict(element=elem_zeff)] = zeff_xr - bckc.zeff.sel(
                element=bckc.main_ion
            )

            # Assume Zeff is all due to first light impurity
            ion_dens_tmp = (bckc.zeff.sum("element") - 1) / (
                bckc.meanz.sel(element=elem_zeff) ** 2 / bckc.el_dens
            )
            bckc.ion_dens.loc[dict(element=elem_zeff)] = ion_dens_tmp

            bckc.propagate_parameters()

            self.bckc = bckc

            if optimize == "temperature":
                self.recover_temperature()
            elif optimize == "density":
                self.recover_density()

    def write_to_pickle(self):

        with open(f"data_{self.pulse}.pkl", "wb") as f:
            pickle.dump(
                self, f,
            )
