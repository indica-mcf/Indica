"""
Functions to perform tests on XRCS parametrisation
to evaluate central temperatures from measured values
"""

from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray
import matplotlib.cm as cm

import hda.simple_profiles as profiles
from indica.readers import ADASReader
from hda.diagnostics.spectrometer import Spectrometer

plt.ion()


class xrcs_tests:
    def __init__(self):
        """
        Initialise forward models and profiles
        """
        self.adasreader = ADASReader()

    def he_like_ar(self):
        self.name = "He-like Ar"
        self.xrcs = Spectrometer(
            self.adasreader,
            "ar",
            "16",
            transition="(1)1(1.0)-(1)0(0.0)",
            wavelength=4.0,
        )

    def h_like_ar(self):
        self.name = "H-like Ar"
        # Lyman alpha 1
        self.xrcs = Spectrometer(
            self.adasreader,
            "ar",
            "17",
            transition="(2)1(1.5)-(2)0(0.5)",
            wavelength=3.73003,
        )

        # Lyman alpha 2
        # self.xrcs = Spectrometer(
        #     self.adasreader,
        #     "ar",
        #     "17",
        #     transition="(2)1(0.5)-(2)0(0.5)",
        #     wavelength=3.73518,
        # )

    def he_like_fe(self):
        self.name = "He-like Fe"
        self.xrcs = Spectrometer(
            self.adasreader,
            "fe",
            "24",
            transition="(1)1(1.0)-(1)0(0.0)",
            wavelength=1.8,
        )

    def __call__(
        self,
        plot=False,
    ):

        self.he_like_ar()
        # self.h_like_ar()
        # self.he_like_fe()

        self.set_parameters()
        Ne, Te, Ti, Nh = self.set_profiles(plot=plot)
        result_broad = self.simulate_measurements(Ne, Te, Ti, Nh)

        self.set_parameters(Te_peaking_mult=1.5)
        Ne, Te, Ti, Nh = self.set_profiles(plot=plot)
        result_peaked = self.simulate_measurements(Ne, Te, Ti, Nh)

        return result_broad, result_peaked

    def set_parameters(
        self,
        Ne_0=5.0e19,
        Ne_1=1.0e19,
        Ne_peaking_mult=1.0,
        Te_peaking_mult=1.0,
        Nh_1=[1.0e10, 1.0e15],
        Nh_0_mult=1.0e-1,
        Nh_decay=5,
        wcenter_exp=0.05,
    ):

        self.Ne_0 = Ne_0
        self.Ne_1 = Ne_1
        self.Ne_peaking_mult = Ne_peaking_mult
        self.Te_peaking_mult = Te_peaking_mult
        self.Nh_1 = Nh_1
        self.Nh_0_mult = Nh_0_mult
        self.Nh_decay = Nh_decay
        self.wcenter_exp = wcenter_exp

    def simulate_measurements(self, Ne, Te, Ti, Nh):
        """
        Estimate parametrisation assuming LTE (corona) equilibrium

        Test combinations of electron density and temperature profiles
        and for one of the above, test Ti scan for each Te
        """

        rho = Ne.rho_poloidal

        result = {"Ne": Ne, "Te": Te, "Ti": Ti, "Nh": Nh}

        Te_0 = DataArray(
            np.zeros((Nh.Nh_1.size, Te.Te_0.size, Ti.Ti_0.size)),
            coords=[("Nh_1", Nh.Nh_1), ("Te_0", Te.Te_0), ("Ti_0", Ti.Ti_0)],
            name="Central electron temperature (eV)",
        )
        Ti_0 = deepcopy(Te_0)
        Ti_0.name = "Central ion temperature (eV)"

        Te_xrcs = deepcopy(Te_0)
        Te_xrcs.name = "XRCS electron temperature (eV)"

        Ti_xrcs = deepcopy(Te_0)
        Ti_xrcs.name = "XRCS ion temperature (eV)"

        emiss = DataArray(
            np.zeros((Nh.Nh_1.size, Te.Te_0.size, Ti.Ti_0.size, rho.size)),
            coords=[
                ("Nh_1", Nh.Nh_1.values),
                ("Te_0", Te.Te_0.values),
                ("Ti_0", Ti.Ti_0.values),
                ("rho_poloidal", rho.values),
            ],
            name="Emission shell",
        )

        fz = DataArray(
            np.zeros(
                (
                    Nh.Nh_1.size,
                    Te.Te_0.size,
                    Ti.Ti_0.size,
                    self.xrcs.ion_charges.size,
                    rho.size,
                )
            ),
            coords=[
                ("Nh_1", Nh.Nh_1.values),
                ("Te_0", Te.Te_0.values),
                ("Ti_0", Ti.Ti_0.values),
                ("ion_charges", self.xrcs.ion_charges),
                ("rho_poloidal", rho.values),
            ],
            name="Ionisation balance",
        )

        niter = Nh.Nh_1.size * Te.Te_0.size * Ti.Ti_0.size
        iteration = 0
        for inh, nh1 in enumerate(Nh.Nh_1):
            iteration += inh
            for ite0, te0 in enumerate(Te.Te_0.values):
                iteration += ite0
                show_status(iteration, niter)
                calc_emiss = True
                for iti0, ti0 in enumerate(Ti.Ti_0.values):
                    if iti0 > 0:
                        calc_emiss = False
                    self.xrcs.simulate_measurements(
                        rho,
                        Ne,
                        Te.sel(Te_0=te0),
                        Ti.sel(Te_0=te0, Ti_0=ti0),
                        Nh=Nh.sel(Nh_1=nh1),
                        calc_emiss=calc_emiss,
                    )
                    Te_0.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = te0
                    Te_xrcs.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = self.xrcs.el_temp

                    Ti_0.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = ti0
                    Ti_xrcs.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = self.xrcs.ion_temp

                    fz.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = self.xrcs.fz
                    emiss.loc[dict(Nh_1=nh1, Te_0=te0, Ti_0=ti0)] = self.xrcs.emiss

        Te_ratio = Te_0 / Te_xrcs
        Te_ratio.name = "Te(0) / Te(XRCS)"
        Ti_ratio = Ti_0 / Ti_xrcs
        Ti_ratio.name = "Ti(0) / Ti(XRCS)"

        result["Te_0"] = Te_0
        result["Ti_0"] = Ti_0
        result["Te_xrcs"] = Te_xrcs
        result["Ti_xrcs"] = Ti_xrcs
        result["fz"] = fz
        result["emiss"] = emiss

        return result

    def set_profiles(
        self,
        plot=False,
    ):

        Ne_0 = self.Ne_0
        Ne_1 = self.Ne_1
        Ne_peaking_mult = self.Ne_peaking_mult
        Te_peaking_mult = self.Te_peaking_mult
        Nh_1 = self.Nh_1
        Nh_0_mult = self.Nh_0_mult
        Nh_decay = self.Nh_decay
        wcenter_exp = self.wcenter_exp

        if (wcenter_exp < 0.02) or (wcenter_exp > 0.15):
            print(
                "\n Accepted values for wcenter_exp = [0.02, 0.15] \n Reverting to default = 0.05"
            )
            wcenter_exp = 0.05

        nte = 20
        nti = 5
        Te_0 = np.linspace(0.5e3, 5.0e3, nte)  # central temperature
        Ti_0 = np.linspace(0.5e3, 5.0e3, nti)  # central temperature
        Te_1 = 50.0

        if np.isscalar(Nh_1):
            Nh_1 = [Nh_1]
        Nh_1 = np.array(Nh_1)
        Nh_0 = Nh_1 * Nh_0_mult

        # Set of possible electron density profiles
        wped, wcenter, Ne_peaking = profiles.get_defaults("density")
        Ne_peaking *= Ne_peaking_mult
        Ne, _ = profiles.build_profile(
            Ne_0 / Ne_peaking,
            Ne_1,
            wped=wped,
            wcenter=wcenter,
            peaking=Ne_peaking,
            name="Electron density ($m^{-3}$)",
        )

        # Set of possible electron temperature profile shapes
        wped, wcenter, Te_peaking = profiles.get_defaults("temperature")
        Te_peaking *= Te_peaking_mult
        Te_tmp = []
        for Te0 in Te_0:
            tmp, _ = profiles.build_profile(
                Te0 / Te_peaking,
                Te_1,
                wped=wped,
                wcenter=wcenter,
                peaking=Te_peaking,
                name="Electron temperature (eV)",
            )
            Te_tmp.append(tmp)
        Te = xr.concat(Te_tmp, "Te_0").assign_coords(Te_0=Te_0)

        # Scan central ion temperature for each central electron temperature given
        # one profile shape (to start with)
        Ti = []
        for Te0 in Te_0:
            Ti_Ti_0 = []
            for Ti0 in Ti_0:
                wcenter_ti = wcenter
                peaking2 = Ti0 / Te0
                if peaking2 > 1:
                    Ti0 = Te0
                    wcenter_ti = wcenter - (peaking2 ** wcenter_exp - 1)

                tmp, _ = profiles.build_profile(
                    Ti0 / Te_peaking,
                    Te_1,
                    wped=wped,
                    wcenter=wcenter_ti,
                    peaking=Te_peaking,
                    peaking2=peaking2,
                    name="Ion temperature (eV)",
                )
                Ti_Ti_0.append(tmp)
            Ti.append(xr.concat(Ti_Ti_0, "Ti_0").assign_coords(Ti_0=Ti_0))
        Ti = xr.concat(Ti, "Te_0").assign_coords(Te_0=Te_0)

        # Scan neutral density profile
        rho = Te.rho_poloidal
        Nh = []
        for i in range(len(Nh_1)):
            Nh.append(rho ** Nh_decay * (Nh_1[i] - Nh_0[i]) + Nh_0[i])
        Nh = xr.concat(Nh, "Nh_1").assign_coords(Nh_1=Nh_1)
        Nh.name = "Neutral density ($m^{-3}$)"

        if plot:
            plt.figure()
            Ne.plot()
            plt.title("Electron density")

            plt.figure()
            for Nh_1 in Nh.Nh_1:
                Nh.sel(Nh_1=Nh_1).plot()
            plt.title("Neutral hydrogen density")
            plt.yscale("log")

            plt.figure()
            Te.sel(Te_0=500.0, method="nearest").plot(
                marker="o",
                linestyle="solid",
                color="black",
                label="electrons",
            )
            Te.sel(Te_0=1500.0, method="nearest").plot(
                marker="o", linestyle="dashed", color="red"
            )
            plt.plot(
                Ti.rho_poloidal,
                Ti.sel(Te_0=500.0, method="nearest").transpose(),
                color="black",
                label="ions",
            )
            plt.plot(
                Ti.rho_poloidal,
                Ti.sel(Te_0=1500.0, method="nearest").transpose(),
                color="red",
                linestyle="dashed",
            )
            plt.title("Temperatures")
            plt.legend()

        return Ne, Te, Ti, Nh


def plot(results):
    plt.figure()
    Ne = results["Ne"]
    Ne.plot()
    plt.title("Electron density")

    plt.figure()
    Nh = results["Nh"]
    for Nh_1 in Nh.Nh_1:
        Nh.sel(Nh_1=Nh_1).plot()
    plt.title("Neutral hydrogen density")
    plt.yscale("log")

    Te = results["Te"]
    Ti = results["Ti"]
    plt.figure()
    Te.sel(Te_0=500.0, method="nearest").plot(
        marker="o", linestyle="solid", color="black", label="electrons"
    )
    Te.sel(Te_0=1500.0, method="nearest").plot(
        marker="o", linestyle="dashed", color="red"
    )
    plt.plot(
        Ti.rho_poloidal,
        Ti.sel(Te_0=500.0, method="nearest").transpose(),
        color="black",
        label="ions",
    )
    plt.plot(
        Ti.rho_poloidal,
        Ti.sel(Te_0=1500.0, method="nearest").transpose(),
        color="red",
        linestyle="dashed",
    )
    plt.title("Temperatures")
    plt.legend()

    plt.figure()
    cols = cm.rainbow(np.linspace(0, 1, Te.Te_0.size))
    for i, te0 in enumerate(Te.Te_0):
        Te.sel(Te_0=te0).plot(color=cols[i])
    plt.title("Electron temperature")

    emiss = results["emiss"]
    Nh = results["Nh"]
    lines = ["solid", "dashed", "dotted"]
    plt.figure()
    for inh, nh1 in enumerate(Nh.Nh_1):
        label = f"Nh(1) = {nh1.values:.0E}"
        emiss.sel(Nh_1=nh1, Ti_0=500.0, Te_0=Te.Te_0[0]).plot(
            color=cols[0], linestyle=lines[inh], label=label
        )
        for ite0, te0 in enumerate(Te.Te_0):
            emiss.sel(Nh_1=nh1, Ti_0=500.0, Te_0=te0).plot(
                color=cols[ite0], linestyle=lines[inh]
            )
    plt.title("Emission shell")
    plt.legend()

    Te_0 = results["Te_0"]
    Te_xrcs = results["Te_xrcs"]
    plt.figure()
    plt.plot(Te_xrcs.Te_0, Te_xrcs.Te_0, color="black", linewidth=3)
    for inh, nh1 in enumerate(Nh.Nh_1):
        label = f"Nh(1) = {nh1.values:.0E}"
        Te_xrcs.sel(Nh_1=nh1, Ti_0=500.0).plot(linestyle=lines[inh], label=label)
    plt.title("Measured vs central electron temperature")
    plt.legend()

    Ti_0 = results["Ti_0"]
    Ti_xrcs = results["Ti_xrcs"]
    plt.figure()
    plt.plot(Te_xrcs.Ti_0, Te_xrcs.Ti_0, color="black", linewidth=3)
    for inh, nh1 in enumerate(Nh.Nh_1):
        label = f"Nh(1) = {nh1.values:.0E}"
        Ti_xrcs.sel(Nh_1=nh1, Te_0=Te.Te_0[0]).plot(
            linestyle=lines[inh], color=cols[0], label=label
        )
        for ite, te0 in enumerate(Ti_xrcs.Te_0):
            Ti_xrcs.sel(Nh_1=nh1, Te_0=te0).plot(linestyle=lines[inh], color=cols[ite])
    plt.title("Measured vs central ion temperature")
    plt.legend()


def show_status(iteration, niter):
    if ((niter - iteration) % 5) == 0:
        print(f"{int(100. * iteration / niter)} %")
