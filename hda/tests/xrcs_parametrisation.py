"""
Functions to perform tests on XRCS parametrisation
to evaluate central temperatures from measured values
"""

from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import pickle

from hda.hdaplot import HDAplot
from hda.hdaworkflow import HDArun
from hda.spline_profiles import Plasma_profs
import hda.simple_profiles as profiles
from hda.hdaadas import ADASReader
from hda.forward_models import Spectrometer as Spectrometer_old
from hda.diagnostics.passive_spectrometer import Spectrometer as Spectrometer_new

import xarray as xr
from xarray import DataArray
from scipy.optimize import least_squares

plt.ion()


class xrcs_tests:
    def __init__(self, new=False):
        """
        Initialise forward models and profiles
        """
        adasreader = ADASReader()
        if new:
            self.xrcs = Spectrometer_new(
                adasreader, "ar", "16", transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0,
            )
        else:
            self.xrcs = Spectrometer_old(
                adasreader, "ar", "16", transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0,
            )

        t = np.linspace(0, 1, 50)
        te_0 = np.linspace(0.5e3, 5.0e3, 50)  # central temperature
        te_sep = 50.0
        ne_0 = 5.0e19
        ne_sep = 1.0e19

        # Set of possible electron density profiles
        p = profiles.get_defaults("density")
        el_dens = []
        peaking = p[2] * np.array([1.0, 1.5, 2.0])
        for pe in peaking:
            dens, _ = profiles.build_profile(
                ne_0, ne_sep, wped=p[0], wcenter=p[1], peaking=pe,
            )
            el_dens.append(dens)
        el_dens = xr.concat(el_dens, "peaking").assign_coords(peaking=peaking)
        el_dens.name = "Electron density ($m^{-3}$)"

        # Set of possible electron temperature profile shapes
        p = profiles.get_defaults("temperature")
        el_temp = []
        peaking = p[2] * np.array([1.0, 1.5, 2.0])
        for pe in peaking:
            el_temp_tmp = []
            for te0 in te_0:
                temp, _ = profiles.build_profile(
                    te0 / pe, te_sep, wped=p[0], wcenter=p[1], peaking=pe,
                )
                el_temp_tmp.append(temp)
            el_temp.append(xr.concat(el_temp_tmp, "t").assign_coords(t=t))
        el_temp = xr.concat(el_temp, "peaking").assign_coords(peaking=peaking)
        el_temp.name = "Electron temperature (eV)"

        # Scan central ion temperature for each central electron temperature given
        # one profile shape (to start with)
        ion_temp = []
        for pe in peaking:
            ion_temp_peak = []
            for time in el_temp.t:
                te0 = el_temp.sel(
                    peaking=pe, t=time, rho_poloidal=0, method="nearest"
                ).values

                ion_temp_time = []
                for ti0 in te_0:
                    wcenter = p[1]
                    peaking2 = ti0 / te0
                    if peaking2 > 1:
                        ti0 = te0
                        wcenter = p[1] - (peaking2 ** 0.05 - 1)

                    tempi, _ = profiles.build_profile(
                        ti0 / pe,
                        te_sep,
                        wped=p[0],
                        wcenter=wcenter,
                        peaking=pe,
                        peaking2=peaking2,
                    )
                    ion_temp_time.append(tempi)
                ion_temp_peak.append(
                    xr.concat(ion_temp_time, "ti0").assign_coords(ti0=te_0)
                )
            ion_temp.append(xr.concat(ion_temp_peak, "t").assign_coords(t=el_temp.t))
        ion_temp = xr.concat(ion_temp, "peaking").assign_coords(peaking=peaking)
        ion_temp.name = "Ion temperature (eV)"

        self.el_dens = el_dens
        self.el_temp = el_temp
        self.ion_temp = ion_temp

    def local_ionisation_equilibrium(self):
        """
        Estimate parametrisation assuming LTE (corona) equilibrium

        Test combinations of electron density and temperature profiles
        and for one of the above, test Ti scan for each Te
        """

        # Scan all combinations

        el_temp_ratio = []
        markers = ["o", "*", "x"]
        colors = ["black", "blue", "red"]

        plt.figure()
        for idens, pe in enumerate(self.el_dens.peaking):
            self.el_dens.sel(peaking=pe).plot(marker=markers[idens])

        plt.figure()
        for itemp, pe in enumerate(self.el_temp.peaking):
            self.el_temp.sel(
                peaking=pe, t=np.mean(self.el_temp[itemp].t), method="nearest"
            ).plot(color=colors[itemp])

        # Test effect of all combinations of Ne and Te profile shapes
        # on electron temperature measurement vs central electron temperature
        # TODO: include neutral thermal H density in fractional abundance
        plt.figure()
        for idens, pe_ne in enumerate(self.el_dens.peaking):
            for itemp, pe_te in enumerate(self.el_temp.peaking):
                self.xrcs.simulate_measurements(
                    self.el_dens.sel(peaking=pe_ne),
                    self.el_temp.sel(peaking=pe_te),
                    self.el_temp.sel(peaking=pe_te),
                )

                te0 = self.el_temp.sel(rho_poloidal=0, peaking=pe_te)
                tmp_te = DataArray(
                    (te0 / self.xrcs.el_temp).values,
                    coords=[("te_xrcs", self.xrcs.el_temp.values)],
                )
                tmp_te = tmp_te.assign_coords(te0=("te_xrcs", te0))
                tmp_te.attrs = {
                    "el_temp": self.el_temp.sel(peaking=pe_te),
                    "el_dens": self.el_dens.sel(peaking=pe_ne),
                }
                el_temp_ratio.append(tmp_te)

                plt.plot(
                    tmp_te.te0, tmp_te.te_xrcs, marker=markers[idens], color=colors[itemp]
                )
        plt.plot(te0, te0, "--k")
        plt.xlabel("Central electron temperature (eV)")
        plt.ylabel("Measured electron temperature (eV)")

        # Test ion temperature measurement for different Ti(0)/Te(0) and
        # different electron temperature profile shapes

        # idens = 0
        # for t in self.el_temp.t:
        #     for iti in range(len(self.ion_temp.ti0)):
        #         self.xrcs.simulate_measurements(
        #             self.el_dens[idens], self.el_temp[ite], self.el_temp[ite]
        #         )
        #
        #         te0 = self.el_temp[ite].sel(rho_poloidal=0)
        #         tmp_te = DataArray(
        #             (te0 / self.xrcs.el_temp).values,
        #             coords=[("te_xrcs", self.xrcs.el_temp.values)],
        #         )
        #         tmp_te = tmp_te.assign_coords(te0=("te_xrcs", te0))
        #         tmp_te.attrs = {
        #             "el_temp": self.el_temp[ite],
        #             "el_dens": self.el_dens[idens],
        #         }
        #         el_temp_ratio.append(tmp_te)
        #
        #         plt.plot(
        #             tmp_te.te0, tmp_te.te_xrcs, marker=markers[idens], color=colors[ite]
        #         )

        # return el_temp_ratio
