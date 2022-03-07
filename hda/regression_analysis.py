"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.regression_analysis as regr
    regr_data = regr.Database(reload=True)
    regr_data()
    regr.plot(regr_data)

    latest_pulse = ...
    regr.add_pulses(regr_data, latest_pulse)


TODO: add the following quantities
Ti/Te, Vloop, all NBI, li, betaP, geometry (volm, elongation, ..)
"""

from copy import deepcopy
import getpass
import pickle
from scipy import constants

import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray
from xarray import Dataset
import os

from indica.readers import ST40Reader
from indica.readers import ADASReader

# First pulse after Boronisation / GDC
BORONISATION = [8441, 8537, 9903]
GDC = [8545]
# for p in np.arange(8547, 8560+1):
#     GDC.append(p)
GDC = np.array(GDC) - 0.5

plt.ion()


class Database:
    def __init__(
        self,
        pulse_start=8207,
        pulse_end=10046,
        tlim=(-0.03, 0.3),
        dt=0.01,
        overlap=0.5,
        t_max=0.02,
        reload=False,
    ):

        self.info = get_data_info()
        self.data_path = f"{os.path.expanduser('~')}/data/"
        self.data_file = f"{pulse_start}_{pulse_end}_regression_database.pkl"
        self.fig_path = f"{os.path.expanduser('~')}/figures/regr_trends/"
        self.fig_file = f"{pulse_start}_{pulse_end}"
        if reload:

            print(f"\n Reading database for pulse range {pulse_start}-{pulse_end}")

            filename = self.data_path + self.data_file
            print(f"Reloading data from {filename}")
            regr_data = pickle.load(open(filename, "rb"))
            self.pulse_start = regr_data.pulse_start
            self.pulse_end = regr_data.pulse_end
            self.tlim = regr_data.tlim
            self.dt = regr_data.dt
            self.overlap = regr_data.overlap
            self.time = regr_data.time
            self.initialize_structures()
            self.binned = regr_data.binned
            self.max_val = regr_data.max_val
            self.pulses = regr_data.pulses
            self.t_max = regr_data.t_max
        else:

            print(f"\n Building database for pulse range {pulse_start}-{pulse_end}")

            self.pulse_start = pulse_start
            self.pulse_end = pulse_end
            self.tlim = tlim
            self.dt = dt
            self.overlap = overlap
            self.time = np.arange(self.tlim[0], self.tlim[1], dt * overlap)
            self.t_max = t_max

            # Initialize data structure
            self.initialize_structures()

            # Read all data
            binned, max_val, pulses = self.read_data(pulse_start, pulse_end)
            self.binned = binned
            self.max_val = max_val
            self.pulses = pulses

    def __call__(self, *args, **kwargs):
        """
        Apply general filters to data
        """
        # Mult factor for calculation of Te,i(0)
        self.temp_ratio = simulate_xrcs()

        # Apply general filters
        self.binned = general_filters(self.binned)

        # Calculate additional quantities and filter again
        self.binned, self.max_val, self.info = calc_additional_quantities(
            self.binned, self.max_val, self.info, self.temp_ratio
        )
        self.binned = general_filters(self.binned)

        # Calculate max values of binned data
        self.max_val = calc_max_val(
            self.binned, self.max_val, self.info, t_max=self.t_max
        )

        # Apply general filters and calculate additional quantities
        self.max_val = general_filters(self.max_val)

        # Apply defult selection criteria
        self.filtered = apply_selection(self.binned)

    def initialize_structures(self):
        value = DataArray(np.full(self.time.shape, np.nan), coords=[("t", self.time)])
        error = xr.full_like(value, 0)
        gradient = xr.full_like(value, np.nan)
        cumul = xr.full_like(value, np.nan)
        self.empty_binned = Dataset(
            {"value": value, "error": error, "gradient": gradient, "cumul": cumul}
        )

        value = DataArray(0.0)
        error = xr.full_like(value, 0)
        time = xr.full_like(value, np.nan)
        self.empty_max_val = Dataset({"value": value, "error": error, "time": time})

    def read_data(self, pulse_start, pulse_end):
        """
        Read data in time-range of interest

        Parameters
        ----------
        pulse_start
            first pulse in range
        pulse_end
            last pulse in range
        info
            dictionary of data to be read

        Returns
        -------
        Saves binned data and maximum values
        """

        binned = {}
        max_val = {}

        for k in self.info.keys():
            binned[k] = []
            max_val[k] = []

        pulses_all = []
        for pulse in np.arange(pulse_start, pulse_end + 1):
            print(pulse)
            reader = ST40Reader(int(pulse), self.tlim[0], self.tlim[1],)

            time, _ = reader._get_signal("", "pfit", ".post_best.results:time", -1)
            if np.array_equal(time, "FAILED"):
                print("no Ip from PFIT")
                continue
            if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
                print("no Ip from PFIT in time range")
                continue
            ipla, _ = reader._get_signal("", "pfit", ".post_best.results.global:ip", -1)
            if np.array_equal(time, "FAILED"):
                print("no Ip from PFIT")
                continue

            tind = np.where(ipla > 50.0e3)[0]
            if len(tind) < 3:
                print("max(Ip) from PFIT < 50 kA")
                continue

            tlim_pulse = (np.min([np.min(time[tind]), -0.01]), np.max(time[tind]))

            pulses_all.append(np.array([pulse] * len(self.time)))

            for k, v in self.info.items():
                if "uid" not in v.keys():
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                err = None
                data, dims = reader._get_data(v["uid"], v["diag"], v["node"], v["seq"])
                if np.array_equal(data, "FAILED") or np.array_equal(dims[0], "FAILED"):
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                time = dims[0]
                if v["err"] is not None:
                    err, _ = reader._get_data(v["uid"], v["diag"], v["err"], v["seq"])

                if np.min(time) > np.max(self.tlim) or np.max(time) < np.min(self.tlim):
                    print(f"{k} wrong time range")
                    binned[k].append(self.empty_binned)
                    max_val[k].append(self.empty_max_val)
                    continue

                binned_tmp, max_val_tmp = self.bin_in_time(data, time, err,)

                binned[k].append(binned_tmp)
                max_val[k].append(max_val_tmp)

        pulses_all = np.array(pulses_all)
        pulses = np.unique(np.array(pulses_all).flatten())

        for k in binned.keys():
            binned[k] = xr.concat(binned[k], "pulse")
            binned[k] = binned[k].assign_coords({"pulse": pulses})
            max_val[k] = xr.concat(max_val[k], "pulse")
            max_val[k] = max_val[k].assign_coords({"pulse": pulses})

        return binned, max_val, pulses

    def bin_in_time(self, data, time, err=None, sign=+1):
        """
        Bin data in time, calculate maximum value in specified time range,
        create datasets for binned data and maximum values

        Parameters
        ----------
        data
            array of data to be binned
        time
            time axis of data to be binned
        err
            error of data to be binned

        Returns
        -------

        """
        time_new = self.time
        dt = self.dt
        binned = deepcopy(self.empty_binned)
        max_val = deepcopy(self.empty_max_val)

        ifin = np.where(np.isfinite(data))[0]
        if len(ifin) < 2:
            return binned, max_val

        time_binning = time_new[
            np.where((time_new >= np.nanmin(time)) * (time_new <= np.nanmax(time)))
        ]
        for t in time_binning:
            tind = (time >= t - dt / 2.0) * (time < t + dt / 2.0)
            tind_lt = time <= t

            if len(tind) > 0:
                data_tmp = data[tind]
                ifin = np.where(np.isfinite(data_tmp))[0]
                if len(ifin) >= 1:
                    binned.value.loc[dict(t=t)] = np.mean(data_tmp[ifin])
                    binned.cumul.loc[dict(t=t)] = np.sum(data[tind_lt]) * dt
                    binned.error.loc[dict(t=t)] = np.std(data_tmp[ifin])
                    if err is not None:
                        err_tmp = err[tind]
                        binned.error.loc[dict(t=t)] += np.sqrt(
                            np.sum(err_tmp[ifin] ** 2)
                        ) / len(ifin)

        binned.gradient.values = binned.value.differentiate("t")
        binned.cumul.values = xr.where(np.isfinite(binned.value), binned.cumul, np.nan)

        # FInd max value in time range of analysis
        tind = np.where((time >= self.tlim[0]) * (time < self.tlim[1]))[0]
        # Look for maximum in the signal in the positive or negative sides or in its absolute value
        if sign > 0:
            max_ind = np.nanargmax(data[tind])
        elif sign < 0:
            max_ind = np.nanargmin(data[tind])
        else:
            max_ind = np.nanargmin(np.abs(data[tind]))

        max_ind = tind[max_ind]
        max_val.value.values = data[max_ind]
        max_val.time.values = time[max_ind]
        if err is not None:
            max_val.error.values = err[max_ind]

        return binned, max_val


def add_pulses(regr_data, pulse_end, reload=False, write=True):
    """
    Add data from newer pulses to binned dictionary

    Parameters
    ----------
    pulse_end
        Last pulse to include in the analysis
    """
    pulse_start = np.array(regr_data.pulses).max() + 1
    if pulse_end < pulse_start:
        print("Only newer pulses can be added (...for the time being...)")
        return

    old = regr_data
    new = Database(pulse_start, pulse_end, reload=reload)

    # Check data-set consistency
    assert old.info == new.info
    assert old.t_max == new.t_max
    assert old.tlim == new.tlim
    assert all(old.time == new.time)

    # Generate new merged database
    merged = deepcopy(old)
    merged.pulses = np.append(old.pulses, new.pulses)
    merged.fig_file = f"{old.pulse_start}_{new.pulse_end}"
    merged.data_file = f"{merged.fig_file}_regression_database.pkl"

    for k in regr_data.info:
        if k in list(old.binned) and k in list(new.binned):
            merged.binned[k] = [old.binned[k], new.binned[k]]
            merged.binned[k] = xr.concat(merged.binned[k], "pulse")
        if k in list(old.max_val) and k in list(new.max_val):
            merged.max_val[k] = [old.max_val[k], new.max_val[k]]
            merged.max_val[k] = xr.concat(merged.max_val[k], "pulse")

    if write:
        write_to_pickle(merged)
    else:
        return merged


def add_quantities(regr_data, info=None, write=True):
    """
    Add additional quantities to the database

    Temporary: define info structure here
    TODO: move info structure somewhere else so that additional quantities can be added permanently
    TODO: add sign key to all info dictionaries for calculation of max_val
        --> sign = +1 : look for maximum on positive side
        --> sign = -1 : look for maximum on negative sige
        --> sign = 0 : absolute value
    """
    if info is None:
        info = {
            "d_i_6561": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".line_evol.d_i_6561:intensity",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "D I 656.1 nm",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_ar": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_ar",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "Ar lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_b": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_b",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "B lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_c": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_c",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "C lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_h": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_h",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "H lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_he": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_he",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "He lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_li": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_li",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "Li lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_n": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_n",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "N lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
            "sum_o": {
                "uid": "spectrom",
                "diag": "avantes.line_mon",
                "node": ".key_species:sum_o",
                "seq": 0,
                "err": None,
                "max": False,
                "label": "O lines",
                "units": "(a.u.)",
                "const": 1.0,
            },
        }
    #     info = {
    #     "D_I_6561": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.D_I_6561:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "D I 656.1 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "HE_II_6678": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.HE_II_6678:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "He II 467.8 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "B_IV_2826": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.B_IV_2826:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "B IV 282.6 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "B_IV_2822": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.B_IV_2822:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "B IV 282.2 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "B_II_7030": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.B_II_7030:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "B II 703.0 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "B_III_2067": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.B_III_2067:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "B III 206.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "B_III_2066": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.B_III_2066:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "B III 206.6 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_V_2787": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_V_2787:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 278.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_V_2781": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_V_2781:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 278.1 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_VI_3834": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_VI_3834:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 383.4 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_VI_3811": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_VI_3811:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 381.1 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_IV_3386": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_IV_3386:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 338.6 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_IV_3063": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_IV_3063:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 306.3 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_II_4417": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_II_4417:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 441.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_III_3047": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_III_3047:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 304.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "O_III_2984": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.O_III_2984:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "O IV 298.4 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "ar_ii_3729": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.ar_ii_3729:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "Ar II 372.9 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "ar_iii_3024": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.ar_iii_3024:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "Ar III 302.4 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_V_2277": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_V_2277:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C V 227.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_V_2271": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_V_2271:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C V 227.1 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_II_6583": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_II_6583:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C II 658.3 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_II_6578": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_II_6578:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C II 657.8 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_II_5133": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_II_5133:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C II 513.3 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_II_2838": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_II_2838:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C II 283.8 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_III_4647": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_III_4647:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C III 464.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    #     "C_III_2297": {
    #         "uid": "spectrom",
    #         "diag": "avantes.line_mon",
    #         "node": ".line_evol.C_III_2297:intensity",
    #         "seq": 0,
    #         "err": None,
    #         "max": False,
    #         "label": "C III 229.7 nm",
    #         "units": "(a.u.)",
    #         "const": 1.0,
    #     },
    # }

    print(f"New items being added: {list(info)}")

    key = list(regr_data.binned)[0]
    pulses = regr_data.binned[key].pulse.values

    # TODO: check whether the new quantities are already in the self.info dictionary
    info_keys = regr_data.info.keys()
    binned = {}
    max_val = {}
    for k in info.keys():
        assert k not in info_keys
        binned[k] = []
        max_val[k] = []

    for pulse in pulses:
        print(pulse)
        reader = ST40Reader(int(pulse), regr_data.tlim[0], regr_data.tlim[1],)

        for k, v in info.items():
            if "uid" not in v.keys():
                binned[k].append(regr_data.empty_binned)
                max_val[k].append(regr_data.empty_max_val)
                continue

            err = None
            data, dims = reader._get_data(v["uid"], v["diag"], v["node"], v["seq"])
            if np.array_equal(data, "FAILED") or np.array_equal(dims[0], "FAILED"):
                binned[k].append(regr_data.empty_binned)
                max_val[k].append(regr_data.empty_max_val)
                continue

            time = dims[0]
            if v["err"] is not None:
                err, _ = reader._get_data(v["uid"], v["diag"], v["err"], v["seq"])

            if np.min(time) > np.max(regr_data.tlim) or np.max(time) < np.min(
                regr_data.tlim
            ):
                print(f"{k} wrong time range")
                binned[k].append(regr_data.empty_binned)
                max_val[k].append(regr_data.empty_max_val)
                continue

            if "sign" in v.keys():
                sign = v["sign"]
            else:
                sign = +1
            binned_tmp, max_val_tmp = regr_data.bin_in_time(data, time, err, sign=sign)

            binned[k].append(binned_tmp)
            max_val[k].append(max_val_tmp)

    merged = deepcopy(regr_data)
    for k in binned.keys():
        binned[k] = xr.concat(binned[k], "pulse")
        binned[k] = binned[k].assign_coords({"pulse": pulses})
        max_val[k] = xr.concat(max_val[k], "pulse")
        max_val[k] = max_val[k].assign_coords({"pulse": pulses})

        merged.binned[k] = binned[k]
        merged.max_val[k] = max_val[k]

    if write:
        old_file = f"{regr_data.data_path}{regr_data.data_file}"
        backup_file = f"{regr_data.data_path}_{regr_data.data_file}"
        os.rename(old_file, backup_file)
        write_to_pickle(merged)
    else:
        return regr_data


def calc_additional_quantities(binned, max_val, info, temp_ratio):
    # Estimate central temperature from parameterization
    binned = calc_central_temperature(binned, temp_ratio)

    pulses = binned["ipla_efit"].pulse
    time = binned["ipla_efit"].t

    # NBI power
    # TODO: propagation of gradient of V and I...
    info["nbi_power"] = {
        "max": False,
        "label": "P$_{NBI}$",
        "units": "(V * A)",
        "const": 1.0,
    }
    max_val["nbi_power"] = deepcopy(max_val["i_hnbi"] * max_val["v_hnbi"])
    binned["nbi_power"] = deepcopy(binned["i_hnbi"] * binned["v_hnbi"])
    binned["nbi_power"].error.values = np.sqrt(
        (binned["i_hnbi"].error.values * binned["v_hnbi"].value.values) ** 2
        + (binned["i_hnbi"].value.values * binned["v_hnbi"].error.values) ** 2
    )

    # Pulse length
    # Ip > 50 kA & up to end of flat-top
    info["pulse_length"] = {
        "max": False,
        "label": "Pulse length",
        "units": "(s)",
        "const": 1.0,
    }

    cond = {
        "Flattop": {
            "ipla_efit": {"var": "value", "lim": (50.0e3, np.nan)},
            "ipla_efit": {"var": "gradient", "lim": (-1e6, np.nan)},
        }
    }
    filtered = apply_selection(binned, cond, default=False)

    max_val["pulse_length"] = deepcopy(max_val["ipla_efit"])
    max_val["pulse_length"].error.values = np.zeros_like(pulses)
    max_val["pulse_length"].time.values = np.zeros_like(pulses)
    pulse_length = []
    for pulse in pulses:
        tind = np.where(filtered["Flattop"]["selection"].sel(pulse=pulse) == True)[0]
        if len(tind) > 0:
            pulse_length.append(time[tind.max()])
        else:
            pulse_length.append(0)
    max_val["pulse_length"].value.values = np.array(pulse_length)

    # Calculate RIP/IMC and add to values to be plotted
    info["rip_efit"] = {
        "max": True,
        "label": "(R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
        "units": " ",
        "const": 1.0,
    }
    info["rip_imc"] = {
        "max": True,
        "label": "(EFIT R$_{geo}$ I$_P$ @ 10 ms) / (R$_{MC}$ max(I$_{MC})$)",
        "units": " ",
        "const": 1.0,
    }
    binned["rip_efit"] = deepcopy(binned["ipla_efit"])
    binned["rip_efit"].value.values = (
        binned["rmag_efit"].value * binned["ipla_efit"].value.values
    )
    rip_efit = binned["rip_efit"].sel(t=0.015, method="nearest").drop("t")
    max_val["rip_imc"] = deepcopy(max_val["imc"])
    max_val["rip_imc"] = rip_efit / (max_val["imc"] * 0.75 * 22)
    max_val["rip_imc"].error.values = rip_efit.error.values / (
        max_val["imc"].value.values * 0.75 * 22
    )
    # Calculate total gas puff = cumulative gas_puff & its max value
    info["gas_cumulative"] = {
        "max": True,
        "label": "Cumulative gas",
        "units": "(V * s)",
        "const": 1.0,
    }
    binned["gas_cumulative"] = deepcopy(binned["gas_puff"])
    binned["gas_cumulative"].value.values = binned["gas_cumulative"].cumul.values
    max_val["gas_cumulative"] = deepcopy(max_val["gas_puff"])
    pulse_length = xr.where(
        max_val["pulse_length"].value > 0.04, max_val["pulse_length"].value, np.nan
    )
    max_val["gas_cumulative"].value.values = (
        binned["gas_cumulative"].value.max("t").values / pulse_length.values
    )

    # Gas prefill
    info["gas_prefill"] = {
        "max": True,
        "label": "Gas prefill",
        "units": "(V * s)",
        "const": 1.0,
    }
    max_val["gas_prefill"] = deepcopy(max_val["gas_puff"])
    max_val["gas_prefill"].value.values = (
        binned["gas_puff"].cumul.sel(t=0, method="nearest").values
    )

    # Gas for t > 0
    info["gas_fuelling"] = {
        "max": True,
        "label": "Gas fuelling t > 0",
        "units": "(V * s)",
        "const": 1.0,
    }
    max_val["gas_fuelling"] = deepcopy(max_val["gas_puff"])
    max_val["gas_fuelling"].value.values = (
        binned["gas_cumulative"].cumul.max("t")
        - binned["gas_puff"].cumul.sel(t=0, method="nearest")
    ).values

    # Calculate total gas puff = cumulative gas_puff & its max value
    info["total_nbi"] = {
        "max": False,
        "label": "Cumulative NBI power",
        "units": "(kV * A * s)",
        "const": 1.0e-3,
    }
    max_val["total_nbi"] = deepcopy(max_val["nbi_power"])
    binned["total_nbi"] = deepcopy(binned["nbi_power"])
    binned["total_nbi"].value.values = binned["nbi_power"].cumul.values

    info["ti_te_xrcs"] = {
        "max": False,
        "label": "Ti/Te (XRCS)",
        "units": "",
        "const": 1.0,
    }
    max_val["ti_te_xrcs"] = deepcopy(max_val["ti_xrcs"])
    binned["ti_te_xrcs"] = deepcopy(binned["ti_xrcs"])
    binned["ti_te_xrcs"].value.values = (
        binned["ti_xrcs"].value.values / binned["te_xrcs"].value.values
    )
    binned["ti_te_xrcs"].error.values = binned["ti_te_xrcs"].value.values * np.sqrt(
        (binned["ti_xrcs"].error.values / binned["ti_xrcs"].value.values) ** 2
        + (binned["ti_xrcs"].error.values / binned["ti_xrcs"].value.values) ** 2
    )

    info["ne_nirh1_te_xrcs"] = {
        "max": False,
        "label": "Ne (NIRH1)/3. * Te (XRCS)",
        "units": "",
        "const": 1.0,
    }
    max_val["ne_nirh1_te_xrcs"] = deepcopy(max_val["te_xrcs"])
    binned["ne_nirh1_te_xrcs"] = deepcopy(binned["te_xrcs"])
    binned["ne_nirh1_te_xrcs"].value.values = (
        binned["ne_nirh1"].value.values
        / 3.0
        * binned["te_xrcs"].value.values
        * constants.e
    )
    binned["ne_nirh1_te_xrcs"].error.values = binned[
        "ne_nirh1_te_xrcs"
    ].value.values * np.sqrt(
        (binned["ne_nirh1"].error.values / binned["ne_nirh1"].value.values) ** 2
        + (binned["te_xrcs"].error.values / binned["te_xrcs"].value.values) ** 2
    )

    return binned, max_val, info


def general_filters(results):
    """
    Apply general filters to data read e.g. NBI power 0 where not positive
    """
    print("Applying general data filters")
    keys = results.keys()

    # Set all negative values to 0
    neg_to_zero = ["nbi_power"]
    for k in neg_to_zero:
        if k in keys:
            cond = (results[k].value > 0) * np.isfinite(results[k].value)
            results[k] = xr.where(cond, results[k], 0,)

    # Set all negative values to Nan
    neg_to_nan = [
        "te_xrcs",
        "ti_xrcs",
        "ti0",
        "te0",
        "ipla_efit",
        "ipla_pfit",
        "wp_efit",
        "ne_nirh1",
        "ne_smmh1",
        "gas_press",
        "rip_imc",
    ]
    for k in neg_to_nan:
        if k in keys:
            cond = (results[k].value > 0) * (np.isfinite(results[k].value))
            results[k] = xr.where(cond, results[k], np.nan,)

    # Set to Nan if values outside specific ranges
    err_perc_cond = {"var": "error", "lim": (np.nan, 0.2)}
    err_perc_keys = [
        "te_xrcs",
        "ti_xrcs",
        "te0",
        "ti0",
        "brems_pi",
        "brems_mp",
        "h_i_6563",
        "he_ii_4686",
        "b_ii_3451",
        "o_iv_3063",
        "ar_ii_4348",
        "ne_smmh1",
        "wp_efit",
        "rip_pfit",
        "imc",
    ]

    for k in err_perc_keys:
        if k in keys:
            cond = {k: err_perc_cond}
            selection = selection_criteria(results, cond)
            results[k] = xr.where(selection, results[k], np.nan)

    lim_cond = {"var": "value", "lim": (np.nan, 100.0e3)}
    lim_keys = [
        "wp_efit",
    ]
    for k in lim_keys:
        if k in keys:
            cond = {k: lim_cond}
            selection = selection_criteria(results, cond)
            results[k] = xr.where(selection, results[k], np.nan)

    return results


def calc_central_temperature(binned, temp_ratio):
    print("Calculating central temperature from parameterization")

    # Central temperatures from XRCS parametrization
    mult_binned = []
    profs = np.arange(len(temp_ratio))
    for i in range(len(temp_ratio)):
        ratio_tmp = xr.full_like(binned["te_xrcs"].value, np.nan)
        # TODO: DataArray interp crashing if all nans (@ home only)
        for p in binned["te_xrcs"].pulse:
            te_xrcs = binned["te_xrcs"].value.sel(pulse=p)
            if any(np.isfinite(te_xrcs)):
                ratio_tmp.loc[dict(pulse=p)] = np.interp(
                    te_xrcs.values, temp_ratio[i].te_xrcs, temp_ratio[i].values,
                )
        mult_binned.append(ratio_tmp)
    mult_binned = xr.concat(mult_binned, "prof")
    mult_binned = mult_binned.assign_coords({"prof": profs})

    # Binned data
    mult_max = mult_binned.max("prof", skipna=True)
    mult_min = mult_binned.min("prof", skipna=True)
    mult_mean = mult_binned.mean("prof", skipna=True)
    binned["te0"].value.values = (binned["te_xrcs"].value * mult_mean).values
    err = np.abs(binned["te0"].value * mult_max - binned["te0"].value * mult_min)
    binned["te0"].error.values = np.sqrt(
        (binned["te_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values
    binned["ti0"].value.values = (binned["ti_xrcs"].value * mult_mean).values
    err = np.abs(binned["ti0"].value * mult_max - binned["ti0"].value * mult_min)
    binned["ti0"].error.values = np.sqrt(
        (binned["ti_xrcs"].error * mult_mean) ** 2 + err ** 2
    ).values

    return binned


def calc_max_val(binned, max_val, info, t_max=0.02, keys=None):
    """
    Calculate maximum value in a pulse using the binned data

    Parameters
    ----------
    t_max
        Time above which the max search should start

    """
    print("Calculating maximum values from binned data")

    # Calculate max values for those quantities where binned data is to be used
    if keys is None:
        keys = list(binned.keys())
    else:
        if type(keys) != list:
            keys = [keys]

    for k in keys:
        if k not in info.keys():
            print(f"\n Max val: key {k} not in info dictionary...")
            continue

        for p in binned[keys[0]].pulse:
            v = info[k]
            if v["max"] is True:
                continue
            max_search = xr.where(
                binned[k].t > t_max, binned[k].value.sel(pulse=p), np.nan
            )
            if not any(np.isfinite(max_search)):
                max_val[k].value.loc[dict(pulse=p)] = np.nan
                max_val[k].error.loc[dict(pulse=p)] = np.nan
                max_val[k].time.loc[dict(pulse=p)] = np.nan
                continue
            tind = max_search.argmax(dim="t", skipna=True).values
            tmax = binned[k].t[tind]
            max_val[k].time.loc[dict(pulse=p)] = tmax
            max_val[k].value.loc[dict(pulse=p)] = binned[k].value.sel(pulse=p, t=tmax)
            max_val[k].error.loc[dict(pulse=p)] = binned[k].error.sel(pulse=p, t=tmax)

    return max_val


def selection_criteria(binned, cond):
    """
    Find values within specified limits

    Parameters
    ----------
    binned
        Database binned result dictionary
    cond
        Dictionary of database keys with respective limits e.g.
        {"nirh1":{"var":"value", "lim":(0, 2.e19)}}
        where:
        - "nirh1" is the key of results dictionary
        - "var" is variable of the dataset to be used for the selection,
        either "value", "perc_error", "gradient", "norm_gradient"
        - "lim" = 2 element tuple with lower and upper limits

    Returns
    -------
        Boolean Dataarray of the same shape as the binned data with
        items == True if satisfying the selection criteria

    """

    k = list(cond.keys())[0]
    selection = xr.where(xr.ones_like(binned[k].value) == 1, True, False)
    for k, c in cond.items():
        item = binned[k]
        if c["var"] == "error":  # percentage error
            val = np.abs(item["error"] / item["value"])
        else:
            # val = flat(item[c["var"]])
            val = item[c["var"]]

        lim = c["lim"]
        if len(lim) == 1:
            selection *= val == lim
        else:
            if not np.isfinite(lim[0]):
                selection *= val < lim[1]
            elif not np.isfinite(lim[1]):
                selection *= val >= lim[0]
            else:
                selection *= (val >= lim[0]) * (val < lim[1])

    return selection


def flat(data: DataArray):
    return data.values.flatten()


def apply_selection(
    binned, cond=None, default=True,
):
    """
    Apply selection criteria as defined in the cond dictionary

    Parameters
    ----------
    binned
        Database class result dictionary of binned quantities
    cond
        Dictionary of selection criteria (see default defined below)
        Different elements in list give different selection, elements
        in sub-dictionary are applied together (&)
    default
        set selection criteria conditions to default defined below

    Returns
    -------

    """
    # TODO: max_val calculation too time-consuming...is it worth it?
    if default:
        cond = {
            "Ohmic": {"nbi_power": {"var": "value", "lim": (0,)},},
            "NBI": {"nbi_power": {"var": "value", "lim": (20, np.nan)},},
        }
    # "te0": {"var": "error", "lim": (np.nan, 0.2)},
    # "ti0": {"var": "error", "lim": (np.nan, 0.2)},

    # Apply selection criteria
    if cond is not None:
        filtered = deepcopy(cond)
        for kcond, c in cond.items():
            binned_tmp = deepcopy(binned)
            selection_tmp = selection_criteria(binned_tmp, c)
            for kbinned in binned_tmp.keys():
                binned_tmp[kbinned] = xr.where(
                    selection_tmp, binned_tmp[kbinned], np.nan
                )

            pulses = []
            for p in binned_tmp[kbinned].pulse:
                if any(selection_tmp.sel(pulse=p)):
                    pulses.append(p)
            pulses = np.array(pulses)

            filtered[kcond]["binned"] = binned_tmp
            filtered[kcond]["selection"] = selection_tmp
            filtered[kcond]["pulses"] = pulses
    else:
        filtered = {"All": {"selection": None, "binned": binned}}

    return filtered


def plot_time_evol(
    regr_data,
    info,
    to_plot,
    pulse_lim=None,
    tplot=None,
    savefig=False,
    vlines=True,
    fig_path="",
    fig_name="",
):
    if savefig:
        plt.ioff()
    if tplot is None:
        tit_front = "Maximum of "
        tit_end = ""
        result = regr_data.max_val
    else:
        tit_front = ""
        tit_end = f" @ t={tplot:.3f} s"
        result = regr_data.binned
    if pulse_lim is None:
        pulse_lim = (np.min(regr_data.pulses) - 5, np.max(regr_data.pulses) + 5)

    for title, ykey in to_plot.items():
        name = "time_evol"
        plt.figure()
        for i, k in enumerate(ykey):
            name += f"_{k}"
            res = result[k] * info[k]["const"]
            if tplot is not None:
                res = res.sel(t=tplot, method="nearest")

            x = res.pulse
            yval = flat(res.value)
            yerr = flat(res.error)

            if len(ykey) > 1:
                label = info[k]["label"]
            else:
                label = ""

            plt.errorbar(
                x, yval, yerr=yerr, fmt="o", label=label, alpha=0.5,
            )
        plt.xlabel("Pulse #")
        plt.ylabel(info[k]["units"])
        plt.title(f"{tit_front}{title}{tit_end}")
        plt.xlim(pulse_lim)

        if "sign" in info[k].keys():
            sign = info[k]["sign"]
        else:
            sign = +1

        if sign > 0:
            plt.ylim(0,)
        elif sign < 0:
            plt.ylim(top=0)
        else:
            ylim = plt.ylim()

        if len(ykey) > 1:
            plt.legend()
        if vlines:
            add_vlines(BORONISATION)
            add_vlines(GDC, color="r")

        if tplot is None:
            name += "_max"
        else:
            name += f"_t_{tplot:.3f}s"

        if savefig:
            save_figure(fig_path, f"{fig_name}_{name}")
    if savefig:
        plt.ion()

    plt.figure()
    cols = cm.rainbow(np.linspace(0, 1, len(regr_data.pulses)))
    for i, p in enumerate(regr_data.pulses):
        (
            regr_data.binned["ipla_efit"].value.sel(pulse=p)
            * regr_data.info["ipla_efit"]["const"]
        ).plot(color=cols[i], alpha=0.5)

    plt.title(f"Pulse range [{regr_data.pulses.min()}, {regr_data.pulses.max()}]")
    plt.xlabel("time (s)")
    plt.ylabel(
        f"{regr_data.info['ipla_efit']['label']} {regr_data.info['ipla_efit']['units']}"
    )
    plt.ylim(bottom=0)

    plt.figure()
    bvl_ov_ip = regr_data.binned["i_bvl"] / regr_data.binned["ipla_efit"]
    cols = cm.rainbow(np.linspace(0, 1, len(regr_data.pulses)))
    for i, p in enumerate(bvl_ov_ip.pulse):
        bvl_ov_ip.value.sel(pulse=p).plot(color=cols[i], alpha=0.5)

    plt.title(f"Pulse range [{regr_data.pulses.min()}, {regr_data.pulses.max()}]")
    plt.xlabel("time (s)")
    plt.ylabel("I$_{BVL}$/I$_P$")
    plt.ylim(top=0)

    plt.figure()
    time = np.arange(0.01, 0.2, 0.02)
    cols = cm.rainbow(np.linspace(0, 1, len(time)))
    for i, t in enumerate(time):
        bvl_ov_ip.value.sel(t=t, method="nearest").plot(
            color=cols[i], label=f"{t:.2f} s", alpha=0.5, marker="o"
        )

    plt.legend()
    plt.title(f"Time range [{time.min():.2f}, {time.max():.2f}]")
    plt.xlabel("Pulse")
    plt.ylabel("I$_{BVL}$/I$_P$")
    plt.ylim(top=0)


def plot_bivariate(
    filtered, info, to_plot, label=None, savefig=False, fig_path="", fig_name="",
):
    if savefig:
        plt.ioff()

    for title, keys in to_plot.items():
        xkey, ykey = keys
        name = f"{ykey}_vs_{xkey}"
        xinfo = info[xkey]
        yinfo = info[ykey]

        plt.figure()
        for label, data in filtered.items():
            binned = data["binned"]
            x = binned[xkey] * xinfo["const"]
            y = binned[ykey] * yinfo["const"]
            xval = flat(x.value)
            xerr = flat(x.error)
            yval = flat(y.value)
            yerr = flat(y.error)

            plt.errorbar(
                xval, yval, xerr=xerr, yerr=yerr, fmt="o", label=label, alpha=0.5,
            )
            plt.xlabel(xinfo["label"] + " " + xinfo["units"])
            plt.ylabel(yinfo["label"] + " " + yinfo["units"])
            plt.title(title)
        if label is not None:
            plt.legend()

        if savefig:
            save_figure(fig_path, f"{fig_name}_{name}")
    if savefig:
        plt.ion()


def plot_trivariate(
    filtered, info, to_plot, nbins=10, savefig=False, fig_path="", fig_name=""
):
    if savefig:
        plt.ioff()

    for title, keys in to_plot.items():
        xkey, ykey, zkey = keys
        xinfo = info[xkey]
        yinfo = info[ykey]
        zinfo = info[zkey]

        ylim = []
        xlim = []
        for i, label in enumerate(filtered.keys()):
            binned = filtered[label]["binned"]
            x = binned[xkey].value.values.flatten() * xinfo["const"]
            y = binned[ykey].value.values.flatten() * yinfo["const"]
            xlim.append([np.nanmin(x), np.nanmax(x)])
            ylim.append([np.nanmin(y), np.nanmax(y)])
        xlim = np.array(xlim)
        ylim = np.array(ylim)
        xlim = (np.min(xlim) * 0.95, np.max(xlim) * 1.05)
        ylim = (np.min(ylim) * 0.95, np.max(ylim) * 1.05)
        for i, label in enumerate(filtered.keys()):
            plt.figure()
            name = f"{zkey}_vs_{ykey}_and_{xkey}"
            binned = filtered[label]["binned"]
            x = binned[xkey].value.values.flatten() * xinfo["const"]
            y = binned[ykey].value.values.flatten() * yinfo["const"]
            z = binned[zkey].value.values.flatten() * zinfo["const"]
            xerr = binned[xkey].error.values.flatten() * xinfo["const"]
            yerr = binned[ykey].error.values.flatten() * yinfo["const"]

            zhist = np.histogram(z[np.where(np.isfinite(z))[0]], bins=nbins)
            bins = zhist[1]
            bins_str = [f"{b:.1f}" for b in bins]
            nbins = len(bins) - 1

            ind = np.argsort(z)
            cmap = plt.cm.get_cmap("rainbow", nbins)
            sc = plt.scatter(
                x[ind],
                y[ind],
                c=z[ind],
                s=40,
                vmin=np.nanmin(z),
                vmax=np.nanmax(z),
                cmap=cmap,
                marker="o",
                alpha=0.5,
            )
            plt.colorbar(sc, label=f"{zinfo['label']} {zinfo['units']}")

            cols = cm.rainbow(np.linspace(0, 1, nbins))
            for ib in range(nbins):
                ind = np.where((z >= bins[ib]) * (z < bins[ib + 1]))
                plt.errorbar(
                    x[ind],
                    y[ind],
                    xerr=xerr[ind],
                    yerr=yerr[ind],
                    fmt="o",
                    color=cols[ib],
                    label=f"[{bins_str[ib]}, {bins_str[ib+1]}]",
                    alpha=0.2,
                )

            plt.xlabel(xinfo["label"] + " " + xinfo["units"])
            plt.ylabel(yinfo["label"] + " " + yinfo["units"])
            plt.title(f"{label}")
            plt.xlim(xlim)
            plt.ylim(ylim)
            name += f"_{label}"
            if savefig:
                save_figure(fig_path, f"{fig_name}_{name}")
    if savefig:
        plt.ion()


def plot_hist(
    filtered,
    info,
    to_plot,
    tplot=None,
    bins=None,
    savefig=False,
    fig_path="",
    fig_name="",
):
    if savefig:
        plt.ioff()

    for title, key in to_plot.items():
        res = []
        labels = []
        for label, data in filtered.items():
            res_tmp = data["binned"][key] * info[key]["const"]
            if tplot is not None:
                res_tmp = res_tmp.sel(t=tplot, method="nearest")
            res_tmp = flat(res_tmp.value)
            res.append(res_tmp)
            labels.append(label)

        plt.figure()
        name = f"hist_{key}"
        plt.hist(res, bins=bins, label=labels, density=True)
        plt.title(f"{info[key]['label']}")
        plt.xlabel(info[key]["units"])
        plt.legend()
        if tplot is not None:
            name += f"_t_{tplot:.3f}s"
        if savefig:
            save_figure(fig_path, f"{fig_name}_{name}")
    if savefig:
        plt.ion()


def max_ti_pulses(regr_data, savefig=False, plot_results=False):
    cond_general = {
        "nbi_power": {"var": "value", "lim": (20, np.nan)},
        "te0": {"var": "error", "lim": (np.nan, 0.2)},
        "ti0": {"var": "error", "lim": (np.nan, 0.2)},
        "ipla_efit": {"var": "gradient", "lim": (-1.0e6, np.nan)},
    }
    cond_special = deepcopy(cond_general)
    cond_special["ti0"] = {"var": "value", "lim": (1.5e3, np.nan)}
    cond = {
        "NBI": cond_general,
        "NBI & Ti > 1.5 keV": cond_special,
    }
    filtered = apply_selection(regr_data.binned, cond, default=False)
    if plot_results or savefig:
        plot(regr_data, filtered, savefig=savefig)

    print("\n Pulses in selection")
    for k in filtered.keys():
        print(f"\n {k}", filtered[k]["pulses"])

    return filtered


def ip_400_500(regr_data, savefig=False, plot_results=False):
    cond_general = {
        "nbi_power": {"var": "value", "lim": (20, np.nan)},
        "te0": {"var": "error", "lim": (np.nan, 0.2)},
        "ti0": {"var": "error", "lim": (np.nan, 0.2)},
        "ipla_efit": {"var": "gradient", "lim": (-1.0e6, np.nan)},
    }
    cond_special = deepcopy(cond_general)
    cond_special["ipla_efit"] = {"var": "value", "lim": (0.4e6, 0.5e6)}
    cond = {
        "NBI": cond_general,
        "NBI & Ip = [400, 500] kA": cond_special,
    }
    filtered = apply_selection(regr_data.binned, cond, default=False)
    if plot_results or savefig:
        plot(regr_data, filtered, savefig=savefig, plot_time=False)

    print("Pulses in selection")
    for k in filtered.keys():
        print(k, filtered[k]["pulses"])

    return filtered


def plot(
    regr_data, filtered=None, tplot=0.03, savefig=False, plot_time=True, pulse_lim=None
):
    if savefig:
        plt.ioff()

    if filtered is None:
        if hasattr(regr_data, "filtered"):
            filtered = regr_data.filtered

    fig_path = regr_data.fig_path
    fig_name = regr_data.fig_file

    info = regr_data.info

    ###################################
    # Simulated XRCS measurements
    ###################################
    if plot_time == True:
        plt.figure()
        temp_ratio = regr_data.temp_ratio
        for i in range(len(temp_ratio)):
            plt.plot(temp_ratio[i].te0, temp_ratio[i].te_xrcs)

        plt.plot(temp_ratio[0].te0, temp_ratio[0].te0, "--k", label="Central Te")
        plt.legend()
        add_to_plot(
            "T$_e$(0)", "T$_{e,i}$(XRCS)", "XRCS measurement vs. Central Te",
        )
        if savefig:
            save_figure(fig_path, f"{fig_name}_XRCS_Te0_parametrization")

        plt.figure()
        for i in range(len(temp_ratio)):
            el_temp = temp_ratio[i].attrs["el_temp"]
            plt.plot(
                el_temp.rho_poloidal,
                el_temp.sel(t=el_temp.t.mean(), method="nearest") / 1.0e3,
            )

        plt.legend()
        add_to_plot(
            "rho_poloidal", "T$_e$ (keV)", "Temperature profiles",
        )
        if savefig:
            save_figure(fig_path, f"{fig_name}_XRCS_parametrization_temperatures")

        plt.figure()
        for i in range(len(temp_ratio)):
            el_dens = temp_ratio[i].attrs["el_dens"]
            plt.plot(
                el_dens.rho_poloidal,
                el_dens.sel(t=el_dens.t.mean(), method="nearest") / 1.0e3,
            )

        plt.legend()
        add_to_plot(
            "rho_poloidal", "n$_e$ (10$^{19}$)", "Density profiles",
        )
        if savefig:
            save_figure(fig_path, f"{fig_name}_XRCS_parametrization_densities")

        ###################################
        # Time evolution of maximum quantities
        ###################################
        to_plot = {
            "Plasma Current": ("ipla_efit",),
            "Pulse Length": ("pulse_length",),
            "Stored Energy": ("wp_efit",),
            "Electron Temperature": ("te_xrcs", "te0"),
            "Ion Temperature": ("ti_xrcs", "ti0"),
            "Ion/Electron Temperature": ("ti_te_xrcs",),
            "Electron Density": ("ne_nirh1",),
            "Electron Pressure": ("ne_nirh1_te_xrcs",),
            "BVL current": ("i_bvl",),
            "MC Current": ("imc",),
            "Gas pressure": ("gas_press",),
            "Gas prefill": ("gas_prefill",),
            "Cumulative gas puff": ("gas_cumulative",),
            "Cumulative NBI power": ("total_nbi",),
            "Plasma current @ 15 ms / MC current ": ("rip_imc",),
        }
        plot_time_evol(
            regr_data,
            info,
            to_plot,
            savefig=savefig,
            fig_path=fig_path,
            fig_name=fig_name,
            pulse_lim=pulse_lim,
        )

        ###################################
        # Time evolution of quantities at specified tplot
        ###################################
        to_plot = {
            "Bremsstrahlung PI": ("brems_pi",),
            "Bremsstrahlung MP": ("brems_mp",),
            "Plasma Current": ("ipla_efit",),
            "Gas pressure": ("gas_press",),
            "Total gas puff": ("gas_cumulative",),
            "Electron Density": ("ne_nirh1",),
            "BVL current": ("i_bvl",),
            "H visible emission": ("h_sum",),
            "He visible emission": ("he_sum",),
            "B visible emission": ("b_sum",),
            "C visible emission": ("c_sum",),
            "N visible emission": ("n_sum",),
            "O visible emission": ("o_sum",),
            "Ar visible emission": ("ar_sum",),
        }
        # "H-alpha": ("h_i_6563",),
        # "Helium": ("he_ii_4686",),
        # "Boron": ("b_ii_3451",),
        # "Oxygen": ("o_iv_3063",),
        # "Argon": ("ar_ii_4348",),

        plot_time_evol(
            regr_data,
            info,
            to_plot,
            tplot=tplot,
            savefig=savefig,
            fig_path=fig_path,
            fig_name=fig_name,
            pulse_lim=pulse_lim,
        )

    ###################################
    # Bivariate distributions for data-points which satisfy selection criteria
    ###################################
    if filtered is not None:
        to_plot = {
            "T$_e$(0) vs. I$_P$": ("ipla_efit", "te0"),
            "T$_i$(0) vs. I$_P$": ("ipla_efit", "ti0"),
            "T$_i$(0) vs. n$_e$(NIRH1)": ("ne_nirh1", "ti0"),
            "T$_i$(0) vs. gas pressure": ("gas_press", "ti0"),
            "T$_i$(0) vs. Cumulative gas puff": ("gas_cumulative", "ti0"),
            "T$_i$(0) vs. Electron pressure": ("ne_nirh1_te_xrcs", "ti0"),
        }

        plot_bivariate(
            filtered,
            info,
            to_plot,
            savefig=savefig,
            fig_path=fig_path,
            fig_name=fig_name,
        )

        to_plot = {
            "Plasma Current": "ipla_efit",
            "Electron Density": "ne_nirh1",
            "Central Electron Temperature": "te0",
            "Central Ion Temperature": "ti0",
            "Gas Pressure": "gas_press",
            "Cumulative Gas Puff": "gas_cumulative",
        }

        plot_hist(
            filtered,
            info,
            to_plot,
            tplot=None,
            bins=None,
            savefig=savefig,
            fig_path=fig_path,
            fig_name=fig_name,
        )

        to_plot = {
            "Plasma Current": ("te0", "ti0", "ipla_efit"),
            "Electron Density": ("te0", "ti0", "ne_nirh1"),
            "Gas Pressure, Te, Ti": ("te0", "ti0", "gas_press"),
            "Gas Pressure, Ne, Ti": ("ne_nirh1", "ti0", "gas_press"),
            "Cumulative Gas Puff, Te, Ti": ("te0", "ti0", "gas_cumulative"),
            "Cumulative Gas Puff, Ne, Ti": ("ne_nirh1", "ti0", "gas_cumulative"),
        }

        # filtered["All"] = {"selection": None, "binned": regr_data.binned}
        plot_trivariate(
            filtered,
            info,
            to_plot,
            savefig=savefig,
            fig_path=fig_path,
            fig_name=fig_name,
        )

    if savefig:
        plt.ion()


def write_to_csv(regr_data):

    results = regr_data.filtered
    ipla_max = results["ipla_efit_max"].value.values
    ipla_max_time = results["ipla_efit_max"].time.values
    ti_max = results["ti_xrcs_max"].value.values
    ti_max_err = results["ti_xrcs_max"].error.values
    ti_max_time = results["ti_xrcs_max"].time.values
    ratio = regr_data.temp_ratio.sel(
        te_xrcs=results["te_xrcs_max"].value.values, method="nearest"
    ).values
    ti_0 = ti_max * ratio
    ipla_at_max = []
    nbi_at_max = []

    pulses = regr_data.pulses
    for i, p in enumerate(pulses):
        if np.isfinite(ti_max_time[i]):
            ipla = results["ipla_efit"].sel(pulse=p, t=ti_max_time[i]).value.values
            nbi = results["nbi_power"].sel(pulse=p, t=ti_max_time[i]).value.values
        else:
            ipla = np.nan
            nbi = np.nan
        ipla_at_max.append(ipla)
        nbi_at_max.append(nbi)

    ipla_at_max = np.array(ipla_at_max)
    nbi_at_max = np.array(nbi_at_max)

    to_write = {
        "pulse": pulses,
        "Ti max (eV)": ti_max,
        "Error of Ti max (eV)": ti_max_err,
        "Time (s) of Ti max": ti_max_time,
        "Ip (A) at time of max Ti": ipla_at_max,
        "NBI power (W) at time of max Ti": nbi_at_max,
        "Ip max (A)": ipla_max,
        "Time (s) of Ip max": ipla_max_time,
        "Ti (0) (keV)": ti_0,
    }
    df = pd.DataFrame(to_write)
    # df.to_csv()
    return df


def add_vlines(xvalues, color="k"):
    ylim = plt.ylim()
    for b in xvalues:
        plt.vlines(b, ylim[0], ylim[1], linestyles="dashed", colors=color, alpha=0.5)


def save_figure(fig_path, fig_name, orientation="landscape", ext=".jpg"):
    plt.savefig(
        fig_path + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def simulate_xrcs(pickle_file="XRCS_temperature_parametrization.pkl", write=False):
    print("Simulating XRCS measurement for Te(0) re-scaling")

    adasreader = ADASReader()
    xrcs = Spectrometer(
        adasreader, "ar", "16", transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0,
    )

    time = np.linspace(0, 1, 50)
    te_0 = np.linspace(0.5e3, 8.0e3, 50)  # central temperature
    te_sep = 50  # separatrix temperature

    # Test two different profile shapes: flat (Ohmic) and slightly peaked (NBI)
    peaked = profiles_peaked()
    broad = profiles_broad()

    temp = [broad.te, peaked.te]
    dens = [broad.ne, peaked.ne]

    el_temp = deepcopy(temp)
    el_dens = deepcopy(dens)

    for i in range(len(dens)):
        el_dens[i] = el_dens[i].expand_dims({"t": len(time)})
        el_dens[i] = el_dens[i].assign_coords({"t": time})
        el_temp[i] = el_temp[i].expand_dims({"t": len(time)})
        el_temp[i] = el_temp[i].assign_coords({"t": time})
        temp_tmp = deepcopy(el_temp[i])
        for it, t in enumerate(time):
            temp_tmp.loc[dict(t=t)] = scale_prof(temp[i], te_0[it], te_sep).values
        el_temp[i] = temp_tmp

    temp_ratio = []
    for idens in range(len(dens)):
        for itemp in range(len(dens)):
            xrcs.simulate_measurements(el_dens[idens], el_temp[itemp], el_temp[itemp])

            tmp = DataArray(
                te_0 / xrcs.el_temp.values, coords=[("te_xrcs", xrcs.el_temp.values)]
            )
            tmp.attrs = {"el_temp": el_temp[itemp], "el_dens": el_dens[idens]}
            temp_ratio.append(tmp.assign_coords(te0=("te_xrcs", te_0)))

    if write:
        pickle.dump(temp_ratio, open(f"/home/marco.sertoli/data/{pickle_file}", "wb"))

    return temp_ratio


def scale_prof(profile, centre, separatrix):
    scaled = profile - profile.sel(rho_poloidal=1.0)
    scaled /= scaled.sel(rho_poloidal=0.0)
    scaled = scaled * (centre - separatrix) + separatrix

    return scaled


def profiles_broad(te_sep=50):
    rho = np.linspace(0, 1, 100)
    profs = fac.Plasma_profs(rho)

    ne_0 = 5.0e19
    profs.ne = profs.build_density(
        y_0=ne_0,
        y_ped=ne_0,
        x_ped=0.88,
        w_core=4.0,
        w_edge=0.1,
        datatype=("density", "electron"),
    )
    te_0 = 1.0e3
    profs.te = profs.build_temperature(
        y_0=te_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.6,
        w_edge=0.05,
        datatype=("temperature", "electron"),
    )
    profs.te = scale_prof(profs.te, te_0, te_sep)

    ti_0 = 1.0e3
    profs.ti = profs.build_temperature(
        y_0=ti_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.6,
        w_edge=0.05,
        datatype=("temperature", "ion"),
    )
    profs.ti = scale_prof(profs.ti, ti_0, te_sep)

    return profs


def profiles_peaked(te_sep=50):
    rho = np.linspace(0, 1, 100)
    profs = fac.Plasma_profs(rho)

    # slight central peaking and lower separatrix
    ne_0 = 5.0e19
    profs.ne = profs.build_density(
        y_0=ne_0,
        y_ped=ne_0 / 1.25,
        x_ped=0.85,
        w_core=4.0,
        w_edge=0.1,
        datatype=("density", "electron"),
    )
    te_0 = 1.0e3
    profs.te = profs.build_temperature(
        y_0=te_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.4,
        w_edge=0.05,
        datatype=("temperature", "electron"),
    )
    profs.te = scale_prof(profs.te, te_0, te_sep)

    ti_0 = 1.0e3
    profs.ti = profs.build_temperature(
        y_0=ti_0,
        y_ped=50,
        x_ped=1.0,
        w_core=0.4,
        w_edge=0.05,
        datatype=("temperature", "ion"),
    )
    profs.ti = scale_prof(profs.ti, ti_0, te_sep)

    return profs


def calc_mean_std(time, data, tstart, tend, lower=0.0, upper=None, toffset=None):

    avrg = np.nan
    std = 0.0
    offset = 0
    if (
        not np.array_equal(data, "FAILED")
        and (np.size(data) == np.size(time))
        and np.size(data) > 1
    ):
        it = (time >= tstart) * (time <= tend)
        if lower is not None:
            it *= data > lower
        if upper is not None:
            it *= data < upper

        it = np.where(it)[0]
        if len(it) > 1:
            if toffset is not None:
                it_offset = np.where(time <= toffset)[0]
                if len(it_offset) > 1:
                    offset = np.mean(data[it_offset])

            avrg = np.mean(data[it] + offset)
            if len(it) >= 2:
                std = np.std(data[it] + offset)

    return avrg, std


def write_to_pickle(regr_data):
    picklefile = f"{regr_data.data_path}{regr_data.data_file}"
    print(f"Saving regression database to \n {picklefile}")
    pickle.dump(regr_data, open(picklefile, "wb"))


def get_data_info():

    info = {
        "ipla_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".constraints.ip:cvalue",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "I$_P$ EFIT",
            "units": "(MA)",
            "const": 1.0e-6,
        },
        "wp_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".virial:wp",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "W$_P$ EFIT",
            "units": "(kJ)",
            "const": 1.0e-3,
        },
        "q95_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:q95",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "q$_{95}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "volm_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:volm",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "V$_p$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "elon_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:elon",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "Elongation EFIT",
            "units": "",
            "const": 1.0,
        },
        "zmag_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:zmag",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{mag}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "rmag_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:rmag",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{mag}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "rmin_efit": {
            "uid": "",
            "diag": "efit",
            "node": ".global:cr0",
            "seq": 0,
            "err": None,
            "max": True,
            "label": "R$_{min}$ EFIT",
            "units": "",
            "const": 1.0,
        },
        "ipla_pfit": {
            "uid": "",
            "diag": "pfit",
            "node": ".post_best.results.global:ip",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_P$ PFIT",
            "units": "(MA)",
            "const": 1.0e-6,
        },
        "rip_pfit": {
            "uid": "",
            "diag": "pfit",
            "node": ".post_best.results.global:rip",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "W$_P$ EFIT",
            "units": "(kJ)",
            "const": 1.0e-3,
        },
        "vloop": {
            "uid": "",
            "diag": "mag",
            "node": ".floop.l026:v",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "V$_{loop}$ L026",
            "units": "(V)",
            "const": 1.0,
        },
        "ne_nirh1": {
            "uid": "interferom",
            "diag": "nirh1",
            "node": ".line_int:ne",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "n$_{e}$-int NIRH1",
            "units": "($10^{19}$ $m^{-3}$)",
            "const": 1.0e-19,
        },
        "ne_smmh1": {
            "uid": "interferom",
            "diag": "smmh1",
            "node": ".line_int:ne",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "n$_{e}$-int SMMH1",
            "units": "($10^{19}$ $m^{-3}$)",
            "const": 1.0e-19,
        },
        "gas_puff": {
            "uid": "",
            "diag": "gas",
            "node": ".puff_valve:gas_total",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "Total gas",
            "units": "(V)",
            "const": 1.0,
        },
        "gas_press": {
            "uid": "",
            "diag": "mcs",
            "node": ".mcs004:ch019",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "Total Pressure",
            "units": "(a.u.)",
            "const": 1.0e3,
        },
        "imc": {
            "uid": "",
            "diag": "psu",
            "node": ".mc:i",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_{MC}$",
            "units": "(kA)",
            "const": 1.0e-3,
        },
        "itf": {
            "uid": "",
            "diag": "psu",
            "node": ".tf:i",
            "seq": -1,
            "err": None,
            "max": True,
            "label": "I$_{TF}$",
            "units": "(kA)",
            "const": 1.0e-3,
        },
        "brems_pi": {
            "uid": "spectrom",
            "diag": "princeton.passive",
            "node": ".dc:brem_mp",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Bremsstrahlung PI",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "brems_mp": {
            "uid": "spectrom",
            "diag": "lines",
            "node": ".brem_mp1:intensity",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "Bremsstrahlung MP",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "te_xrcs": {
            "uid": "sxr",
            "diag": "xrcs",
            "node": ".te_kw:te",
            "seq": 0,
            "err": ".te_kw:te_err",
            "max": False,
            "label": "T$_e$ XRCS",
            "units": "(keV)",
            "const": 1.0e-3,
        },
        "ti_xrcs": {
            "uid": "sxr",
            "diag": "xrcs",
            "node": ".ti_w:ti",
            "seq": 0,
            "err": ".ti_w:ti_err",
            "max": False,
            "label": "T$_i$ XRCS",
            "units": "(keV)",
            "const": 1.0e-3,
        },
        "i_hnbi": {
            "uid": "raw_nbi",
            "diag": "hnbi1",
            "node": ".hv_ps:i_jema",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "I$_{HNBI}$",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "v_hnbi": {
            "uid": "raw_nbi",
            "diag": "hnbi1",
            "node": ".hv_ps:v_jema",
            "seq": -1,
            "err": None,
            "max": False,
            "label": "V$_{HNBI}$",
            "units": "(a.u.)",
            "const": 1.0e-3,
        },
        "h_i_6563": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.h_i_6563:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "H I 656.3 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "he_ii_4686": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.he_ii_4686:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "He II 468.6 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "b_ii_3451": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.b_ii_3451:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "B II 345.1 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "o_iv_3063": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.o_iv_3063:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "O IV 306.3 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "ar_ii_4348": {
            "uid": "spectrom",
            "diag": "avantes.line_mon",
            "node": ".line_evol.ar_ii_4348:intensity",
            "seq": 0,
            "err": None,
            "max": False,
            "label": "Ar II 434.8 nm",
            "units": "(a.u.)",
            "const": 1.0,
        },
        "i_bvl": {
            "uid": "",
            "diag": "psu",
            "node": ".bvl:i",
            "seq": -1,
            "err": None,
            "max": True,
            "sign": -1,
            "label": "I$_{BVL}$ PSU",
            "units": "(kA)",
            "const": 1.0e-3,
        },
        "te0": {"max": False, "label": "T$_e$(0)", "units": "(keV)", "const": 1.0e-3},
        "ti0": {"max": False, "label": "T$_i$(0)", "units": "(keV)", "const": 1.0e-3},
    }

    return info


def add_to_plot(xlab, ylab, tit, legend=True, vlines=False):
    if vlines:
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(tit)
    if legend:
        plt.legend()
