from copy import deepcopy

from scipy import constants
import matplotlib.pylab as plt
import numpy as np
from indica.readers import ST40Reader

import xarray as xr
from xarray import DataArray

plt.ion()

# TODO: add elongation and triangularity in all equations


class ST40data:
    def __init__(
        self, pulse: int = 8256, tstart: float = -0.03, tend: float = 0.3,
    ):
        """
        Read experimental data and save to dictionary

        Parameters
        ----------
        pulse
            Plasma pulse
        tstart
            Start time for reading raw data
        tend
            End time for reading raw data
        """
        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.reader = ST40Reader(pulse, tstart, tend)
        self.data = {}

    def get_all(self, efit_rev=0, xrcs_rev=0, nirh1_rev=0, smmh1_rev=0):
        self.get_efit(revision=efit_rev)
        self.get_xrcs(revision=xrcs_rev)
        self.get_nirh1(revision=nirh1_rev)
        self.get_smmh1(revision=smmh1_rev)
        self.get_other_data()

    def get_efit(self, revision=0):

        if (
            self.pulse == 8303
            or self.pulse == 8322
            or self.pulse == 8323
            or self.pulse == 8324
        ):
            if revision != 2:
                print(f"\nRecommended revision for pulse {self.pulse} = {2}\n")

        data = self.reader.get("", "efit", revision)

        if len(data) > 0:
            self.data["efit"] = data
            self.data["ipla"] = data["ipla"]
            self.data["R_0"] = data["rmag"]
            self.data["wmhd"] = data["wp"]

    def get_xrcs(self, revision=0):
        data = self.reader.get("sxr", "xrcs", revision)
        if len(data) > 0:
            self.data["xrcs"] = data

    def get_nirh1(self, revision=0):
        data = self.reader.get("", "nirh1", revision)
        if len(data) > 0:
            self.data["nirh1"] = data

        data_bin = self.reader.get("", "nirh1_bin", revision)
        if len(data_bin) > 0:
            self.data["nirh1_bin"] = data_bin

    def get_smmh1(self, revision=0):
        data = self.reader.get("", "smmh1", revision)
        if len(data) > 0:
            self.data["smmh1"] = data

    def get_other_data(self):
        # Read Vloop and toroidal field
        # TODO temporary MAG reader --> : insert in reader class !!!
        # vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l026:v", 0)
        vloop, vloop_dims = self.reader._get_data("", "mag", ".floop.l016:v", 0)
        if not np.array_equal(vloop, "FAILED"):
            # vloop_dims, _ = self.reader._get_signal_dims(vloop_path, len(vloop.shape))
            vloop = DataArray(vloop, dims=("t",), coords={"t": vloop_dims[0]},)
            vloop = vloop.sel(t=slice(self.reader._tstart, self.reader._tend))
            meta = {
                "datatype": ("voltage", "loop"),
                "error": xr.zeros_like(vloop),
            }
            vloop.attrs = meta
            self.data["vloop"] = vloop

        # TODO temporary BT reader --> to be calculated using equilibrium class
        tf_i, tf_i_dims = self.reader._get_data("", "psu", ".tf:i", -1)
        if not np.array_equal(tf_i, "FAILED"):
            # tf_i_dims, _ = self.reader._get_signal_dims(tf_i_path, len(tf_i.shape))
            bt_0 = tf_i * 24.0 * constants.mu_0 / (2 * np.pi * 0.4)
            bt_0 = DataArray(bt_0, dims=("t",), coords={"t": tf_i_dims[0]},)
            bt_0 = bt_0.sel(t=slice(self.reader._tstart, self.reader._tend))
            meta = {
                "datatype": ("field", "toroidal"),
                "error": xr.zeros_like(bt_0),
            }
            bt_0.attrs = meta
            self.data["bt_0"] = bt_0
            self.data["R_bt_0"] = 0.4
