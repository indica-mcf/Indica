from copy import deepcopy

from scipy import constants
import matplotlib.pylab as plt
import numpy as np
from indica.readers import ST40Reader

import xarray as xr
from xarray import DataArray
from MDSplus.mdsExceptions import TreeNNF

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

    def get_all(
        self,
        efit_rev=0,
        efit_pulse=None,
        xrcs_rev=0,
        nirh1_rev=0,
        smmh1_rev=0,
        brems_rev=-1,
        sxr_rev=0,
        sxr=False,
    ):
        plt.ioff()
        self.get_efit(revision=efit_rev, pulse=efit_pulse)
        self.get_xrcs(revision=xrcs_rev)
        if sxr:
            self.get_sxr(revision=sxr_rev)
        self.get_brems(revision=brems_rev)
        self.get_nirh1(revision=nirh1_rev)
        self.get_smmh1(revision=smmh1_rev)
        self.get_other_data()
        plt.ion()

    def get_sxr(self, revision=0):
        data = self.reader.get("sxr", "diode_arrays", revision, ["filter_4"])
        self.data["sxr"] = data

    def get_efit(self, revision=0, pulse=None):

        if pulse is None:
            pulse = self.pulse
        if (
            pulse == 8303
            or pulse == 8322
            or pulse == 8323
            or pulse == 8324
        ):
            if revision != 2:
                print(f"\nRecommended revision for pulse {pulse} = {2}\n")

        if pulse != self.pulse:
            reader = ST40Reader(pulse, self.tstart, self.tend)
        else:
            reader = self.reader

        data = reader.get("", "efit", revision)

        if len(data) > 0:
            self.data["efit"] = data
            self.data["ipla"] = data["ipla"]
            self.data["R_0"] = data["rmag"]
            self.data["wmhd"] = data["wp"]

    def get_xrcs(self, revision=0):
        data = self.reader.get("sxr", "xrcs", revision)
        if len(data) > 0:
            # Add line ratios to data, propagate uncertainties
            lines = [("int_k", "int_w"), ("int_n3", "int_w"), ("int_n3", "int_tot")]
            for l in lines:
                if l[0] not in data.keys() or l[1] not in data.keys():
                    continue
                ratio_key = f"{l[0]}/{l[1]}"
                num = data[l[0]]
                denom = data[l[1]]
                ratio_tmp = num / denom
                ratio_tmp_err = np.sqrt(
                    (num.attrs["error"] * ratio_tmp / num) ** 2
                    + (denom.attrs["error"] * ratio_tmp / denom) ** 2
                )
                ratio_tmp.attrs["error"] = ratio_tmp_err
                data[ratio_key] = ratio_tmp

            self.data["xrcs"] = data

    def get_princeton(self, revision=0):
        data = self.reader.get("spectrom", "princeton", revision)
        if len(data) > 0:
            self.data["princeton"] = data

    def get_brems(self, revision=-1):
        data = self.reader.get("spectrom", "lines", revision)
        if len(data) > 0:
            self.data["lines"] = data

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

    # def get_sxr(self, revision=0):
    #     data = self.reader.get("sxr", "diode_arrays", revision)
    #     if len(data) > 0:
    #         self.data["sxr"] = data

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
            self.data["R_bt_0"] = DataArray(0.4)

        # TODO temporary NBI power reader
        mds_path = ".NBI.RFX.RUN1:PINJ"
        try:
            prfx = np.array(self.reader._conn_get(mds_path)) * 1.0e6
            prfx_dims, _ = self.reader._get_signal_dims(mds_path, len(prfx.shape))
            prfx = DataArray(prfx, dims=("t",), coords={"t": prfx_dims[0]},)
            prfx = prfx.sel(t=slice(self.reader._tstart, self.reader._tend))
            self.data["rfx"] = {}
            self.data["rfx"]["pin"] = prfx
        except TreeNNF:
            print("no RFX power data in MDS+")

        mds_path = ".NBI.HNBI1.RUN1:PINJ"
        try:
            phnbi1 = np.array(self.reader._conn_get(mds_path)) * 1.0e6
            phnbi1_dims, _ = self.reader._get_signal_dims(mds_path, len(phnbi1.shape))
            phnbi1 = DataArray(phnbi1, dims=("t",), coords={"t": phnbi1_dims[0]},)
            phnbi1 = phnbi1.sel(t=slice(self.reader._tstart, self.reader._tend))
            self.data["hnbi1"] = {}
            self.data["hnbi1"]["pin"] = phnbi1
        except TreeNNF:
            print("no HNBI power data in MDS+")

        # TODO temporary SXR single point reader
        data, dims = self.reader._get_data("sxr", "diode_detr", ".filter_001:signal", 0)
        if not np.array_equal(data, "FAILED"):
            data = DataArray(data, dims=("t",), coords={"t": dims[0]},)
            data = data.sel(t=slice(self.reader._tstart, self.reader._tend))
            self.data["diode_detr"] = {}
            self.data["diode_detr"]["filter_001"] = data

        # TODO temporary SXR single point reader
        data, dims = self.reader._get_data("", "diode_detr", ".filter_001:signal", 0)
        if not np.array_equal(data, "FAILED"):
            data = DataArray(data, dims=("t",), coords={"t": dims[0]},)
            data = data.sel(t=slice(self.reader._tstart, self.reader._tend))
            self.data["diode_detr"] = {}
            self.data["diode_detr"]["filter_001"] = data
