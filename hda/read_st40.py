from copy import deepcopy

import pickle

from scipy import constants
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import math
import hda.fac_profiles as fac
import hda.physics as ph
from hda.atomdat import fractional_abundance
from hda.atomdat import get_atomdat
from hda.atomdat import radiated_power
from numpy.testing import assert_almost_equal

from indica.readers import ADASReader
from indica.equilibrium import Equilibrium
from indica.readers import ST40Reader
from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time

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

        efit = self.reader.get("", "efit", revision)

        if efit is not None:
            self.data["efit"] = efit
            self.data["ipla"] = efit["ipla"]
            self.data["R_0"] = efit["rmag"]
            self.data["wmhd"] = efit["wp"]

    def get_xrcs(self, revision=0):
        xrcs = self.reader.get("sxr", "xrcs", revision)
        self.data["xrcs"] = xrcs

    def get_nirh1(self, revision=0):
        nirh1 = self.reader.get("", "nirh1", revision)
        value, dims = self.reader._get_data(
            "interferom", "nirh1", ".line_int.ne_bin", nirh1["ne"].attrs["revision"]
        )
        times = dims[0]
        error, _ = self.reader._get_data(
            "interferom", "nirh1", ".line_int.ne_bin_err", nirh1["ne"].attrs["revision"]
        )
        error_sys, _ = self.reader._get_data(
            "interferom", "nirh1", ".line_int.ne_syserr", nirh1["ne"].attrs["revision"]
        )

        transform = nirh1["ne"].attrs["transform"]
        coords = {"t": times}
        dims = ["t"]
        length = len(nirh1["ne"].shape)
        if length > 1:
            dims.append(transform.x1_name)
            coords[transform.x1_name] = np.arange(length)
        else:
            coords[transform.x1_name] = 0
        meta = deepcopy(nirh1["ne"].attrs)

        error = DataArray(
            error, coords, dims,
        ).sel(t=slice(self.reader._tstart, self.reader._tend))
        meta["error"] = error

        error_sys = DataArray(
            error_sys, coords, dims,
        ).sel(t=slice(self.reader._tstart, self.reader._tend))
        meta["error_sys"] = error_sys

        ne_bin = DataArray(
            value, coords, dims, attrs=meta, name="nirh1_ne_bin"
        ).sel(t=slice(self.reader._tstart, self.reader._tend))

        nirh1["ne_bin"] = ne_bin

        self.data["nirh1"] = nirh1

    def get_smmh1(self, revision=0):
        smmh1 = self.reader.get("", "smmh1", revision)
        self.data["smmh1"] = smmh1

    def get_other_data(self):
        # Read Vloop and toroidal field
        # TODO temporary MAG reader --> : insert in reader class !!!
        # vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l026:v", 0)
        vloop, vloop_dims = self.reader._get_data("", "mag", ".floop.l016:v", 0)
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
