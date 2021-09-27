from copy import deepcopy

import pickle

from scipy import constants
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import math
import hda.fac_profiles as fac
from hda.forward_models import Spectrometer
import hda.physics as ph
from hda.atomdat import fractional_abundance
from hda.atomdat import get_atomdat
from hda.atomdat import radiated_power

# from hda.hdaadas import ADASReader

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
        self,
        pulse: int = 8256,
        tstart: float = 0.01,
        tend: float = 0.1,
    ):
        """

        Parameters
        ----------
        pulse

        """

        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend

        self.reader = ST40Reader(pulse, tstart - 0.02, tend + 0.02)

        # EFIT
        if pulse == 8303 or pulse == 8322 or pulse == 8323 or pulse == 8324:
            revision = 2
        else:
            revision = 0
        efit = self.reader.get("", "efit", revision)

        if efit is not None:
            efit["revision"] = revision
            self.efit = efit
            equilibrium = Equilibrium(efit)
            self.equilibrium = equilibrium

            self.ipla = efit["ipla"]
            self.R_mag = self.equilibrium.rmag
            self.R_0 = efit["rmag"]
            self.wmhd = efit["wp"]

    def get_xrcs(self):
        xrcs = self.reader.get("sxr", "xrcs", 0)
        if xrcs is not None and hasattr(self, "equilibrium"):
            for k in xrcs.keys():
                xrcs[k].attrs["transform"].set_equilibrium(self.equilibrium)
        self.xrcs = xrcs

    def get_nirh1(self):
        nirh1 = self.reader.get("", "nirh1", 0)
        if nirh1 is not None and hasattr(self, "equilibrium"):
            for k in nirh1.keys():
                nirh1[k].attrs["transform"].set_equilibrium(self.equilibrium)
        self.nirh1 = nirh1

    def get_smmh1(self):
        smmh1 = self.reader.get("", "smmh1", 0)
        if smmh1 is not None and hasattr(self, "equilibrium"):
            for k in smmh1.keys():
                smmh1[k].attrs["transform"].set_equilibrium(self.equilibrium)
        self.smmh1 = smmh1

    def get_other_data(self):
        # Read Vloop and toroidal field
        # TODO temporary MAG reader --> : insert in reader class !!!
        # vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l026:v", 0)
        vloop, vloop_path = self.reader._get_signal("", "mag", ".floop.l016:v", 0)
        vloop_dims, _ = self.reader._get_signal_dims(vloop_path, len(vloop.shape))
        vloop = DataArray(
            vloop,
            dims=("t",),
            coords={"t": vloop_dims[0]},
        )
        vloop = vloop.sel(t=slice(self.reader._tstart, self.reader._tend))
        meta = {
            "datatype": ("voltage", "loop"),
            "error": xr.zeros_like(vloop),
        }
        vloop.attrs = meta
        self.vloop = vloop

        # TODO temporary BT reader --> to be calculated using equilibrium class
        tf_i, tf_i_path = self.reader._get_signal("", "psu", ".tf:i", -1)
        tf_i_dims, _ = self.reader._get_signal_dims(tf_i_path, len(tf_i.shape))
        bt_0 = tf_i * 24.0 * constants.mu_0 / (2 * np.pi * 0.4)
        bt_0 = DataArray(
            bt_0,
            dims=("t",),
            coords={"t": tf_i_dims[0]},
        )
        bt_0 = bt_0.sel(t=slice(self.reader._tstart, self.reader._tend))
        meta = {
            "datatype": ("field", "toroidal"),
            "error": xr.zeros_like(bt_0),
        }
        bt_0.attrs = meta
        self.bt_0 = bt_0
