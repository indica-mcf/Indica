from copy import deepcopy

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.equilibrium import Equilibrium
from indica.numpy_typing import RevisionLike
from indica.readers import ST40Reader
from indica.utilities import print_like

REVISIONS = {
    "efit": 0,
    "brems": -1,
    "halpha": -1,
    "nirh1": 0,
    "smmh1": 0,
    "cxff_pi": 0,
    "cxff_tws_c": 0,
    "cxqf_tws_c": 0,
    "sxr_camera_4": 0,
    "sxr_diode_1": 0,
    "xrcs": 0,
    "ts": 0,
}

FILTER_LIMITS = {
    "cxff_pi": (0, np.inf),
    "cxff_tws_c": (0, np.inf),
    "cxqf_tws_c": (0, np.inf),
    "xrcs": (0, np.inf),
    "brems": (0, np.inf),
    "halpha": (0, np.inf),
    "sxr_diode_1": (0, np.inf),
    "sxr_camera_4": (0, np.inf),
    "ts": (0, np.inf),
}

LINESTYLES = {
    "ts": "solid",
    "cxff_pi": "solid",
    "cxff_tws_c": "dashed",
    "cxqf_tws_c": "dotted",
}
MARKERS = {"ts": "o", "cxff_pi": "s", "cxff_tws_c": "*", "cxqf_tws_c": "x"}
YLABELS = {
    "te": "Te (eV)",
    "ne": "Ne (m$^{-3}$)",
    "ti": "Ti (eV)",
    "vtor": "Vtor (m/s)",
    "chi2": r"$\chi^2$",
}
XLABELS = {"rho": "Rho-poloidal", "R": "R (m)"}


class ReadST40:
    def __init__(
        self,
        pulse: int,
        tstart: float = 0.02,
        tend: float = 0.2,
        dt: float = 0.01,
        tree="ST40",
    ):
        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        self.reader = ST40Reader(pulse, tstart - 0.02, tend + 0.02, tree=tree)

        self.equilibrium: Equilibrium
        self.raw_data: dict = {}
        self.binned_data: dict = {}
        self.transforms: dict = {}

    def reset_data(self):
        self.raw_data = {}
        self.binned_data = {}

    def get_equilibrium(
        self,
        instrument: str = "efit",
        revision: RevisionLike = 0,
        R_shift: float = 0.0,
        z_shift: float = 0.0,
    ):
        if instrument not in self.raw_data:
            equilibrium_data = self.get_raw_data("", instrument, revision)
            equilibrium = Equilibrium(
                equilibrium_data, R_shift=R_shift, z_shift=z_shift
            )
            self.equilibrium = equilibrium

    def get_raw_data(self, uid: str, instrument: str, revision: RevisionLike = 0):
        data = self.reader.get(uid, instrument, revision)
        if hasattr(self, "equilibrium"):
            for quant in data.keys():
                if "transform" not in data[quant].attrs:
                    continue

                transform = data[quant].transform
                if hasattr(transform, "set_equilibrium"):
                    transform.set_equilibrium(self.equilibrium)
                self.transforms[instrument] = transform
        self.raw_data[instrument] = data

        return data

    def bin_data_in_time(
        self,
        instruments: list,
        tstart: float = 0.02,
        tend: float = 0.1,
        dt: float = 0.01,
    ):
        for instr in instruments:
            binned_quantities = {}
            for quant in self.raw_data[instr].keys():
                data_quant = deepcopy(self.raw_data[instr][quant])

                if "t" in data_quant.coords:
                    if tstart < data_quant.t.min():
                        tstart = data_quant.t.min()
                    if tend > data_quant.t.max():
                        tend = data_quant.t.max()
                    data_quant = convert_in_time_dt(tstart, tend, dt, data_quant)
                binned_quantities[quant] = data_quant
            self.binned_data[instr] = binned_quantities

    def map_diagnostics(self, instruments: list, map_raw: bool = False):
        if len(self.binned_data) == 0:
            raise ValueError("Bin data in time before remapping!")

        attr_to_map = ["binned_data"]
        if map_raw:
            attr_to_map.append("raw_data")

        for attr in attr_to_map:
            data_to_map = getattr(self, attr)
            for instr in instruments:
                if instr == "efit":
                    continue
                for quant in data_to_map[instr]:
                    data = data_to_map[instr][quant]
                    transform = data.transform
                    if hasattr(data.transform, "convert_to_rho"):
                        transform.convert_to_rho(t=data.t)
                    else:
                        break

    def filter_data(self, instruments: list):

        if not hasattr(self, "binned_data"):
            raise ValueError(
                "Bin data before filtering. No action permitted on raw data structure!"
            )

        for instr in instruments:
            if instr not in FILTER_LIMITS.keys():
                continue

            quantities = list(self.binned_data[instr])
            filter_general(
                self.binned_data[instr],
                quantities,
                lim=FILTER_LIMITS[instr],
            )

    def filter_ts(self, chi2_limit: float = 2.0):
        if "ts" not in self.binned_data.keys():
            print("No TS data to filter")
            return

        # Filter out any radial point where the chi2 is above limit
        condition = self.binned_data["ts"]["chi2"] < chi2_limit
        for quantity in self.binned_data["ts"].keys():
            attrs = self.binned_data["ts"][quantity].attrs
            filtered = xr.where(condition, self.binned_data["ts"][quantity], np.nan)
            filtered.attrs = attrs
            self.binned_data["ts"][quantity] = filtered

    def plot_profile(
        self,
        instrument: str,
        quantity: str,
        tplot: list = None,
        plot_raw: bool = False,
        xcoord: str = "rho",
        figure: bool = True,
        xlim: tuple = (0, 1.1),
        linestyle: str = None,
        plot_error: bool = True,
        **kwargs,
    ):
        R_offset = ""
        if np.abs(self.equilibrium.R_offset) > 0.01:
            R_offset = " ($R_{shift}$=" + f"{self.equilibrium.R_offset:1.2f})"
        if plot_raw:
            data_to_plot = self.raw_data[instrument][quantity]
            data_type = "Raw"
        else:
            data_to_plot = self.binned_data[instrument][quantity]
            data_type = "Binned"

        if figure:
            plt.figure()

        ymax = 0
        rho = data_to_plot.transform.rho
        R = data_to_plot.transform.R
        value = data_to_plot
        error = data_to_plot.error
        if tplot is None:
            tplot = np.array([t for t in value.t if any(np.isfinite(value.sel(t=t)))])

        tplot = list(np.array(tplot, ndmin=1))
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(tplot), dtype=float))
        for it in range(len(tplot)):
            t = tplot[it]
            _t = value.t.sel(t=t, method="nearest").values
            if xcoord == "rho":
                x = rho.sel(t=_t, method="nearest")
            elif xcoord == "R":
                x = R
            y = value.sel(t=_t, method="nearest")
            yerr = error.sel(t=_t, method="nearest")
            if linestyle is None:
                linestyle = LINESTYLES[instrument]
            if any(np.isfinite(y)):
                plt.plot(
                    x,
                    y,
                    label=f"{data_type} {instrument.upper()} "
                    f"{quantity} @ t={_t:1.3f} s",
                    color=cols_time[it],
                    marker=MARKERS[instrument],
                    linestyle=linestyle,
                )
                if plot_error:
                    plt.fill_between(
                        x, y - yerr, y + yerr, color=cols_time[it], alpha=0.5
                    )
                ymax = np.max([ymax, np.max(y + yerr)])

        plt.title(f"Pulse {self.pulse}{R_offset}")
        plt.xlabel(XLABELS[xcoord])
        plt.ylabel(YLABELS[quantity])
        plt.xlim(xlim)
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        plt.legend()

    def __call__(
        self,
        instruments: list = None,
        revisions: dict = None,
        map_raw: bool = False,
        tstart: float = None,
        tend: float = None,
        dt: float = None,
        R_shift: float = 0.0,
        chi2_limit: float = 2.0,
        map_diagnostics: bool = False,
        debug: bool = False,
    ):

        if instruments is None:
            instruments = list(REVISIONS.keys())
        if revisions is None:
            revisions = REVISIONS
        if tstart is None:
            tstart = self.tstart
        if tend is None:
            tend = self.tend
        if dt is None:
            dt = self.dt

        self.reset_data()
        self.get_equilibrium(R_shift=R_shift)
        for instrument in instruments:
            print(f"Reading {instrument}")
            if debug:
                self.get_raw_data("", instrument, revisions[instrument])
            else:
                try:
                    self.get_raw_data("", instrument, revisions[instrument])
                except Exception as e:
                    print(f"Error reading {instrument}: {e}")

        instruments = list(self.raw_data)

        print_like("Binning in time")
        self.bin_data_in_time(instruments, tstart=tstart, tend=tend, dt=dt)
        print_like("Filtering")
        self.filter_data(instruments)
        self.filter_ts(chi2_limit=chi2_limit)
        if map_diagnostics:
            print_like("Mapping to equilibrium")
            self.map_diagnostics(instruments, map_raw=map_raw)


def filter_general(data: DataArray, quantities: list, lim: tuple = (-np.inf, np.inf)):
    for quantity in quantities:
        attrs = data[quantity].attrs
        condition = (data[quantity] >= lim[0]) * (data[quantity] < lim[1])
        filtered = xr.where(condition, data[quantity], np.nan)
        filtered.attrs = attrs
        data[quantity] = filtered


def astra_equilibrium(pulse: int, revision: RevisionLike):
    """Assign ASTRA to equilibrium class"""
