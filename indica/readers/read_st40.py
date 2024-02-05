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

INSTRUMENTS: list = [
    "efit",
    "lines",
    "nirh1",
    "smmh1",
    "smmh",
    "cxff_pi",
    "cxff_tws_c",
    "cxqf_tws_c",
    "sxr_spd",
    "sxr_camera_4",
    "sxrc_xy1",
    "sxrc_xy2",
    "sxr_diode_1",
    "xrcs",
    "ts",
    "pi",
    "tws_c",
]

FILTER_LIMITS = {
    "cxff_pi": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "cxff_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "cxqf_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "xrcs": {"ti_w": (0, np.inf), "te_kw": (0, np.inf), "te_n3w": (0, np.inf)},
    "brems": {"brightness": (0, np.inf)},
    "halpha": {"brightness": (0, np.inf)},
    "sxr_spd": {"brightness": (0, np.inf)},
    "sxr_diode_1": {"brightness": (0, np.inf)},
    "sxr_camera_4": {"brightness": (0, np.inf)},
    "sxrc_xy1": {"brightness": (0, np.inf)},
    "sxrc_xy2": {"brightness": (0, np.inf)},
    "blom_xy1": {"brightness": (0, np.inf)},
    "ts": {"te": (0, np.inf), "ne": (0, np.inf)},
    "pi": {"spectra": (0, np.inf)},
    "tws_c": {"spectra": (0, np.inf)},
}

LINESTYLES = {
    "ts": "solid",
    "cxff_pi": "solid",
    "cxff_tws_c": "dashed",
    "cxqf_tws_c": "dotted",
    "pi": "solid",
    "tws_c": "dashed",
}
MARKERS = {
    "ts": "o",
    "cxff_pi": "s",
    "cxff_tws_c": "*",
    "cxqf_tws_c": "x",
    "pi": "s",
    "tws_c": "*",
}
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
        tend: float = 0.1,
        dt: float = 0.01,
        tree="ST40",
    ):
        self.debug = False
        self.pulse = pulse
        self.tstart = tstart
        self.tend = tend
        self.dt = dt

        _tend = tend + dt * 2
        _tstart = tstart - dt * 2
        if _tstart < 0:
            _tstart = 0.0
        self.reader = ST40Reader(pulse, _tstart, _tend, tree=tree)
        self.reader_equil = ST40Reader(pulse, _tstart, _tend, tree=tree)

        self.equilibrium: Equilibrium
        self.raw_data: dict = {}
        self.raw_data_trange: dict = {}
        self.binned_data: dict = {}
        self.transforms: dict = {}

    def reset_data(self):
        self.raw_data = {}
        self.raw_data_trange = {}
        self.binned_data = {}

    def get_equilibrium(
        self,
        instrument: str = "efit",
        revision: RevisionLike = 0,
        R_shift: float = 0.0,
        z_shift: float = 0.0,
    ):

        equilibrium_data = self.reader_equil.get("", instrument, revision)
        equilibrium = Equilibrium(equilibrium_data, R_shift=R_shift, z_shift=z_shift)
        self.equilibrium = equilibrium

    def get_raw_data(self, uid: str, instrument: str, revision: RevisionLike = 0):
        data = self.reader.get(uid, instrument, revision)
        if hasattr(self, "equilibrium"):
            for quant in data.keys():
                if "transform" not in data[quant].attrs:
                    continue

                transform = data[quant].transform
                # if hasattr(transform, "set_equilibrium"):
                #     transform.set_equilibrium(self.equilibrium)
                self.transforms[instrument] = transform
        self.raw_data[instrument] = data

        self.raw_data_trange[instrument] = {}
        for quant in data.keys():
            if "t" in data[quant].dims:
                self.raw_data_trange[instrument][quant] = data[quant].sel(
                    t=slice(self.tstart, self.tend)
                )
                if "error" in data[quant].attrs:
                    self.raw_data_trange[instrument][quant].attrs["error"] = data[
                        quant
                    ].error.sel(t=slice(self.tstart, self.tend))
                else:
                    self.raw_data_trange[instrument][quant] = data[quant]
        return data

    def bin_data_in_time(
        self,
        instruments: list,
        tstart: float = 0.02,
        tend: float = 0.1,
        dt: float = 0.01,
    ):
        for instr in instruments:
            if self.debug:
                print(instr)
            binned_quantities = {}
            for quant in self.raw_data[instr].keys():
                if self.debug:
                    print(f"   {quant}")
                data_quant = deepcopy(self.raw_data[instr][quant])

                if "t" in data_quant.coords:
                    data_quant = convert_in_time_dt(tstart, tend, dt, data_quant)

                binned_quantities[quant] = data_quant
            self.binned_data[instr] = binned_quantities

    def map_diagnostics(self, instruments: list, map_raw: bool = False):
        if len(self.binned_data) == 0:
            raise ValueError("Bin data in time before remapping!")

        attr_to_map = ["binned_data"]
        if map_raw:
            attr_to_map = ["raw_data"]

        for attr in attr_to_map:
            data_to_map = getattr(self, attr)
            for instr in instruments:
                if instr == "efit":
                    continue
                if self.debug:
                    print(instr)
                for quant in data_to_map[instr]:
                    if self.debug:
                        print(f"   {quant}")
                    data = data_to_map[instr][quant]
                    transform = data.transform
                    if hasattr(data.transform, "convert_to_rho_theta"):
                        transform.convert_to_rho_theta(t=data.t)
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
                limits=FILTER_LIMITS[instr],
            )

    def filter_ts(self, chi2_limit: float = 3.0):
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

    # def add_mhd(self):
    #     t_slice = slice(self.tstart, self.tend)
    #     rev = 0
    #
    #     even, even_dims = self.reader._get_data(
    #         "", "mhd_tor_mode", ".output.spectrogram:ampl_even", rev
    #     )
    #     odd, odd_dims = self.reader._get_data(
    #         "", "mhd_tor_mode", ".output.spectrogram:ampl_odd", rev
    #     )
    #     try:
    #         even = DataArray(even, coords=[("t", even_dims[0])]).sel(t=t_slice)
    #         odd = DataArray(odd, coords=[("t", odd_dims[0])]).sel(t=t_slice)
    #         self.raw_data["mhd"] = {}
    #         self.raw_data["mhd"]["ampl_even_n"] = even
    #         self.raw_data["mhd"]["ampl_odd_n"] = odd
    #     except IndexError:
    #         return

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
        instruments: list = [],
        revisions: dict = None,
        filters: dict = None,
        map_raw: bool = False,
        tstart: float = None,
        tend: float = None,
        dt: float = None,
        R_shift: float = 0.0,
        chi2_limit: float = 3.0,
        map_diagnostics: bool = False,
        raw_only: bool = False,
        debug: bool = False,
    ):
        self.debug = debug
        if len(instruments) == 0:
            instruments = INSTRUMENTS
        if revisions is None:
            revisions = {instrument: 0 for instrument in instruments}
        for instr in instruments:
            if instr not in revisions.keys():
                revisions[instr] = 0
        if "efit" not in revisions:
            revisions["efit"] = 0
        if not filters:
            # TODO: fix default behaviour if missing key
            filters = FILTER_LIMITS
        if tstart is None:
            tstart = self.tstart
        if tend is None:
            tend = self.tend
        if dt is None:
            dt = self.dt

        self.reset_data()
        self.get_equilibrium(R_shift=R_shift, revision=revisions["efit"])
        for instrument in instruments:
            print(f"Reading {instrument}")
            try:
                self.get_raw_data("", instrument, revisions[instrument])
            except Exception as e:
                print(f"Error reading {instrument}: {e}")
                if debug:
                    raise e

        if raw_only:
            return

        instruments = list(self.raw_data)

        print_like("Binning in time")
        self.bin_data_in_time(instruments, tstart=tstart, tend=tend, dt=dt)
        print_like("Filtering")
        self.filter_data(instruments)
        self.filter_ts(chi2_limit=chi2_limit)
        if map_diagnostics or map_raw:
            print_like("Mapping to equilibrium")
            self.map_diagnostics(instruments, map_raw=map_raw)


def filter_general(data: DataArray, quantities: list, limits: dict):
    for quantity in quantities:
        if quantity in limits.keys():
            lim = limits[quantity]
            attrs = data[quantity].attrs
            condition = (data[quantity] >= lim[0]) * (data[quantity] < lim[1])
            filtered = xr.where(condition, data[quantity], np.nan)
            filtered.attrs = attrs
            data[quantity] = filtered


def astra_equilibrium(pulse: int, revision: RevisionLike):
    """Assign ASTRA to equilibrium class"""


def sxr_spd():
    import indica.readers.read_st40 as read_st40
    from indica.utilities import set_axis_sci

    st40 = read_st40.ReadST40(11215)
    st40(["sxr_spd"])

    st40.raw_data["sxr_spd"]["brightness"].transform.plot()

    plt.figure()
    st40.raw_data["sxr_spd"]["brightness"].sel(channel=0).plot()
    set_axis_sci()

    return st40.raw_data["sxr_spd"]["brightness"]
