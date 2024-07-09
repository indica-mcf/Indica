from copy import deepcopy
from typing import Callable
from typing import Dict

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.equilibrium import Equilibrium
from indica.numpy_typing import FloatOrDataArray
from indica.numpy_typing import RevisionLike
from indica.readers import ST40Reader

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
    "ppts",
]

FILTER_LIMITS: Dict[str, Dict[str, tuple]] = {
    "cxff_pi": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "cxff_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "cxqf_tws_c": {"ti": (0, np.inf), "vtor": (0, np.inf)},
    "xrcs": {
        "ti_w": (0, np.inf),
        "ti_z": (0, np.inf),
        "te_kw": (0, np.inf),
        "te_n3w": (0, np.inf),
        "intens": (0, np.inf),
        "radiance": (0, np.inf),
        "emission": (0, np.inf),
        "spec_rad": (0, np.inf),
    },
    "brems": {"brightness": (0, np.inf)},
    "halpha": {"brightness": (0, np.inf)},
    "sxr_spd": {"brightness": (0, np.inf)},
    "sxr_diode_1": {"brightness": (0, np.inf)},
    "sxr_camera_4": {"brightness": (0, np.inf)},
    "sxrc_xy1": {"brightness": (0, np.inf)},
    "sxrc_xy2": {"brightness": (0, np.inf)},
    "blom_xy1": {"brightness": (0, np.inf)},
    "ts": {"te": (0, np.inf), "ne": (0, np.inf)},
    "ppts": {
        "te_rho": (1, np.inf),
        "ne_rho": (1, np.inf),
        "pe_rho": (1, np.inf),
        "te_R": (1, np.inf),
        "ne_R": (1, np.inf),
        "pe_R": (1, np.inf),
    },
    "pi": {"spectra": (0, np.inf)},
    "tws_c": {"spectra": (0, np.inf)},
}

FILTER_COORDS: Dict[str, Dict[str, tuple]] = {
    "cxff_pi": {"ti": ("channel", (0, np.inf)), "vtor": ("channel", (0, np.inf))},
    "cxff_tws_c": {"ti": ("channel", (0, np.inf)), "vtor": ("channel", (0, np.inf))},
    "xrcs": {
        "intens": ("wavelength", (0.0, np.inf)),
        "radiance": ("wavelength", (0.0, np.inf)),
        "emission": ("wavelength", (0.0, np.inf)),
        "sepc_rad": ("wavelength", (0.0, np.inf)),
    },
    "ts": {"te": ("channel", (0, np.inf)), "ne": ("channel", (0, np.inf))},
}

LINESTYLES = {
    "ts": "solid",
    "ppts": "dashed",
    "cxff_pi": "solid",
    "cxff_tws_c": "dashed",
    "cxqf_tws_c": "dotted",
    "pi": "solid",
    "tws_c": "dashed",
}
MARKERS = {
    "ts": "o",
    "ppts": "x",
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
        R_shift: FloatOrDataArray = 0.0,
        z_shift: FloatOrDataArray = 0.0,
    ):

        equilibrium_data = self.reader.get("", instrument, revision)
        equilibrium = Equilibrium(equilibrium_data, R_shift=R_shift, z_shift=z_shift)
        self.equilibrium = equilibrium

    def get_raw_data(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike = 0,
        set_equilibrium: bool = False,
    ):
        data = self.reader.get(uid, instrument, revision)
        for quant in data.keys():
            if "transform" not in data[quant].attrs:
                continue

            transform = data[quant].transform
            if hasattr(transform, "set_equilibrium") and set_equilibrium:
                transform.set_equilibrium(self.equilibrium)
            self.transforms[instrument] = transform

        self.raw_data[instrument] = data

        return data

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
        filter_limits: dict = None,
        filter_coords: dict = None,
        map_raw: bool = False,
        tstart: float = None,
        tend: float = None,
        dt: float = None,
        R_shift: FloatOrDataArray = 0.0,
        chi2_limit: float = 100.0,
        map_diagnostics: bool = False,
        raw_only: bool = False,
        debug: bool = False,
        set_equilibrium: bool = False,
        fetch_equilbrium = True,
    ):
        self.debug = debug

        if tstart is None:
            tstart = self.tstart
        if tend is None:
            tend = self.tend
        if dt is None:
            dt = self.dt
        if instruments is None:
            instruments = INSTRUMENTS
        if revisions is None:
            revisions = {instrument: 0 for instrument in instruments}
        for instr in instruments:
            if instr not in revisions.keys():
                revisions[instr] = 0
        if "efit" not in revisions:
            revisions["efit"] = 0
        if filter_limits is None:
            filter_limits = FILTER_LIMITS
        if filter_coords is None:
            filter_coords = FILTER_COORDS

        self.reset_data()
        if fetch_equilbrium:
            self.get_equilibrium(R_shift=R_shift, revision=revisions["efit"])
        for instrument in instruments:
            print(f"Reading {instrument}")
            try:
                self.get_raw_data(
                    "",
                    instrument,
                    revisions[instrument],
                    set_equilibrium=set_equilibrium,
                )
            except Exception as e:
                print(f"error reading: {instrument} \nException: {e}")
                if debug:
                    raise e

        if raw_only:
            return

        print("Filtering")
        self.filtered_data = apply_filter(
            self.raw_data,
            filters=filter_limits,
            filter_func=limit_condition,
            filter_func_name="limits",
        )
        self.filtered_data = apply_filter(
            self.filtered_data,
            filters=filter_coords,
            filter_func=coord_condition,
            filter_func_name="co-ordinate",
        )

        print("Binning in time")
        self.binned_data = bin_data_in_time(
            self.filtered_data, tstart=tstart, tend=tend, dt=dt
        )

        filter_ts(self.binned_data, chi2_limit=chi2_limit)
        if map_diagnostics or map_raw:
            print("Mapping to equilibrium")
            instruments = list(self.raw_data)
            self.map_diagnostics(instruments, map_raw=map_raw)


def bin_data_in_time(
    raw_data: Dict[str, Dict[str, xr.DataArray]],
    tstart: float = 0.02,
    tend: float = 0.1,
    dt: float = 0.01,
    debug=False,
):
    binned_data = {}
    for instr in raw_data.keys():
        if debug:
            print(f"instr: {instr}")
        binned_quantities = {}
        for quant in raw_data[instr].keys():
            if debug:
                print(f"quant: {quant}")
            data_quant = deepcopy(raw_data[instr][quant])

            if "t" in data_quant.coords:
                data_quant = convert_in_time_dt(tstart, tend, dt, data_quant)
            # Using groupedby_bins always removes error from coords so adding it back
            if "error" in raw_data[instr][quant].coords:
                error = convert_in_time_dt(
                    tstart, tend, dt, raw_data[instr][quant].error
                )
                data_quant = data_quant.assign_coords(
                    error=(raw_data[instr][quant].dims, error)
                )
            binned_quantities[quant] = data_quant
        binned_data[instr] = binned_quantities
    return binned_data


def apply_filter(
    data: Dict[str, Dict[str, xr.DataArray]],
    filters: Dict[str, Dict[str, tuple]],
    filter_func: Callable,
    filter_func_name="limits",
    verbose = True,
):

    filtered_data = {}
    for instrument, quantities in data.items():
        if instrument not in filters.keys():
            if verbose:
                print(f"no {filter_func_name} filter for {instrument}")
            filtered_data[instrument] = deepcopy(data[instrument])
            continue

        filtered_data[instrument] = {}
        for quantity_name, quantity in quantities.items():
            if quantity_name not in filters[instrument]:
                filtered_data[instrument][quantity_name] = deepcopy(
                    data[instrument][quantity_name]
                )
                continue

            filter_info = filters[instrument][quantity_name]
            filtered_data[instrument][quantity_name] = filter_func(
                quantity, filter_info
            )
    return filtered_data


def limit_condition(data: DataArray, limits: tuple):
    condition = (data >= limits[0]) * (data < limits[1])
    filtered_data = xr.where(condition, data, np.nan)
    filtered_data.attrs = data.attrs
    return filtered_data


def coord_condition(data: DataArray, coord_info: tuple):
    coord_name: str = coord_info[0]
    coord_slice: tuple = coord_info[1]
    condition = (data.coords[coord_name] >= coord_slice[0]) * (
        data.coords[coord_name] < coord_slice[1]
    )
    filtered_data = data.where(condition, np.nan)
    filtered_data.attrs = data.attrs
    return filtered_data


def filter_ts(binned_data: dict, chi2_limit: float = 3.0):
    if "ts" not in binned_data.keys():
        print("No TS data to filter")
        return

    # Filter out any radial point where the chi2 is above limit
    condition = binned_data["ts"]["chi2"] < chi2_limit
    for quantity in binned_data["ts"].keys():
        attrs = binned_data["ts"][quantity].attrs
        filtered = xr.where(condition, binned_data["ts"][quantity], np.nan)
        filtered.attrs = attrs
        binned_data["ts"][quantity] = filtered
