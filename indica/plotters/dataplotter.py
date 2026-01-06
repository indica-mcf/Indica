"""Plotting routines for Indica read/modelled data"""
from copy import deepcopy
from getpass import getuser
from typing import Dict
from typing import List
from typing import Union

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica import Equilibrium
from indica import Plasma
from indica.configs import MACHINE_CONFS
from indica.numpy_typing import LabeledArray
from indica.utilities import assign_datatype
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.utilities import set_plot_rcparams

xr.set_options(keep_attrs=True)
RHOP_VALS = ("0.1", "0.7")

CM, COLORS = set_plot_colors()
FIG_PATH = f"/home/{getuser()}/figures/"


class DataPlotter:
    def __init__(
        self,
        pulse: int,
        t: LabeledArray,
        tstart: float = None,
        tend: float = None,
        machine: str = "st40",
        ttol: float = 0.005,
        nplot: int = 3,
        rc_params: dict = None,
    ):
        """
        pulse = Pulse number from which the data originated
        t = Time at which to plot profiles
        tstart = Start of time range for which to plot
        tend = End of time range for which to plot
        ttol = tolerance for "nearest" timepoint selection
        nplot = reduce number of plots by nplot times
        """

        self.conf = MACHINE_CONFS[machine]()

        set_plot_rcparams("profiles", rc_params=rc_params)

        self.pulse = pulse
        self.ttol = ttol
        self.nplot = nplot

        if tstart is None:
            tstart = float(np.min(t))
        if tend is None:
            tend = float(np.max(t))
        self.tstart = tstart
        self.tend = tend

        _t = np.array(t, ndmin=1)
        self.times = _t[np.where((_t >= tstart) * (_t <= tend))[0]]
        self.title = f"{pulse} @ t=[{tstart:.3f}, {tend:.3f}] s"
        self.fig_name = f"{pulse}_{tstart:.3f}_{tend:.3f}_s"
        self.colors = CM(np.linspace(0.1, 0.75, np.size(self.times), dtype=float))

    # Profiles
    def _plot_profile(
        self,
        data: DataArray,
        xdim: str = None,
        use_label: bool = True,
        **kwargs,
    ):
        _kwargs = pop_kwargs(kwargs, ["label", "color"])
        for i, t in enumerate(self.times):
            _t = self.within_tolerance(data, t)

            if _t is None or i % self.nplot:
                continue

            x, y, err = select_x_y_err(data, t=_t, xdim=xdim)

            label = None
            if use_label:
                label = f"{_t:.3f} s"

            # Plot data
            y.plot(label=label, color=self.colors[i], **_kwargs)

            # Plot uncertainty band
            plt.fill_between(x, y - err, y + err, color=self.colors[i], alpha=0.5)

    # Experimental profile data
    def _plot_profile_data(
        self,
        data: DataArray,
        xdim: str = None,
        use_label: bool = True,
        **kwargs,
    ):
        _kwargs = pop_kwargs(kwargs, ["label", "color"])
        if "linestyle" not in _kwargs:
            _kwargs["linestyle"] = ""

        for i, t in enumerate(self.times):
            _t = self.within_tolerance(data, t)

            if _t is None or i % self.nplot or not np.any(np.isfinite(data.sel(t=_t))):
                continue

            x, y, err = select_x_y_err(data, _t, xdim=xdim)

            label = None
            if use_label:
                label = f"{_t:.3f} s"

            y.plot(label=label, color=self.colors[i], **_kwargs)
            plt.errorbar(x, y, err, color=self.colors[i])

    # Time evolution
    def _plot_time_evolution(
        self,
        data,
        label: str = None,
        xdim: str = "t",
        **kwargs,
    ):
        _kwargs = pop_kwargs(kwargs, ["label"])
        if "marker" not in _kwargs:
            _kwargs["marker"] = "o"
        if "linestyle" not in _kwargs:
            _kwargs["linestyle"] = ""

        x, y, err = select_x_y_err(data, xdim=xdim)

        y.plot(label=label, **_kwargs)
        plt.fill_between(y.t, y - err, y + err, alpha=0.5)
        plt.xlim(self.tstart, self.tend)
        plt.legend()

    # Instrument geometry
    def plot_transform(self, instrument: str, data: dict, quantity: str):
        fig_name = f"{self.fig_name}_{instrument.upper()}_transform"
        data[quantity].transform.plot(fig_name=fig_name, save_fig=save_fig)

    def plot(
        self,
        data: Union[Dict[str, Dict[str, DataArray]], Plasma],
        instrument: str,
        quantity: str,
        sci: bool = False,
        ylog: bool = False,
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str = None,
        fig_name: str = None,
        new_fig: bool = True,
        save_fig: bool = False,
        **kwargs,
    ):
        """
        General plotting method for all Indica native data, whether experimental,
        back-calculated modelled, or Plasma

        Parameters
        ----------
        data
        - experimental/modelled data as read by .indica.readers.datareader.__call__()
        quantity
        - instrument quantity to be plotted
        instrument
        - instrument identifier as defined in the machine-specific reader conf
        sci
        - True if y-axis labels in scientific format
        ylog
        - True if y-axis in log scale
        xlim
        ylim
        new_fig
        save_fig
        kwargs

        Returns
        -------

        """
        if instrument == "plasma":
            plot_method = "plot_plasma"
            _data = data
        else:
            plot_method = self.conf.INSTRUMENT_METHODS[instrument].replace(
                "get_", "plot_"
            )
            _data = data[instrument]

        if title is None:
            title = f"{instrument.upper()} for {self.title}"
        if fig_name is None:
            fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"

        new_figure(new_fig)
        getattr(self, plot_method)(_data, quantity, fig_name=fig_name, **kwargs)
        common_plot_calls(title, sci, ylog, xlim=xlim, ylim=ylim)
        save_figure(FIG_PATH, fig_name, save_fig=save_fig)

    # Instrument specific methods
    def plot_thomson_scattering(
        self,
        data: dict,
        quantity: str = "ne",
        fig_name: str = "",
        marker="o",
        linestyle="",
        **kwargs,
    ):
        xdim = "R"
        y = (
            data[quantity]
            .assign_coords(R=("channel", data[xdim].data))
            .swap_dims({"channel": xdim})
        )
        assign_datatype(y.coords[xdim], xdim)
        self._plot_profile_data(y, xdim, marker=marker, linestyle=linestyle, **kwargs)

    def plot_profile_fits(
        self,
        data: dict,
        quantity: str = "ne_rhop",
        fig_name: str = "",
        marker="o",
        linestyle="",
        **kwargs,
    ):
        xdim = "rhop"
        y_fit = xr.where(data[quantity] > 0, data[quantity], np.nan)
        try:
            _y_exp = xr.where(
                y_fit.mean(xdim) > 0,
                data[quantity.replace(f"_{xdim}", "_data")],
                np.nan,
            )
            y_exp = _y_exp.assign_coords(rhop=(_y_exp.dims, data[f"{xdim}_data"].data))
        except KeyError:
            y_exp = None

        if y_exp is not None:
            self._plot_profile_data(
                y_exp,
                xdim=xdim,
                marker=marker,
                linestyle=linestyle,
                use_label=False,
                **kwargs,
            )
        self._plot_profile(y_fit, xdim=xdim, **kwargs)

    def plot_spectrometer(
        self,
        data: dict,
        quantity: str = "spectra",
        fig_name: str = "",
        channel: int = None,
        **kwargs,
    ):
        # Spectra vs wavelength
        xdim = "wavelength"
        if channel is None:
            channel = data[quantity].channel.median()
        y = data[quantity].sel(channel=channel)
        self._plot_profile(y, xdim, **kwargs)

    def plot_zeff(
        self,
        data: dict,
        quantity: str = "zeff",
        fig_name: str = "",
        **kwargs,
    ):
        xdim = "rhop"
        y = data[quantity]
        self._plot_profile(y, xdim, **kwargs)
        plt.hlines(1, 0, 1, linestyle="dotted", color="k")

    def plot_radiation_inversion(
        self,
        data: dict,
        quantity: str = "emission_rhop",
        fig_name: str = "",
        **kwargs,
    ):
        xdim = "rhop"
        y = xr.where(data[quantity] > 0, data[quantity], np.nan)
        if quantity == "emission_rhop":
            self._plot_profile(y, xdim=xdim, **kwargs)
        elif quantity == "prad":
            self._plot_time_evolution(y, **kwargs)
        else:
            print(f"Plotting for {quantity} not implemented")

    def plot_charge_exchange(
        self,
        data: dict,
        quantity: str = "ti",
        fig_name: str = "",
        **kwargs,
    ):
        # Keep only channels where fits have been performed
        xdim = "R"
        all_chans = np.where(
            [
                np.any((data[quantity].sel(channel=ch) > 0).data)
                for ch in data[quantity].channel
            ]
        )[0]
        trans = data[quantity].sel(channel=all_chans[0]).transform
        R = trans.R.sel(channel=all_chans)

        y = (
            data[quantity]
            .sel(channel=all_chans)
            .assign_coords(R=("channel", R.data))
            .swap_dims({"channel": xdim})
        )
        for dim in y.dims:
            assign_datatype(y.coords[dim], dim)
        y.attrs = data[quantity].attrs
        self._plot_profile_data(y, xdim, **kwargs)

    def plot_equilibrium(
        self,
        data: dict,
        quantity: str = "ipla",
        fig_name: str = "",
        **kwargs,
    ):
        self._plot_time_evolution(data[quantity], **kwargs)

    def plot_radiation(
        self,
        data: dict,
        quantity: str = "brightness",
        fig_name: str = "",
        **kwargs,
    ):
        self._plot_profile_data(
            data[quantity],
            **kwargs,
        )

    def plot_helike_spectroscopy(
        self,
        data: dict,
        quantity: str = "ti_w",
        fig_name: str = "",
        **kwargs,
    ):
        y = data[quantity]
        if "ti" in quantity or "te" in quantity:
            self._plot_time_evolution(y, label=quantity, **kwargs)
        elif "spectra" in quantity:
            self._plot_profile(y, "wavelength", **kwargs)

    def plot_diode_filters(
        self,
        data: dict,
        quantity: str = "",
        fig_name: str = "",
    ):
        raise NotImplementedError

    def plot_interferometry(
        self,
        data: dict,
        quantity: str = "ne",
        fig_name: str = "",
        **kwargs,
    ):
        self._plot_time_evolution(data[quantity], **kwargs)

    def plot_plasma(
        self,
        plasma: Plasma,
        quantity: str,
        fig_name: str = "",
        element: str = None,
        **kwargs,
    ):
        xdim = "rhop"
        _y = deepcopy(getattr(plasma, quantity))

        if type(_y) is dict:
            _y = _y[element]

        if "element" in _y.dims:
            if element is not None:
                elem = element
                fig_name += f"{elem}"
                _y = _y.sel(element=elem)
            else:
                _y = _y.sum("element")

        if "rhop" in _y.dims:
            # Profiles (e.g. Ti)
            if "ion_charge" in _y.dims:
                use_label = True
                for q in _y.ion_charge:
                    y = _y.sel(ion_charge=q)
                    self._plot_profile(y, xdim=xdim, use_label=use_label, **kwargs)
                    use_label = False
            else:
                y = xr.where(_y > 0, _y, np.nan)
                self._plot_profile(y, xdim=xdim, **kwargs)
        else:
            # 1D integral values (e.g. Prad)
            y = _y
            self._plot_time_evolution(y, **kwargs)

    def within_tolerance(self, y, t):
        try:
            _t = y.t.sel(t=t, method="nearest", tolerance=self.ttol)
            return _t
        except Exception as e:
            print(f"Skipping t={t:.3f} - beyond tolerance. \n --> error: {e}")
            return None


def pop_kwargs(
    kwargs,
    to_pop: List[str],
):
    """
    Pop list of keys from kwargs dictionary
    """
    _kwargs = deepcopy(kwargs)
    for key in to_pop:
        if key in _kwargs:
            _kwargs.pop(key)

    return _kwargs


def select_x_y_err(data, t: float = None, xdim: str = None):
    if t is not None:
        y = data.sel(t=t)
    else:
        y = deepcopy(data)
    if xdim is None:
        xdim = y.dims[0]

    if xdim not in y.dims:
        y = y.swap_dims({y.dims[0]: xdim})
    x = getattr(y, xdim)

    err = xr.full_like(y, 0.0)
    if hasattr(y, "error"):
        err = y.error

    return x, y, err


def common_plot_calls(
    title: str = "",
    sci: bool = True,
    ylog: bool = False,
    ylim: tuple = (None, None),
    xlim: tuple = (None, None),
):
    plt.legend()
    plt.title(title)

    if sci:
        set_axis_sci()

    if ylog:
        plt.yscale("log")

    plt.ylim(ylim[0], ylim[1])
    plt.xlim(xlim[0], xlim[1])


def new_figure(new_fig):
    if new_fig:
        plt.figure()


if __name__ == "__main__":
    from indica.readers import ReaderProcessor
    from indica.readers import ST40Reader
    from indica.examples import example_plasma

    processor = ReaderProcessor()

    pulse = 11419
    tstart = 0.015
    tend = 0.195
    dt = 0.01
    t = 0.07, 0.11, 0.13, 0.16
    save_fig = True

    st40 = ST40Reader(pulse, 0, 0.2)
    raw = st40(["pi", "xrcs", "cxff_pi", "ts", "psu", "mag", "efit", "sxrc_xy1"])
    post_processed = st40(
        ["ppts", "t1d_blom_xy1", "t1d_sxrc_xy1", "cxff_pi", "bda", "zeff_brems"],
        revisions={"bda": 1},
    )
    processed = processor(raw, tstart=tstart, tend=tend, dt=dt)

    plt.ioff()
    data_plotter = DataPlotter(pulse, t, tstart, tend, nplot=1)
    data_plotter.plot(raw, "ts", "ne")
    data_plotter.plot(post_processed, "ppts", "ne_rhop")
    data_plotter.plot(post_processed, "bda", "ti_rhop")
    data_plotter.plot(raw, "pi", "spectra")
    data_plotter.plot(post_processed, "zeff_brems", "zeff", ylim=(0, 6))
    data_plotter.plot(post_processed, "t1d_sxrc_xy1", "emission_rhop", sci=True)
    data_plotter.plot(post_processed, "t1d_sxrc_xy1", "prad", sci=True)
    data_plotter.plot(post_processed, "cxff_pi", "ti", sci=True)
    data_plotter.plot(raw, "efit", "ipla", sci=True, ylim=(0, None))
    data_plotter.plot(raw, "sxrc_xy1", "brightness", sci=True, ylim=(0, None))
    data_plotter.plot(raw, "xrcs", "ti_w", sci=True, ylim=(0, None))
    data_plotter.plot(raw, "xrcs", "te_n3w", sci=True, ylim=(0, None), new_fig=False)
    data_plotter.plot(raw, "xrcs", "spectra_raw", ylim=(0, None), xlim=(0.394, 0.401))

    plasma = example_plasma()
    plasma.set_equilibrium(Equilibrium(raw["efit"]))
    plasma_plotter = DataPlotter(pulse, plasma.t, nplot=1)
    plasma_plotter.plot(plasma, "plasma", "electron_density")
    plasma_plotter.plot(plasma, "plasma", "zeff")
    plasma_plotter.plot(plasma, "plasma", "prad_tot", linestyle="solid")
    plt.show()
