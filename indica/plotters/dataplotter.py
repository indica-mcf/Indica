"""Plotting routines for Indica read/modelled data"""
from copy import deepcopy
from getpass import getuser

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica import Plasma
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

    def _plot_profile(
        self,
        data: DataArray,
        xdim: str = None,
        label: str = None,
        use_label: bool = True,
        **kwargs,
    ):
        for i, t in enumerate(self.times):
            _t = self.within_tolerance(data, t)

            if _t is None or i % self.nplot:
                continue

            x, y, err = select_x_y_err(data, t=_t, xdim=xdim)

            if use_label:
                label = f"{_t:.3f} s"

            self.populate_kwargs(kwargs, label=label, color=self.colors[i])

            # Plot data
            y.plot(**kwargs)

            # Plot uncertainty band
            plt.fill_between(x, y - err, y + err, color=kwargs["color"], alpha=0.5)

    def _plot_profile_data(
        self,
        data: DataArray,
        xdim: str = None,
        label: str = None,
        use_label: bool = True,
        **kwargs,
    ):
        marker = "o"
        linestyle = ""
        for i, t in enumerate(self.times):
            _t = self.within_tolerance(data, t)

            if _t is None or i % self.nplot or not np.any(np.isfinite(data.sel(t=_t))):
                continue

            x, y, err = select_x_y_err(data, _t, xdim=xdim)

            if use_label and label is not None:
                label = f"{_t:.3f} s"
            self.populate_kwargs(
                kwargs,
                label=label,
                color=self.colors[i],
                marker=marker,
                linestyle=linestyle,
            )

            y.plot(**kwargs)
            plt.errorbar(x, y, err, linestyle="", color=self.colors[i])

    def _plot_time_evolution(
        self,
        data,
        label: str = None,
        xdim: str = "t",
        **kwargs,
    ):
        marker = "o"

        x, y, err = select_x_y_err(data, xdim=xdim)

        self.populate_kwargs(kwargs, label=label, marker=marker)

        y.plot(**kwargs)
        plt.fill_between(y.t, y - err, y + err, alpha=0.5)
        plt.xlim(self.tstart, self.tend)
        plt.legend()

    # Instrument specific methods
    def plot_thomson_scattering(
        self,
        instrument: str,
        data: dict,
        quantities: tuple = ("ne", "te"),
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        **kwargs,
    ):
        title = f"{instrument.upper()} for {self.title}"
        xdim = "R"
        for quantity in quantities:
            fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
            y = (
                data[quantity]
                .assign_coords(R=("channel", data[xdim].data))
                .swap_dims({"channel": xdim})
            )
            assign_datatype(y.coords[xdim], xdim)

            new_figure(new_fig)
            self._plot_profile_data(
                y,
                xdim,
            )
            common_plot_calls(title, True, False)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_profile_data", save_fig=save_fig)

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig, fig_name=fig_name)

    def plot_profile_fits(
        self,
        instrument: str,
        data: dict,
        new_fig: bool = True,
        save_fig: bool = False,
        quantities: tuple = ("ne_rhop", "te_rhop"),
        to_plot: tuple = ("profiles", "time"),
        xplot: tuple = RHOP_VALS,
        **kwargs,
    ):
        title = f"{instrument.upper()} for {self.title}"
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")
        xdim = "rhop"

        sci = True
        if "sci" in kwargs:
            sci = kwargs["sci"]
            kwargs.pop("sci")
        ymax = None
        if "ymax" in kwargs:
            ymax = kwargs["ymax"]
            kwargs.pop("ymax")
        for quantity in quantities:
            fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
            # Profiles on rhop
            if "profiles" in to_plot:
                y_fit = xr.where(data[quantity] > 0, data[quantity], np.nan)
                try:
                    _y_exp = xr.where(
                        y_fit.mean(xdim) > 0,
                        data[quantity.replace(f"_{xdim}", "_data")],
                        np.nan,
                    )
                    y_exp = _y_exp.assign_coords(
                        rhop=(_y_exp.dims, data[f"{xdim}_data"].data)
                    )
                except KeyError:
                    y_exp = None

                new_figure(new_fig)
                if y_exp is not None:
                    self._plot_profile_data(y_exp, xdim=xdim, label="")
                self._plot_profile(y_fit, xdim=xdim, **kwargs)
                common_plot_calls(title, sci, False, 0, ymax, 0, 1.2)
                if save_fig:
                    save_figure(FIG_PATH, f"{fig_name}_profile", save_fig=save_fig)

            # Time evolution of central values
            if "time" in to_plot:
                new_figure(new_fig)
                for _xplot in xplot:
                    _y = data[quantity].sel(rhop=float(_xplot), method="nearest")
                    y = xr.where(_y > 0, _y, np.nan)
                    xname = data[quantity].coords[xdim].long_name

                    label = f"{instrument.upper()} {y.long_name}({xname}={_xplot})"
                    if "label" in kwargs is None:
                        label = kwargs["label"]

                    self._plot_time_evolution(y, label=label)
                common_plot_calls(title, sci, False, 0)
                if save_fig:
                    save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

    def plot_bayesian(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        quantities: tuple = ("ne_rhop", "te_rhop", "ti_rhop"),
        to_plot: tuple = ("profiles", "time"),
        xplot: tuple = RHOP_VALS,
        **kwargs,
    ):
        title = f"{instrument.upper()} for {self.title}"
        xdim = "rhop"

        for quantity in quantities:
            fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
            # Profiles on rhop
            if "profiles" in to_plot:
                y_fit = xr.where(data[quantity] > 0, data[quantity], np.nan)

                new_figure(new_fig)
                self._plot_profile(y_fit, xdim=xdim, **kwargs)
                common_plot_calls(title, True, False, 0, None, 0, 1.2)
                if save_fig:
                    save_figure(FIG_PATH, f"{fig_name}_profile", save_fig=save_fig)

            # Time evolution of central values
            if "time" in to_plot:
                new_figure(new_fig)
                for _xplot in xplot:
                    _y = data[quantity].sel(rhop=float(_xplot), method="nearest")
                    y = xr.where(_y > 0, _y, np.nan)
                    xname = data[quantity].coords[xdim].long_name

                    label = f"{instrument.upper()} {y.long_name}({xname}={_xplot})"
                    if "label" in kwargs is None:
                        label = kwargs["label"]

                    self._plot_time_evolution(y, label=label)
                common_plot_calls(title, True, False, 0)
                if save_fig:
                    save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

    def plot_spectrometer(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        channel: int = None,
        **kwargs,
    ):
        # Spectra vs wavelength
        title = f"{instrument.upper()} for {self.title}"
        xdim = "wavelength"
        quantity = "spectra"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
        if channel is None:
            channel = data[quantity].channel.median()
        y = data[quantity].sel(channel=channel)

        new_figure(new_fig)
        self._plot_profile(y, xdim)
        xmin = None
        xmax = None
        if xdim == "rhop":
            xmin = 0
            xmax = 1.2
        common_plot_calls(title, True, False, 0, None, xmin, xmax)
        if save_fig:
            save_figure(FIG_PATH, f"{fig_name}_data", save_fig=save_fig)

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig)

    def plot_zeff(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        to_plot: tuple = ("profiles", "time"),
        **kwargs,
    ):
        # Zeff profile evolution in time
        title = f"{instrument.upper()} for {self.title}"
        xdim = "rhop"
        quantity = "zeff"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
        ylim = (0, 6)
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]

        if "profiles" in to_plot:
            new_figure(new_fig)
            y = data[quantity]
            self._plot_profile(y, xdim)
            common_plot_calls(title, False, False, ylim[0], ylim[1])
            plt.hlines(1, 0, 1, linestyle="dotted", color="k")
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_profile", save_fig=save_fig)

        # Time evolution
        if "time" in to_plot:
            new_figure(new_fig)
            for xplot in RHOP_VALS:
                _y = data[quantity].sel(rhop=float(xplot), method="nearest")
                y = xr.where(_y > 0, _y, np.nan)
                xname = data[quantity].coords[xdim].long_name

                label = f"{instrument.upper()} {y.long_name}({xname}={xplot})"
                if "label" in kwargs is None:
                    label = kwargs["label"]
                self._plot_time_evolution(y, label=label)
            y = data["zeff_avrg"]
            self._plot_time_evolution(
                y, label=f"{instrument.upper()} {y.long_name} LOS-avrg"
            )
            common_plot_calls(title, False, False, ylim[0], ylim[1])
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

    def plot_radiation_inversion(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        to_plot: tuple = ("profiles", "time", "integral"),
        **kwargs,
    ):
        title = f"{instrument.upper()} for {self.title}"
        if "title" in kwargs:
            title = kwargs["title"]
        quantity = "emission_rhop"
        xdim = "rhop"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"

        # Profiles on rhop
        sci = True
        if "sci" in kwargs:
            sci = kwargs["sci"]
            kwargs.pop("sci")
        if "profiles" in to_plot:
            y_fit = xr.where(data[quantity] > 0, data[quantity], np.nan)
            new_figure(new_fig)
            self._plot_profile(y_fit, xdim=xdim)
            common_plot_calls(title, sci, False, 0, None, 0, 1)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_profile", save_fig=save_fig)

        if "time" in to_plot:
            # Time evolution of central values
            new_figure(new_fig)
            for xplot in RHOP_VALS:
                _y = data[quantity].sel(rhop=float(xplot), method="nearest")
                y = xr.where(_y > 0, _y, np.nan)
                xname = data[quantity].coords[xdim].long_name
                label = f"{instrument.upper()} {y.long_name}({xname}={xplot})"
                if "label" in kwargs:
                    label = kwargs["label"]
                self._plot_time_evolution(y, label=label)
            common_plot_calls(title, sci, False, 0)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol_emission", save_fig=save_fig)

        if "integral" in to_plot:
            # Time evolution of total radiated power
            quantity = "prad"
            new_figure(new_fig)
            label = f"{instrument.upper()}"
            if "label" in kwargs:
                label = kwargs["label"]
            self._plot_time_evolution(data[quantity], label=label)
            common_plot_calls(title, sci, False, 0)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol_integral", save_fig=save_fig)

    def plot_charge_exchange(
        self,
        instrument: str,
        data: dict,
        quantities: tuple = ("ti", "vtor"),
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        xplot: float = 0.48,
        to_plot: tuple = ("profiles", "time"),
        **kwargs,
    ):
        # Keep only channels where fits have been performed
        xdim = "R"
        chans = np.where(
            [
                np.any((data[quantities[0]].sel(channel=ch) > 0).data)
                for ch in data[quantities[0]].channel
            ]
        )[0]
        trans = data[quantities[0]].sel(channel=chans[0]).transform
        R = trans.R.sel(channel=chans)

        # Profiles on R
        title = f"{instrument.upper()} for {self.title}"
        if "title" in kwargs:
            title = kwargs["title"]
        _data = {}
        for quantity in quantities:
            _data[quantity] = (
                data[quantity]
                .sel(channel=chans)
                .assign_coords(R=("channel", R.data))
                .swap_dims({"channel": xdim})
            )
            for dim in _data[quantity].dims:
                assign_datatype(_data[quantity].coords[dim], dim)
            _data[quantity].attrs = data[quantity].attrs

        ylim = (0, None)
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]

        if "profiles" in to_plot:
            for quantity in quantities:
                fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
                y = xr.where(_data[quantity] > 0, _data[quantity], np.nan)

                new_figure(new_fig)
                self._plot_profile_data(
                    y,
                    xdim,
                )
                common_plot_calls(title, True, False, ymin=ylim[0], ymax=ylim[1])
                if save_fig:
                    save_figure(FIG_PATH, f"{fig_name}_profile_data", save_fig=save_fig)

        if "time" in to_plot:
            # Time evolution of central values
            for quantity in quantities:
                new_figure(new_fig)
                fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
                _xplot = _data[quantity].R.sel(R=xplot, method="nearest").data
                _y = _data[quantity].sel(R=_xplot, method="nearest")
                y = xr.where(_y > 0, _y, np.nan)
                xname = y.coords[xdim].long_name

                label = f"{instrument.upper()} {y.long_name}({xname}={_xplot:.2f})"
                if "label" in kwargs is None:
                    label = kwargs["label"]

                self._plot_time_evolution(y, label=label)

                common_plot_calls(title, True, False, 0)
                if save_fig:
                    save_figure(
                        FIG_PATH, f"{fig_name}_tevol_emission", save_fig=save_fig
                    )

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig, fig_name=fig_name)

    def plot_equilibrium(
        self,
        instrument: str,
        data: dict,
        new_fig: bool = True,
        save_fig: bool = False,
        **kwargs,
    ):

        title = f"{instrument.upper()} for {self.title}"
        if "title" in kwargs:
            title = kwargs["title"]
            kwargs.pop("title")

        sci = True
        if "sci" in kwargs:
            sci = kwargs["sci"]
            kwargs.pop("sci")

        quantity = "ipla"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
        y = data[quantity]
        new_figure(new_fig)
        self._plot_time_evolution(y, **kwargs)
        common_plot_calls(title, sci, False, 0)
        if save_fig:
            save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

    def plot_radiation(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        **kwargs,
    ):
        # Brightness vs channel
        quantity = "brightness"
        title = f"{instrument.upper()} for {self.title}"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
        y = data[quantity]

        new_figure(new_fig)
        self._plot_profile_data(
            y,
            **kwargs,
        )
        common_plot_calls(title, True, False, ymin=0)
        if save_fig:
            save_figure(FIG_PATH, f"{fig_name}_profile_data", save_fig=save_fig)

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig)

    def plot_helike_spectroscopy(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        quantities: tuple = ("ti_w", "te_kw", "int_w", "spectra"),
        **kwargs,
    ):
        title = f"{instrument.upper()} for {self.title}"
        if "title" in kwargs:
            title = kwargs["title"]
        ind_temp = np.where([("ti" in q) or ("te" in q) for q in quantities])[0]
        if len(ind_temp) > 0:
            fig_name = f"{self.fig_name}_{instrument.upper()}_Ti_Te"
            new_figure(new_fig)
            quant_temp = [quantities[i] for i in ind_temp]
            for quantity in quant_temp:
                y = data[quantity]
                label = f"{instrument.upper()} {y.long_name}"
                if "label" in kwargs is None:
                    label = kwargs["label"]
                self._plot_time_evolution(y, label=label)
            common_plot_calls(title, True, False, 0)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig)

    def plot_diode_filters(
        self,
        data: dict,
    ):
        raise NotImplementedError

    def plot_interferometry(
        self,
        instrument: str,
        data: dict,
        plot_transform: bool = False,
        new_fig: bool = True,
        save_fig: bool = False,
        **kwargs,
    ):
        quantity = "ne"
        title = f"{instrument.upper()} for {self.title}"
        fig_name = f"{self.fig_name}_{instrument.upper()}_{quantity}"
        y = data[quantity]
        new_figure(new_fig)
        self._plot_time_evolution(y, **kwargs)
        common_plot_calls(title, True, False, 0)
        if save_fig:
            save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

        if plot_transform:
            data[quantity].transform.plot(save_fig=save_fig)

    def plot_plasma_attribute(
        self,
        attribute: str,
        plasma: Plasma,
        new_fig: bool = True,
        save_fig: bool = False,
        ylog: bool = False,
        sci: bool = True,
        to_plot: tuple = ("profiles", "time"),
        element: str = "",
        **kwargs,
    ):
        title = ""
        if len(element) > 0:
            title += f"{element[0].upper() + element[1:]} "
        title += f"{attribute.upper()} for {self.title}"
        fig_name = f"{self.fig_name}_{attribute}"

        # Profiles on rhop
        xdim = "rhop"
        attr = attribute.split(":")
        _y = deepcopy(getattr(plasma, attr[0]))

        if type(_y) is dict:
            _y = _y[element]

        if "element" in _y.dims:
            if len(element) > 0:
                elem = element
                fig_name += f"{elem}"
                _y = _y.sel(element=elem)
            else:
                _y = _y.sum("element")

        if "profiles" in to_plot:
            new_figure(new_fig)
            if "ion_charge" in _y.dims:
                for q in _y.ion_charge:
                    y = _y.sel(ion_charge=q)
                    self._plot_profile(y, xdim=xdim, **kwargs)
            else:
                y = xr.where(_y > 0, _y, np.nan)
                self._plot_profile(y, xdim=xdim, **kwargs)
            common_plot_calls(title, sci, ylog, 0, None, 0, 1)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_profile", save_fig=save_fig)

        # Time evolution of central values
        if "time" in to_plot:
            new_figure(new_fig)
            for xplot in RHOP_VALS:
                xname = y.coords[xdim].long_name
                y_rhop = y.sel(rhop=float(xplot), method="nearest")
                label = f"Plasma {y.long_name}({xname}={xplot})"
                if "label" in kwargs is None:
                    label = kwargs["label"]
                self._plot_time_evolution(y_rhop, label=label)
            common_plot_calls(title, sci, False, 0)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol", save_fig=save_fig)

        # Time evolution of 1D values
        if "integral" in to_plot:
            y = _y
            new_figure(new_fig)
            self._plot_time_evolution(y, **kwargs)
            common_plot_calls(title, sci, False, 0)
            if save_fig:
                save_figure(FIG_PATH, f"{fig_name}_tevol_integral", save_fig=save_fig)

    def within_tolerance(self, y, t):
        try:
            _t = y.t.sel(t=t, method="nearest", tolerance=self.ttol)
            return _t
        except Exception as e:
            print(f"Skipping t={t:.3f} - beyond tolerance. \n --> error: {e}")
            return None

    def populate_kwargs(
        self,
        kwargs,
        label: str = None,
        color=None,
        alpha: float = 0.8,
        marker: str = None,
        linestyle: str = "solid",
    ):
        if label is not None:
            kwargs["label"] = label
        if color is not None:
            kwargs["color"] = color
        if alpha is not None:
            kwargs["alpha"] = alpha
        if marker is not None:
            kwargs["marker"] = marker
        if linestyle is not None:
            kwargs["linestyle"] = linestyle


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
    ymin: float = None,
    ymax: float = None,
    xmin: float = None,
    xmax: float = None,
):
    plt.legend()
    plt.title(title)

    if sci:
        set_axis_sci()
    if ylog:
        plt.yscale("log")

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)


def new_figure(new_fig):
    if new_fig:
        plt.figure()
