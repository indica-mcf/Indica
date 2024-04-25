import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.operators.spline_fit_R_shift import fit_profile_and_R_shift
from indica.readers.read_st40 import ReadST40
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_plot_colors


def fit_ts(
    te_data: DataArray,
    te_err: DataArray,
    ne_data: DataArray,
    ne_err: DataArray,
    fit_R_shift: bool = True,
    verbose: bool = False,
):
    """
    Fit TS data including (or not) an ad-hoc R_shift of scattering volume positions
    (that can also be interpreted as a shift in the equilibrium)

    R_shift calculated only for Te and then applied to Ne!!!!

    Parameters
    ----------
    te_data - electron temperature data
    te_err - electron temperature error
    ne_data - electron density data
    ne_err - electron density data
    fit_R_shift - True if R_shift is to be fitted

    Returns
    -------
    Te and Ne fits, and corresponding R_shift

    """
    if not fit_R_shift:
        R_shift = xr.full_like(te_data.t, 0.0)
    else:
        R_shift = None

    ts_R = te_data.transform.R
    ts_z = te_data.transform.z
    equilibrium = te_data.transform.equilibrium
    print("  Te")
    te_fit, te_R_shift, te_rho = fit_profile_and_R_shift(
        ts_R,
        ts_z,
        te_data,
        te_err,
        xknots=[0, 0.4, 0.85, 0.9, 0.98, 1.1],
        equilibrium=equilibrium,
        R_shift=R_shift,
        verbose=verbose,
    )
    print("  Ne")
    ne_fit, ne_R_shift, ne_rho = fit_profile_and_R_shift(
        ts_R,
        ts_z,
        ne_data,
        ne_err,
        xknots=[0, 0.4, 0.85, 0.9, 0.98, 1.1],
        equilibrium=equilibrium,
        R_shift=te_R_shift,
        verbose=verbose,
    )

    te_fit.attrs["R_shift"] = te_R_shift
    ne_fit.attrs["R_shift"] = ne_R_shift
    te_data.attrs["rho"] = te_rho
    ne_data.attrs["rho"] = ne_rho

    return te_fit, ne_fit


def example_run(
    pulse: int = 11314,
    tstart: float = 0.04,
    tend: float = 0.13,
    dt: float = 0.01,
    fit_R_shift: bool = True,
    verbose: bool = False,
    nplot: int = 2,
    save_fig: bool = False,
):

    cm, cols = set_plot_colors()

    st40 = ReadST40(pulse, tstart=tstart, tend=tend, dt=dt)
    st40(["ts"], set_equilibrium=True)
    te_data = st40.raw_data["ts"]["te"]
    te_err = st40.raw_data["ts"]["te"].error
    ne_data = st40.raw_data["ts"]["ne"]
    ne_err = st40.raw_data["ts"]["ne"].error
    time = te_data.t
    cols = cm(np.linspace(0.1, 0.75, len(time), dtype=float))

    te_fit, ne_fit = fit_ts(
        te_data,
        te_err,
        ne_data,
        ne_err,
        fit_R_shift=fit_R_shift,
        verbose=verbose,
    )

    plt.figure()
    _R_shift = [f"{(ne_fit.R_shift.sel(t=t).values * 100):.1f}" for t in time]
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        plt.errorbar(
            ne_data.rho.sel(t=t),
            ne_data.sel(t=t),
            ne_data.error.sel(t=t),
            color=cols[i],
            marker="o",
            label=rf"t={int(t*1.e3)} ms $\delta$R={_R_shift[i]} cm",
            alpha=0.6,
        )
        ne_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Ne (m${-3}$)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Ne data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(ne_fit) * 1.2)
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"{pulse}_TS_Ne_fits", save_fig=save_fig)

    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        plt.errorbar(
            te_data.rho.sel(t=t),
            te_data.sel(t=t),
            te_data.error.sel(t=t),
            color=cols[i],
            marker="o",
            label=rf"t={int(t*1.e3)} ms $\delta$R={_R_shift[i]} cm",
            alpha=0.6,
        )
        te_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Te (eV)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Te data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(te_fit) * 1.2)
    plt.legend()
    if save_fig:
        save_figure(FIG_PATH, f"{pulse}_TS_Te_fits", save_fig=save_fig)

    return te_data, ne_data, te_fit, ne_fit


if __name__ == "__main__":
    plt.ioff()
    example_run(11312)
    plt.show()
