import matplotlib.pylab as plt
import numpy as np

from indica.defaults.load_defaults import load_default_objects
from indica.readers.modelreader import ModelReader
from indica.utilities import set_plot_colors
from indica.workflows.fit_ts_rshift import fit_ts

PLASMA = load_default_objects("st40", "plasma")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
GEOMETRY = load_default_objects("st40", "geometry")
LOS_TRANSFORM = GEOMETRY["sxrc_xy2"]
LOS_TRANSFORM.set_equilibrium(EQUILIBRIUM)
PLASMA.set_equilibrium(EQUILIBRIUM)


def example_fit_ts(
    machine: str = "st40",
    fit_R_shift: bool = True,
    verbose: bool = False,
    nplot: int = 2,
):

    cm, cols = set_plot_colors()

    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    _reader = ModelReader(machine, instruments=["ts"])
    _reader.set_geometry_transforms(transforms["ts"])
    plasma.set_equilibrium(equilibrium)
    _reader.set_plasma(plasma)
    raw_data = _reader()
    fit_R_shift = False

    te_data = raw_data["ts"]["te"]
    te_err = raw_data["ts"]["te"].error
    ne_data = raw_data["ts"]["ne"]
    ne_err = raw_data["ts"]["ne"].error
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
    plt.title(f"TS Ne data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(ne_fit) * 1.2)
    plt.legend()

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
    plt.title(f"TS Te data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(te_fit) * 1.2)
    plt.legend()

    return te_data, ne_data, te_fit, ne_fit
