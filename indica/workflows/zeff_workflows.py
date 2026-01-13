import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica import Plasma
from indica.defaults.load_defaults import load_default_objects
from indica.models import BremsstrahlungDiode
from indica.models import ThomsonScattering
from indica.operators import tomo_1D
from indica.operators.tomo_1D import SXR_tomography
import indica.physics as ph
from indica.readers.modelreader import ModelReader
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors

CMAP, COLORS = set_plot_colors()

PLASMA = load_default_objects("st40", "plasma")
TRANSFORMS = load_default_objects("st40", "geometry")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")


def example_zeff_bremstrahlung(
    plot: bool = True,
    nplot: int = 2,
    save_fig: bool = False,
):
    """
    Test workflows to calculate Zeff LOS-averaged and profile using
    Bremsstrahlung radiation measurement from passive spectroscopy.
    """

    models = {"ts": ThomsonScattering, "lines": BremsstrahlungDiode}
    TRANSFORMS["lines"] = TRANSFORMS["pi"]  # Use multi-LOS spectrometer transform
    model_reader = ModelReader(models)
    model_reader.set_plasma(PLASMA)
    model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
    bckc = model_reader()

    filter_data = bckc["lines"]["brightness"]
    filter_data = filter_data.assign_coords(
        error=(filter_data.dims, (filter_data * 0.05).data)
    )

    print("Calculate LOS-averaged Zeff")
    zeff_los_avrg = calculate_zeff_los_averaged(
        filter_data,
        PLASMA.electron_temperature,
        PLASMA.electron_density,
        models["lines"].filter_wavelength,
    )

    print("Calculating Zeff profile from Bremss inversion (including error)")
    zeff_profile, tomo = calculate_zeff_profile(
        filter_data,
        PLASMA.electron_temperature,
        PLASMA.electron_density,
        models["lines"].filter_wavelength,
    )
    # Add phantom emissivity to tomo class for plotting purposes
    tomo.expected_emissivity = models["lines"].emissivity

    if plot:
        plot_results(
            0,
            filter_data,
            zeff_los_avrg,
            zeff_profile,
            tomo,
            nplot=nplot,
            save_fig=save_fig,
            PLASMA=PLASMA,
        )

    return zeff_los_avrg, zeff_profile, filter_data, tomo


def calculate_zeff_los_averaged(
    filter_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_wavelength: float,
):
    """
    Calculate LOS-averaged Zeff any number of independent LOS

    filter_data
        Filtered diode output given the input spectra.
    te_fit
        Te profile (pon rhop)
    ne_fit
        Ne profile (pon rhop)
    filter_wavelength
        Centroid of the interference filter
    """

    los_transform = filter_data.transform
    factor_profile = ph.zeff_bremsstrahlung(
        te_fit,
        ne_fit,
        filter_wavelength,
        bremsstrahlung=xr.full_like(te_fit, 1.0),
        gaunt_approx="callahan",
    )
    factor_los_int = los_transform.integrate_on_los(factor_profile, t=filter_data.t)
    zeff_los_avrg = filter_data * factor_los_int / los_transform.los_length**2

    return zeff_los_avrg


def calculate_zeff_profile(
    filter_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_wavelength: float,
):
    """
    Calculate Zeff profile from multiple LOS Bremsstrahlung brightness measurement
    using 1D inversion routine + Ne and Te profiles

    filter_data
        Filtered diode output given the input spectra.
    te_fit
        Te profile (pon rhop)
    ne_fit
        Ne profile (pon rhop)
    filter_wavelength
        Centroid of the interference filter
    """
    data = filter_data.values
    t = filter_data.t.values
    error = filter_data.error.values

    los_transform = filter_data.transform
    R = los_transform.R.mean("beamlet").values
    z = los_transform.z.mean("beamlet").values
    has_data = [True] * filter_data.shape[1]
    rho_equil = los_transform.equilibrium.rhop

    input_dict = dict(
        brightness=data,
        brightness_error=error,
        t=t,
        dl=los_transform.dl,
        R=R,
        z=z,
        rho_equil=dict(
            R=rho_equil.R.values,
            z=rho_equil.z.values,
            t=rho_equil.t.values,
            rho=rho_equil.values,
        ),
        has_data=has_data,
        debug=False,
    )
    tomo = tomo_1D.SXR_tomography(
        input_dict,
    )
    tomo_result = tomo()

    coords = [
        ("t", tomo_result["t"]),
        ("rhop", tomo_result["profile"]["rhop"][0, :]),
    ]
    emissivity = DataArray(
        tomo_result["profile"]["sym_emissivity"],
        coords=coords,
    )
    _error = DataArray(
        tomo_result["profile"]["sym_emissivity_err"],
        coords=coords,
    )
    emissivity = emissivity.assign_coords(error=(emissivity.dims, _error.data))

    wlnght = filter_wavelength
    _te = te_fit.interp(rhop=emissivity.rhop)
    _ne = ne_fit.interp(rhop=emissivity.rhop)
    zeff_profile = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=emissivity,
        gaunt_approx="callahan",
    )
    zeff_lo = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=emissivity - emissivity.error,
        gaunt_approx="callahan",
    )
    zeff_up = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=emissivity + emissivity.error,
        gaunt_approx="callahan",
    )
    zeff_profile = zeff_profile.assign_coords(
        error=(zeff_profile.dims, np.abs(zeff_up - zeff_lo).data)
    )
    return zeff_profile, tomo


def plot_results(
    pulse: int,
    filter_data,
    zeff_los_avrg: DataArray,
    zeff_profile: DataArray,
    tomo: SXR_tomography,
    nplot: int = 2,
    save_fig: bool = False,
    fig_path: str = None,
    PLASMA: Plasma = None,
):
    los_transform = filter_data.transform

    if fig_path is None:
        fig_path = FIG_PATH

    cm, cols = set_plot_colors()

    los_transform.plot(fig_path=fig_path, fig_name=f"{pulse}_", save_fig=save_fig)

    time = zeff_profile.t
    cols = cm(np.linspace(0.1, 0.75, len(time), dtype=float))
    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue

        zeff_profile.sel(t=t).plot(color=cols[i], label=f"t={int(t*1.e3)} ms")
        plt.fill_between(
            zeff_profile.rhop,
            (zeff_profile - zeff_profile.error).sel(t=t),
            (zeff_profile + zeff_profile.error).sel(t=t),
            color=cols[i],
            alpha=0.5,
        )
        if PLASMA is not None:
            PLASMA.zeff.sum("element").sel(t=t).plot(color=cols[i], linestyle="dashed")
    if PLASMA is not None:
        PLASMA.zeff.sum("element").sel(t=time[0]).plot(
            color=cols[0], linestyle="dashed", label="Phantom"
        )
    plt.ylabel("Zeff")
    plt.legend()
    plt.title(f"{pulse} Zeff profile")
    plt.legend()
    plt.ylim(0.5, zeff_profile.sel(rhop=slice(0, 0.5)).max())
    save_figure(fig_path, f"{pulse}_zeff_profile_PI_inversion", save_fig=save_fig)

    # Select only channels with impact parameter inside the separatrix by 1 cm
    plt.figure()
    _mean = zeff_los_avrg.mean("channel", skipna=True)
    _lo = zeff_los_avrg.min("channel", skipna=True)
    _up = zeff_los_avrg.max("channel", skipna=True)
    _mean.plot(marker="o")
    plt.fill_between(_mean.t, _lo, _up, alpha=0.5)
    plt.ylabel("Zeff")
    plt.xlabel("Time (s)")
    plt.title(f"{pulse} Zeff LOS- & channel-averaged")
    plt.ylim(
        0.5,
    )
    save_figure(fig_path, f"{pulse}_zeff_LOS_and_channel-avrg", save_fig=save_fig)

    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        los_transform.calc_impact_parameter()
        R = los_transform.impact_parameter["R"].mean("t")
        to_plot = (
            filter_data.sel(t=t)
            .assign_coords(R=("channel", R.data))
            .swap_dims({"channel": "R"})
        )
        to_plot_err = filter_data.error.sel(t=t)
        plt.plot(
            R,
            to_plot,
            color=cols[i],
            marker="o",
            label=f"t={int(t*1.e3)} ms",
            alpha=0.5,
        )
        plt.errorbar(
            R,
            to_plot,
            to_plot_err,
            color=cols[i],
            alpha=0.5,
        )
        plt.scatter(
            R,
            tomo.backprojection[i, :],
            color=cols[i],
            marker="x",
            linewidths=3,
        )

    plt.scatter(
        R,
        tomo.backprojection[0, :],
        color=cols[0],
        marker="x",
        linewidths=3,
        label="Back-calculated",
    )
    plt.ylabel("Brightness (W/m$^{2}$)")
    plt.xlabel("Channel")
    plt.title("")
    plt.legend()
    set_axis_sci()
    save_figure(fig_path, f"{pulse}_bremsstrahlung_brightness", save_fig=save_fig)

    tomo.show_reconstruction()


if __name__ == "__main__":
    plt.ioff()
    _ = example_zeff_bremstrahlung()
    plt.show()
