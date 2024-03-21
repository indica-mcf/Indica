import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.examples.los_transform_examples import tangential_xy
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.plasma import Plasma
from indica.operators import tomo_1D
from indica.operators.tomo_1D import SXR_tomography
import indica.physics as ph
from indica.readers.read_phantoms import PhantomReader
from indica.readers.read_st40 import ReadST40
from indica.utilities import FIG_PATH
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.workflows.fit_ts_rshift import fit_ts


def calculate_zeff(
    pulse=0,
    tstart=0.03,
    tend=0.1,
    dt=0.01,
    filter_wavelength: float = 532.1,
    filter_fwhm: float = 1,
    revisions: dict = None,
    fit_R_shift: bool = True,
    reg_level_guess: float = 0.6,
    plot: bool = True,
    nplot: int = 2,
    save_fig: bool = False,
    dR_limit: float = 0.1,
    default_perc_err: float = 0.05,
):
    """
    Test workflows to calculate Zeff LOS-averaged and profile using CXRS spectra.

    Parameters
    ----------
    pulse
        Pulse to analyse - set to 0 for 100% phantom data.
    tstart
        Start time
    tend
        End time
    dt
        Delta t for time binning of all quantities.

    ...see other parameter definition in separate methods...

    Returns
    -------
    zeff_los_avrg
        LOS averaged Zeff for all LOS separately
    zeff_profile
        Effective charge profile
    filter_data
        Filtered diode output given the input spectra.
    spectra_to_integrate
        Filtered spectra used for the computation of the Bremsstrahlung
    """

    print("Read data")
    plasma = None
    if pulse > 0:
        st40 = ReadST40(pulse, tstart, tend, dt)
        st40(
            ["pi", "tws_c", "ts", "efit"],
            revisions=revisions,
            set_equilibrium=True,
        )
        binned_data = st40.binned_data
    else:
        st40_phantom = PhantomReader(pulse, tstart, tend, dt)
        _los_transform = tangential_xy(st40_phantom._machine_dims)
        st40_phantom.instr_models["pi"].set_los_transform(_los_transform)
        st40_phantom(
            ["pi", "tws_c", "ts", "efit"], revisions=revisions, set_equilibrium=True
        )
        binned_data = st40_phantom.binned_data
        plasma = st40_phantom.plasma

    te_data = binned_data["ts"]["te"]
    te_err = binned_data["ts"]["te"].error
    ne_data = binned_data["ts"]["ne"]
    ne_err = binned_data["ts"]["ne"].error
    if np.all(te_err) == 0:
        te_err = te_data * 0.05
    if np.all(ne_err) == 0:
        ne_err = ne_data * 0.05

    print("Fit TS data")
    te_fit, ne_fit = fit_ts(
        te_data,
        te_err,
        ne_data,
        ne_err,
        fit_R_shift=fit_R_shift,
    )
    equilibrium = te_data.transform.equilibrium

    # TODO: R_shift is time-dependent, but not in equilibrium!!!
    equilibrium.R_offset = te_fit.R_shift.mean("t").values

    print("Interpolate spectra to TS times, and map it to equilibrium")
    if "spectra" in binned_data["pi"]:
        spectra = binned_data["pi"]["spectra"]
        print("Calculate spectral integral for Bremsstrahlung calculation")
        filter_data, spectra_to_integrate, filter_model = filter_passive_spectra(
            spectra,
            filter_wavelength,
            filter_fwhm,
            default_perc_err=default_perc_err,
        )
    else:
        filter_data = binned_data["pi"]["brightness"]
        spectra_to_integrate = None

    print("Calculate LOS-averaged Zeff")
    zeff_los_avrg = calculate_zeff_los_averaged(
        filter_data,
        te_fit,
        ne_fit,
        filter_wavelength,
        dR_limit=dR_limit,
    )

    print("Calculating Zeff profile from Bremss inversion (including error)")
    zeff_profile, tomo = calculate_zeff_profile(
        filter_data,
        te_fit,
        ne_fit,
        filter_wavelength,
        reg_level_guess,
    )

    if plot:
        plot_results(
            pulse,
            filter_data,
            zeff_los_avrg,
            zeff_profile,
            te_data,
            ne_data,
            te_fit,
            ne_fit,
            tomo,
            spectra_to_integrate,
            nplot=nplot,
            save_fig=save_fig,
            plasma=plasma,
        )

    return zeff_los_avrg, zeff_profile, filter_data, spectra_to_integrate, tomo


def filter_passive_spectra(
    spectra: DataArray,
    filter_wavelength,
    filter_fwhm,
    default_perc_err: float = 0.05,
):
    """
    Create object representing a filtered diode measuring Bremsstrahlung and
    integrate spectrometer data to emulate what the diode should be measuring.

    Parameters
    ----------
    spectra
        Passive spectra
    filter_wavelength
        Centroid of the interference filter
    filter_fwhm
        FWHM of the interference filter
    default_perc_err
        Default percentage error of the output if not self-consistently calculated

    Returns
    -------
    filter_data
        Filtered diode output given the input spectra.
    spectra_to_integrate
        Filtered spectra
    filter_model
        Model object used to simulate the fitlered diode

    """
    spectra.transform.convert_to_rho_theta(t=spectra.t)
    filter_model = BremsstrahlungDiode(
        "pi", filter_wavelength=filter_wavelength, filter_fwhm=filter_fwhm
    )
    spectra_to_integrate, filter_data = filter_model.integrate_spectra(
        spectra, fit_background=False
    )
    filter_data.attrs["transform"] = spectra_to_integrate.transform
    if not hasattr(filter_data, "error"):
        filter_data.attrs["error"] = filter_data * default_perc_err
    if np.all(filter_data.error) == 0:
        filter_data.attrs["error"] = filter_data * default_perc_err

    return filter_data, spectra_to_integrate, filter_model


def calculate_zeff_los_averaged(
    filter_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_wavelength: float,
    dR_limit: float = 0.1,
):
    """

    Parameters
    ----------
    filter_data
        Filtered diode output given the input spectra.
    te_fit
        Te profile (pon rho_poloidal)
    ne_fit
        Ne profile (pon rho_poloidal)
    filter_wavelength
        Centroid of the interference filter
    dR_limit
        DIfference between LOS impact parameter and plasma edge (m)
        --> result set to NaN for LOS within dR_limit of the HFS / LFS boundaries.

    Returns
    -------
    LOS averaged Zeff for all LOS separately
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

    if dR_limit > 0:
        equilibrium = los_transform.equilibrium
        Rlfs = equilibrium.rbnd.max("arbitrary_index").interp(t=filter_data.t)
        Rhfs = equilibrium.rbnd.min("arbitrary_index").interp(t=filter_data.t)
        lfs_bound = los_transform.impact_parameter.R + dR_limit
        hfs_bound = los_transform.impact_parameter.R - dR_limit
        good_channels = (hfs_bound > Rhfs) * (lfs_bound < Rlfs)
        good_channels.coords["channel"] = filter_data.channel
        zeff_los_avrg = xr.where(good_channels, zeff_los_avrg, np.nan)

    return zeff_los_avrg


def calculate_zeff_profile(
    filter_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_wavelength: float,
    reg_level_guess: float = 0.6,
):
    """

    Parameters
    ----------
    filter_data
        Filtered diode output given the input spectra.
    te_fit
        Te profile (pon rho_poloidal)
    ne_fit
        Ne profile (pon rho_poloidal)
    filter_wavelength
        Centroid of the interference filter
    reg_level_guess
        Inversion regularisation parameters (larger value --> stiffer profiles)

    Returns
    -------
    zeff_profile
        Effective charge profile
    tomo
        Inversion object
    """
    data = filter_data.values
    t = filter_data.t.values
    error = filter_data.error.values

    los_transform = filter_data.transform
    R = los_transform.R.mean("beamlet").values - los_transform.equilibrium.R_offset
    z = los_transform.z.mean("beamlet").values - los_transform.equilibrium.z_offset
    has_data = [True] * filter_data.shape[1]
    rho_equil = los_transform.equilibrium.rho

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
        reg_level_guess=reg_level_guess,
    )
    tomo_result = tomo()

    coords = [
        ("t", tomo_result["t"]),
        ("rho_poloidal", tomo_result["profile"]["rho_poloidal"][0, :]),
    ]
    emissivity = DataArray(
        tomo_result["profile"]["sym_emissivity"],
        coords=coords,
    )
    _error = DataArray(
        tomo_result["profile"]["sym_emissivity_err"],
        coords=coords,
    )
    emissivity.attrs["error"] = _error

    wlnght = filter_wavelength
    _te = te_fit.interp(rho_poloidal=emissivity.rho_poloidal)
    _ne = ne_fit.interp(rho_poloidal=emissivity.rho_poloidal)
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
    zeff_profile.attrs["error"] = np.abs(zeff_up - zeff_lo)

    return zeff_profile, tomo


def plot_results(
    pulse: int,
    filter_data,
    zeff_los_avrg: DataArray,
    zeff_profile: DataArray,
    te_data: DataArray,
    ne_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    tomo: SXR_tomography,
    spectra_to_integrate: DataArray = None,
    nplot: int = 2,
    save_fig: bool = False,
    fig_path: str = None,
    plasma: Plasma = None,
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
            zeff_profile.rho_poloidal,
            (zeff_profile - zeff_profile.error).sel(t=t),
            (zeff_profile + zeff_profile.error).sel(t=t),
            color=cols[i],
            alpha=0.5,
        )
        if plasma is not None:
            plasma.zeff.sum("element").sel(t=t).plot(color=cols[i], linestyle="dashed")
    if plasma is not None:
        plasma.zeff.sum("element").sel(t=time[0]).plot(
            color=cols[0], linestyle="dashed", label="Phantom"
        )
    plt.ylabel("Zeff")
    plt.legend()
    plt.title(f"{pulse} Zeff profile")
    plt.legend()
    plt.ylim(
        0.5,
    )
    ylim = plt.ylim()
    if ylim[1] > 10:
        plt.ylim(0.5, 10)
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
    ylim = plt.ylim()
    if ylim[1] > 8:
        plt.ylim(0.5, 8)
    save_figure(fig_path, f"{pulse}_zeff_LOS_and_channel-avrg", save_fig=save_fig)

    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        _R_shift = int(ne_fit.R_shift.sel(t=t) * 100)
        plt.errorbar(
            ne_data.rho.sel(t=t),
            ne_data.sel(t=t),
            ne_data.error.sel(t=t),
            color=cols[i],
            marker="o",
            label=rf"t={int(t*1.e3)} ms $\delta$R={_R_shift} cm",
            alpha=0.6,
        )
        ne_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Ne (m${-3}$)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Ne data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(ne_data) * 1.1)
    plt.legend()
    save_figure(fig_path, f"{pulse}_TS_Ne_fits", save_fig=save_fig)

    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        _R_shift = int(te_fit.R_shift.sel(t=t) * 100)
        plt.errorbar(
            te_data.rho.sel(t=t),
            te_data.sel(t=t),
            te_data.error.sel(t=t),
            color=cols[i],
            marker="o",
            label=rf"t={int(t*1.e3)} ms $\delta$R={_R_shift} cm",
            alpha=0.6,
        )
        te_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Te (eV)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Te data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(te_data) * 1.1)
    plt.legend()
    save_figure(fig_path, f"{pulse}_TS_Te_fits", save_fig=save_fig)

    plt.figure()
    for i, t in enumerate(time.values):
        if i % nplot:
            continue
        R = los_transform.impact_parameter.R
        to_plot = (
            filter_data.sel(t=t)
            .assign_coords(R=("channel", R))
            .swap_dims({"channel": "R"})
        )
        to_plot_err = filter_data.error.sel(t=t)
        to_plot.plot(color=cols[i], marker="o", label=f"t={int(t*1.e3)} ms", alpha=0.5)
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

    if spectra_to_integrate is not None:
        plt.figure()
        central_channel = spectra_to_integrate.channel[
            los_transform.impact_rho.mean("t").argmin()
        ].values
        R_impact = los_transform.impact_parameter.R.sel(channel=central_channel).values
        _spectra = spectra_to_integrate.sel(channel=central_channel)
        for i, t in enumerate(time.values):
            if i % nplot:
                continue
            _spectra.sel(t=t).plot(
                color=cols[i],
                label=f"t={int(t*1.e3)} ms",
                alpha=0.8,
            )
        plt.ylabel("Brightness ($W/m^2$)")
        plt.xlabel("Wavelength (nm)")
        plt.title(f"{pulse} Spectra filtered")
        set_axis_sci()
        plt.legend()
        save_figure(fig_path, f"{pulse}_PI_filtered_spectra", save_fig=save_fig)

        # Average/Std of the spectra in the filter wavelength region
        plt.figure()
        spectra_mean = spectra_to_integrate.sel(channel=central_channel).mean(
            "wavelength"
        )
        spectra_std = spectra_to_integrate.sel(channel=central_channel).std(
            "wavelength"
        )
        plt.fill_between(
            spectra_mean.t,
            (spectra_mean - spectra_std),
            (spectra_mean + spectra_std),
            alpha=0.5,
        )
        spectra_mean.plot(
            marker="o", zorder=0, label="Channel @ $R_{impact}$=" + f"{R_impact:.2f} m"
        )
        plt.ylabel("Brightness ($W/m^2$)")
        plt.xlabel("Time (s)")
        plt.ylim(
            0,
        )
        plt.title(rf"{pulse} Spectra averaged over $\lambda$")
        set_axis_sci()
        plt.legend()
        save_figure(
            fig_path, f"{pulse}_PI_Brightness_avrg_central_channel", save_fig=save_fig
        )

        # Integral of the spectra in the filter wavelength region
        plt.figure()
        spectra_int = spectra_to_integrate.sel(channel=central_channel).sum(
            "wavelength"
        )
        spectra_int.plot(
            marker="o", zorder=0, label="Channel @ $R_{impact}$=" + f"{R_impact:.2f} m"
        )
        plt.ylabel("Brightness ($W/m^2$)")
        plt.xlabel("Time (s)")
        plt.ylim(
            0,
        )
        plt.title(rf"{pulse} Spectra integrated over $\lambda$")
        set_axis_sci()
        plt.legend()
        save_figure(
            fig_path, f"{pulse}_PI_Brightness_avrg_central_channel", save_fig=save_fig
        )


if __name__ == "__main__":
    plt.ioff()
    calculate_zeff()
    plt.show()
