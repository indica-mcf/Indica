import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.models.diode_filters import BremsstrahlungDiode
from indica.operators import tomo_1D
from indica.operators.tomo_1D import SXR_tomography
import indica.physics as ph
from indica.readers.read_st40 import ReadST40
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.workflows.fit_ts_rshift import fit_ts


def calculate_zeff(
    pulse,
    tstart=0.03,
    tend=0.1,
    dt=0.01,
    reg_level_guess: float = 0.3,
    fit_Rshift: bool = True,
    plot: bool = True,
    nplot: int = 2,
    verbose: bool = False,
):
    print("Reading data")
    st40 = ReadST40(pulse, tstart, tend, dt)
    st40(["pi", "tws_c", "ts", "efit"])

    print("Fitting TS data")
    te_data = st40.raw_data_trange["ts"]["te"]
    te_err = st40.raw_data_trange["ts"]["te"].error
    ne_data = st40.raw_data_trange["ts"]["ne"]
    ne_err = st40.raw_data_trange["ts"]["ne"].error
    time = te_data.t
    te_fit, ne_fit = fit_ts(
        te_data, te_err, ne_data, ne_err, fit_Rshift=fit_Rshift, verbose=verbose
    )

    print("Calculate PI spectral integral for Bremsstrahlung calculation")
    spectra = st40.raw_data["pi"]["spectra"].interp(t=time)
    (
        zeff_los_avrg,
        filter_data,
        filter_model,
        spectra_to_integrate,
    ) = calculate_zeff_los_averaged(spectra, te_fit, ne_fit)

    print("Calculating Zeff profile from Bremss inversion (including error)")
    zeff_profile, tomo = calculate_zeff_profile(
        te_fit,
        ne_fit,
        filter_data,
        filter_model,
        reg_level_guess,
    )

    if plot:
        plot_results(
            pulse,
            spectra_to_integrate,
            zeff_los_avrg,
            zeff_profile,
            te_data,
            ne_data,
            te_fit,
            ne_fit,
            tomo,
            nplot=nplot,
        )

    return zeff_los_avrg, zeff_profile, spectra_to_integrate


def calculate_zeff_los_averaged(
    spectra: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_wavelength=531.7,
    filter_fwhm=0.6,
    dR_limit: float = 0.01,
):
    # Interpolating Te and Ne on PI line of sight
    los_transform = spectra.transform
    spectra.attrs["rho"], _ = los_transform.convert_to_rho_theta(t=te_fit.t)
    spectra.rho.coords["channel"] = spectra.channel

    dl = los_transform.dl
    los_length = (xr.where(np.isfinite(spectra.rho), 1, 0) * dl).sum("los_position")
    los_length.coords["channel"] = spectra.channel

    # Instatiating filter model, integrating spectra and calculating LOS-avrg
    filter_model = BremsstrahlungDiode(
        "pi", filter_wavelength=filter_wavelength, filter_fwhm=filter_fwhm
    )
    filter_model.set_los_transform(los_transform)
    spectra_to_integrate, filter_data_los_int = filter_model.integrate_spectra(
        spectra, fit_background=False
    )
    filter_data_los_avrg = filter_data_los_int / los_length

    # Calculating Bremsstrahlung --> Zeff conversion and LOS-avrg
    te_along_los = xr.where(
        spectra.rho <= 1.0, te_fit.interp(rho_poloidal=spectra.rho), np.nan
    )
    ne_along_los = xr.where(
        spectra.rho <= 1.0, ne_fit.interp(rho_poloidal=spectra.rho), np.nan
    )
    factor = ph.zeff_bremsstrahlung(
        te_along_los,
        ne_along_los,
        filter_model.filter_wavelength,
        bremsstrahlung=xr.full_like(te_along_los, 1.0),
        gaunt_approx="callahan",
    )
    factor_los_int = factor.sum("los_position") * dl
    factor_los_avrg = factor_los_int / los_length

    # Calculate Zeff los-average
    zeff_los_avrg = filter_data_los_avrg * factor_los_avrg
    if "beamlet" in zeff_los_avrg.dims:
        zeff_los_avrg = zeff_los_avrg.mean("beamlet")

    # Use only channels viewing far from central column and LFS boundary
    Rlfs = spectra.transform.equilibrium.rmjo.interp(rho_poloidal=1, t=spectra.t)
    Rhfs = spectra.transform.equilibrium.rmji.interp(rho_poloidal=1, t=spectra.t)
    Rimpact = spectra.transform.impact_parameter.R
    if "beamlet" in Rimpact.dims:
        Rimpact = Rimpact.mean("beamlet")
    good_channels = ((Rimpact - dR_limit) > Rhfs) * ((Rimpact + dR_limit) < Rlfs)
    good_channels.coords["channel"] = spectra.channel
    zeff_los_avrg = xr.where(good_channels, zeff_los_avrg, np.nan)

    return zeff_los_avrg, filter_data_los_int, filter_model, spectra_to_integrate


def calculate_zeff_profile(
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_data: DataArray,
    filter_model: BremsstrahlungDiode,
    reg_level_guess: float = 0.3,
    zeff_max: float = 10.0,
):
    # Invert Bremsstrahlung emission to get local values
    has_data = np.isfinite(filter_data)
    data_to_invert = filter_data.where(has_data, drop=True)
    channels = data_to_invert.channel
    has_data = [True] * len(channels)
    rho_equil = filter_model.los_transform.equilibrium.rho
    input_dict = dict(
        brightness=data_to_invert.data,
        brightness_error=data_to_invert.data * 0.1,
        t=data_to_invert.t.data,
        dl=filter_model.los_transform.dl,
        R=filter_model.los_transform.R.mean("beamlet").values,
        z=filter_model.los_transform.z.mean("beamlet").values,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        has_data=has_data,
        debug=False,
    )
    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)
    tomo()

    # Calculate local Zeff & uncertainty propagating inversion local emission error
    emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    _error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    emissivity.attrs["error"] = _error

    wlnght = filter_model.filter_wavelength
    _te = te_fit.interp(rho_poloidal=emissivity.rho_poloidal)
    _ne = ne_fit.interp(rho_poloidal=emissivity.rho_poloidal)
    _emiss = emissivity
    _zeff = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=_emiss,
        gaunt_approx="callahan",
    )
    _zeff_lo = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=emissivity - emissivity.error,
        gaunt_approx="callahan",
    )
    _zeff_up = ph.zeff_bremsstrahlung(
        _te,
        _ne,
        wlnght,
        bremsstrahlung=emissivity + emissivity.error,
        gaunt_approx="callahan",
    )

    # Filter data for Zeff < zeff_max
    zeff_profile = xr.where(_zeff < zeff_max, _zeff, np.nan)
    zeff_lo = xr.where(_zeff_lo < zeff_max, _zeff_lo, np.nan)
    zeff_up = xr.where(_zeff_up < zeff_max, _zeff_up, np.nan)
    zeff_error = zeff_up - zeff_lo

    # Filter data for rho < max rho impact of LOS
    # filter_model.los_transform
    rho_max = filter_model.los_transform.impact_rho.max("channel") - 0.1
    zeff_profile.attrs["error"] = xr.where(
        zeff_profile.rho_poloidal < rho_max, zeff_error, zeff_profile
    )

    return zeff_profile, tomo


def plot_results(
    pulse: int,
    spectra_to_integrate: DataArray,
    zeff_los_avrg: DataArray,
    zeff_profile: DataArray,
    te_data: DataArray,
    ne_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    tomo: SXR_tomography,
    nplot: int = 2,
):
    # pathname = "./plots/"
    cm, cols = set_plot_colors()
    los_transform = spectra_to_integrate.transform

    los_transform.plot()

    time = zeff_profile.t
    cols = cm(np.linspace(0.1, 0.75, len(time), dtype=float))
    plt.figure()
    for i, t in enumerate(time.values):
        if not (i % nplot):
            zeff_profile.sel(t=t).plot(color=cols[i], label=f"t={int(t*1.e3)} ms")
            plt.fill_between(
                zeff_profile.rho_poloidal,
                (zeff_profile - zeff_profile.error).sel(t=t),
                (zeff_profile + zeff_profile.error).sel(t=t),
                color=cols[i],
                alpha=0.5,
            )
    plt.ylim(0.5, 8)
    plt.ylabel("Zeff")
    plt.legend()
    plt.title(f"{pulse} Zeff profile")
    plt.legend()
    plt.ylim(
        0.5,
    )
    ylim = plt.ylim()
    if ylim[1] > 8:
        plt.ylim(0.5, 8)

    # Select only channels with impact parameter inside the separatrix by 1 cm
    plt.figure()
    _mean = zeff_los_avrg.mean("channel", skipna=True)
    _std = zeff_los_avrg.std("channel", skipna=True)
    _mean.plot(marker="o")
    plt.fill_between(_mean.t, _mean - _std, _mean + _std, alpha=0.5)
    plt.ylabel("Zeff")
    plt.xlabel("Time (s)")
    plt.title(f"{pulse} Zeff LOS- & channel-averaged")
    plt.ylim(
        0.5,
    )
    ylim = plt.ylim()
    if ylim[1] > 8:
        plt.ylim(0.5, 8)

    plt.figure()
    for i, t in enumerate(time.values):
        if not (i % nplot):
            _Rshift = int(ne_fit.Rshift.sel(t=t) * 100)
            plt.errorbar(
                ne_data.rho.sel(t=t),
                ne_data.sel(t=t),
                ne_data.error.sel(t=t),
                color=cols[i],
                marker="o",
                label=rf"t={int(t*1.e3)} ms $\delta$R={_Rshift} cm",
                alpha=0.6,
            )
            ne_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Ne (m${-3}$)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Ne data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(ne_data) * 1.1)
    plt.legend()

    plt.figure()
    for i, t in enumerate(time.values):
        if not (i % nplot):
            _Rshift = int(te_fit.Rshift.sel(t=t) * 100)
            plt.errorbar(
                te_data.rho.sel(t=t),
                te_data.sel(t=t),
                te_data.error.sel(t=t),
                color=cols[i],
                marker="o",
                label=rf"t={int(t*1.e3)} ms $\delta$R={_Rshift} cm",
                alpha=0.6,
            )
            te_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Te (eV)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Te data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(te_data) * 1.1)
    plt.legend()

    plt.figure()
    central_channel = spectra_to_integrate.channel[
        los_transform.impact_rho.mean("t").argmin()
    ].values
    R_impact = los_transform.impact_parameter.R.sel(channel=central_channel).values
    _spectra = spectra_to_integrate.sel(channel=central_channel)
    for i, t in enumerate(time.values):
        if not (i % nplot):
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

    plt.figure()
    spectra_mean = spectra_to_integrate.sel(channel=central_channel).mean("wavelength")
    spectra_std = spectra_to_integrate.sel(channel=central_channel).std("wavelength")
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
    plt.title(rf"{pulse} Spectra integrated over $\lambda$")
    set_axis_sci()
    plt.legend()

    tomo.show_reconstruction()


if __name__ == "__main__":
    plt.ioff()
    calculate_zeff(
        11314,
        tstart=0.03,
        tend=0.14,
        dt=0.01,
        reg_level_guess=0.3,
        fit_Rshift=True,
        plot=True,
    )
    plt.show()
