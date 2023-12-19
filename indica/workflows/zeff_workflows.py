from copy import deepcopy

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
    verbose: bool = False,
):
    print("Reading experimental data")
    st40 = ReadST40(pulse, tstart, tend, dt)
    st40(["pi", "tws_c", "ts", "efit"])
    te_data = st40.raw_data["ts"]["te"].sel(t=slice(tstart, tend))
    te_err = st40.raw_data["ts"]["te"].error.sel(t=slice(tstart, tend))
    ne_data = st40.raw_data["ts"]["ne"].sel(t=slice(tstart, tend))
    ne_err = st40.raw_data["ts"]["ne"].error.sel(t=slice(tstart, tend))
    te_data.attrs["error"] = te_err
    ne_data.attrs["error"] = ne_err

    print("Fitting TS data")
    te_fit, ne_fit, te_Rshift, ne_Rshift = fit_ts(
        te_data, te_err, ne_data, ne_err, fit_Rshift=fit_Rshift, verbose=verbose
    )
    time = te_fit.t
    ts_rho, _ = ne_data.transform.convert_to_rho_theta(t=time)
    te_data.attrs["rho"] = ts_rho
    ne_data.attrs["rho"] = ts_rho
    te_data.attrs["Rshift"] = te_Rshift
    ne_data.attrs["Rshift"] = ne_Rshift

    print("Calculate PI spectral integral for Bremsstrahlung calculation")
    spectra = st40.raw_data["pi"]["spectra"].interp(t=time)
    _spectra_rho, _ = spectra.transform.convert_to_rho_theta(t=time)
    spectra_rho = deepcopy(_spectra_rho)
    spectra_rho.coords["channel"] = spectra.channel
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
            spectra,
            spectra_to_integrate,
            zeff_los_avrg,
            zeff_profile,
            te_data,
            ne_data,
            te_fit,
            ne_fit,
            tomo,
        )

    return zeff_los_avrg, zeff_profile, spectra


def calculate_zeff_los_averaged(
    spectra: DataArray, te_fit: DataArray, ne_fit: DataArray
):
    # Interpolating Te and Ne on PI line of sight
    _pi_rho, _ = spectra.transform.convert_to_rho_theta(t=te_fit.t)
    pi_rho = deepcopy(_pi_rho)
    pi_rho.coords["channel"] = spectra.channel
    te_along_los = xr.where(pi_rho <= 1.0, te_fit.interp(rho_poloidal=pi_rho), np.nan)
    ne_along_los = xr.where(pi_rho <= 1.0, ne_fit.interp(rho_poloidal=pi_rho), np.nan)

    # Instatiating filter model and integrating spectra
    filter_model = BremsstrahlungDiode("pi", filter_wavelength=531.7, filter_fwhm=0.6)
    filter_model.set_los_transform(spectra.transform)
    spectra_to_integrate, filter_data = filter_model.integrate_spectra(
        spectra, fit_background=False
    )

    # Calculating Bremsstrahlung --> Zeff conversion along LOS & LOS-integral
    factor = ph.zeff_bremsstrahlung(
        te_along_los,
        ne_along_los,
        filter_model.filter_wavelength,
        bremsstrahlung=xr.full_like(te_along_los, 1.0),
        gaunt_approx="callahan",
    )

    factor_los_int = factor.sum("los_position") * filter_model.los_transform.dl
    los_length = (
        xr.where(np.isfinite(pi_rho), 1, 0) * filter_model.los_transform.dl
    ).sum("los_position")
    los_length.coords["channel"] = spectra.channel
    zeff_los_avrg = filter_data * factor_los_int / los_length**2
    if "beamlet" in zeff_los_avrg.dims:
        zeff_los_avrg = zeff_los_avrg.mean("beamlet")

    return zeff_los_avrg, filter_data, filter_model, spectra_to_integrate


def calculate_zeff_profile(
    te_fit: DataArray,
    ne_fit: DataArray,
    filter_data: DataArray,
    filter_model: BremsstrahlungDiode,
    reg_level_guess: float = 0.3,
):
    has_data = np.isfinite(filter_data) * (filter_data > 0)
    data_to_invert = filter_data.where(has_data, drop=True)
    channels = data_to_invert.channel
    has_data = [True] * len(channels)
    rho_equil = filter_model.los_transform.equilibrium.rho
    input_dict = dict(
        brightness=data_to_invert.data,
        brightness_error=data_to_invert.data * 0.1,
        t=data_to_invert.t.data,
        dl=filter_model.los_transform.dl,
        R=filter_model.los_transform.R.mean("beamlet"),
        z=filter_model.los_transform.z.mean("beamlet"),
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
    zeff_profile = xr.where(_zeff < 10, _zeff, np.nan)
    zeff_lo = xr.where(_zeff_lo < 10, _zeff_lo, np.nan)
    zeff_up = xr.where(_zeff_up < 10, _zeff_up, np.nan)
    zeff_profile.attrs["error"] = zeff_up - zeff_lo

    return zeff_profile, tomo


def plot_results(
    pulse: int,
    spectra: DataArray,
    spectra_to_integrate: DataArray,
    zeff_los_avrg: DataArray,
    zeff_profile: DataArray,
    te_data: DataArray,
    ne_data: DataArray,
    te_fit: DataArray,
    ne_fit: DataArray,
    tomo: SXR_tomography,
):
    # pathname = "./plots/"
    cm, cols = set_plot_colors()

    time = zeff_profile.t
    cols = cm(np.linspace(0.1, 0.75, len(time), dtype=float))
    plt.figure()
    for i, t in enumerate(time.values):
        if i % 2:
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
    Rlfs = spectra.transform.equilibrium.rmjo.interp(rho_poloidal=1, t=time)
    Rhfs = spectra.transform.equilibrium.rmji.interp(rho_poloidal=1, t=time)
    Rimpact = spectra.transform.impact_parameter.R
    if "beamlet" in Rimpact.dims:
        Rimpact = Rimpact.mean("beamlet")
    good_channels = ((Rimpact - 0.01) > Rhfs) * ((Rimpact + 0.01) < Rlfs)
    good_channels.coords["channel"] = spectra.channel
    _mean = xr.where(good_channels, zeff_los_avrg, np.nan).mean("channel", skipna=True)
    _std = xr.where(good_channels, zeff_los_avrg, np.nan).std("channel", skipna=True)
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
        if i % 2:
            _Rshift = int(ne_data.Rshift.sel(t=t) * 100)
            plt.errorbar(
                ne_data.rho.sel(t=t),
                ne_data.sel(t=t),
                ne_data.error.sel(t=t),
                color=cols[i],
                marker="o",
                label=rf"t={int(t*1.e3)} ms $\delta$R={_Rshift} cm",
                alpha=0.6,
            )
    for i, t in enumerate(time.values):
        if i % 2:
            ne_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Ne (m${-3}$)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Ne data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(ne_data) * 1.1)
    plt.legend()

    plt.figure()
    for i, t in enumerate(time.values):
        if i % 2:
            _Rshift = int(te_data.Rshift.sel(t=t) * 100)
            plt.errorbar(
                te_data.rho.sel(t=t),
                te_data.sel(t=t),
                te_data.error.sel(t=t),
                color=cols[i],
                marker="o",
                label=rf"t={int(t*1.e3)} ms $\delta$R={_Rshift} cm",
                alpha=0.6,
            )
    for i, t in enumerate(time.values):
        if i % 2:
            te_fit.sel(t=t).plot(color=cols[i], linewidth=4, zorder=0)
    plt.ylabel("Te (eV)")
    plt.xlabel("Rho-poloidal")
    plt.title(f"{pulse} TS Te data & fits")
    plt.xlim(0, 1.1)
    plt.ylim(0, np.nanmax(te_data) * 1.1)
    plt.legend()

    plt.figure()
    for i, t in enumerate(time.values):
        if i % 2:
            spectra_mean = spectra_to_integrate.mean("channel")
            spectra_std = spectra_to_integrate.std("channel")
            plt.fill_between(
                spectra_mean.wavelength,
                (spectra_mean - spectra_std).sel(t=t),
                (spectra_mean + spectra_std).sel(t=t),
                color=cols[i],
                label=f"t={int(t*1.e3)} ms",
                alpha=0.5,
            )
    plt.ylabel("(...)")
    plt.xlabel("Wavelength (nm)")
    plt.title(f"{pulse} PI spectra integrated")
    set_axis_sci()
    plt.legend()

    tomo.show_reconstruction()
