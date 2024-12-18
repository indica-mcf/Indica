import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.models import ThomsonScattering
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.operators.tomo_asymmetry import InvertPoloidalAsymmetry
from indica.readers.modelreader import ModelReader
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors
from indica.operators.spline_fit_R_shift import fit_profile_and_R_shift
import xarray as xr

PLASMA = load_default_objects("st40", "plasma")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
TRANSFORMS = load_default_objects("st40", "geometry")
PLASMA.set_equilibrium(EQUILIBRIUM)

NPLOT = 3
CM, COLS = set_plot_colors()

def example_poloidal_asymmetry():
    asymmetry_parameter = centrifugal_asymmetry_parameter(
        PLASMA.ion_density,
        PLASMA.ion_temperature,
        PLASMA.electron_temperature,
        PLASMA.toroidal_rotation,
        PLASMA.meanz,
        PLASMA.zeff,
        PLASMA.main_ion,
    )

    ion_density_2d = centrifugal_asymmetry_2d_map(
        PLASMA.ion_density,
        asymmetry_parameter,
        EQUILIBRIUM,
    )

    return ion_density_2d


def example_tomo_asymmetry(
    instrument: str = "sxrc_xy1",
    asymmetric_profile: bool = True,
    plot: bool = True,
    element: str = "ar",
):

    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry()
    else:
        rho_2d = PLASMA.equilibrium.rhop.interp(t=PLASMA.t)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)

    imp_density_2d = ion_density_2d.sel(element=element).drop_vars("element")
    el_density_2d = PLASMA.electron_density.interp(rhop=imp_density_2d.rhop)
    lz_tot_2d = (
        PLASMA.lz_tot[element].sum("ion_charge").interp(rhop=imp_density_2d.rhop)
    )
    phantom_emission = el_density_2d * imp_density_2d * lz_tot_2d
    los_transform = TRANSFORMS[instrument]
    los_transform.set_equilibrium(EQUILIBRIUM)
    los_integral = los_transform.integrate_on_los(
        phantom_emission, phantom_emission.t.values
    )

    print("\n Running asymmetry inference...")
    invert_asymm = InvertPoloidalAsymmetry()
    bckc_emission, bckc, profile, asymmetry = invert_asymm(
        los_integral,
        los_transform,
    )
    print("...and finished \n")

    if plot:
        plt.ioff()
        los_transform.plot()
        plt.show(block=True)

        for i, t in enumerate(bckc.t):
            if i % np.floor(np.size(bckc.t) / NPLOT) != 0:
                continue
            kwargs_phantom = dict(
                marker="o",
                label=f"Phantom @ t={int(t*1.e3)} ms",
                color="r",
                zorder=1,
                alpha=0.8,
            )
            kwargs_bckc = dict(
                marker="x",
                label=f"Back-calculated @ t={int(t*1.e3)} ms",
                color="b",
                zorder=2,
                alpha=0.8,
            )
            plt.figure()
            plt.errorbar(
                invert_asymm.data.channel,
                invert_asymm.data.sel(t=t, method="nearest"),
                invert_asymm.error.sel(t=t, method="nearest"),
                **kwargs_phantom,
            )
            bckc.attrs = {"long_name": "Brightness", "units": "W/m$^2$"}
            bckc.sel(t=t).plot(**kwargs_bckc)
            plt.title(f"LOS integrals @ t={int(t*1.e3)} ms")
            plt.legend()
            set_axis_sci()

            plt.figure()
            phantom_emission.attrs = {"long_name": "Emissivity", "units": "W/m$^3$"}
            phantom_emission.sel(t=t).plot()
            plt.title(f"2D Phantom @ t={int(t*1.e3)} ms")
            plt.axis("equal")

            # plt.figure()
            # residuals = (bckc_emission - phantom_emission)/phantom_emission * 100
            # residuals.attrs = {"long_name": "(Phantom - Bckc)/Phantom", "units": "%"}
            # residuals = xr.where(residuals < 100, residuals, np.nan)
            # residuals.sel(t=t).plot()
            # plt.title("Phantom - Bckc normalised residuals")
            # plt.axis("equal")

            plt.figure()
            kwargs_phantom = dict(
                label="Phantom",
                color="r",
                zorder=1,
                alpha=0.8,
                linestyle="dashed",
            )
            kwargs_bckc = dict(
                label="Back-calculated",
                color="b",
                zorder=2,
                alpha=0.8,
            )
            if asymmetric_profile:
                phantom_emission.sel(t=t).sel(z=0, method="nearest").plot(
                    **kwargs_phantom,
                )
                bckc_emission.sel(t=t).sel(z=0, method="nearest").plot(
                    **kwargs_bckc,
                )
            else:
                rmag = PLASMA.equilibrium.rmag.sel(t=t, method="nearest")
                rho_lfs = (
                    PLASMA.equilibrium.rhop.sel(t=t, method="nearest")
                    .sel(z=0, method="nearest")
                    .sel(R=slice(rmag - 0.01, 1))
                )
                profile_1d = (
                    (
                        phantom_emission.sel(t=t)
                        .sel(z=0, method="nearest")
                        .interp(R=rho_lfs.R)
                    )
                    .assign_coords(rhop=("R", rho_lfs.data))
                    .swap_dims({"R": "rhop"})
                )
                profile_1d_bckc = (
                    (
                        bckc_emission.sel(t=t)
                        .sel(z=0, method="nearest")
                        .interp(R=rho_lfs.R)
                    )
                    .assign_coords(rhop=("R", rho_lfs.data))
                    .swap_dims({"R": "rhop"})
                )
                profile_1d.plot(
                    **kwargs_phantom,
                )
                profile_1d_bckc.plot(
                    **kwargs_bckc,
                )
            plt.title(f"Midplane cut (z=0) @ t={int(t*1.e3)} ms")
            set_axis_sci()
            plt.legend()
            plt.show()


def example_tomo_1D(
    instrument: str = "sxrc_xy1",
    asymmetric_profile: bool = False,
    element: str = "ar",
    reg_level_guess: float = 0.8,
    plot: bool = True,
):

    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry()
        emissivity = None
    else:
        rho_2d = PLASMA.equilibrium.rhop.interp(t=PLASMA.t.values)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)
        emissivity = (
            PLASMA.electron_density
            * PLASMA.ion_density.sel(element=element).drop_vars("element")
            * PLASMA.lz_tot[element].sum("ion_charge")
        )

    imp_density_2d = ion_density_2d.sel(element=element).drop_vars("element")
    el_density_2d = PLASMA.electron_density.interp(rhop=imp_density_2d.rhop)
    lz_tot_2d = (
        PLASMA.lz_tot[element].sum("ion_charge").interp(rhop=imp_density_2d.rhop)
    )
    phantom_emission = el_density_2d * imp_density_2d * lz_tot_2d
    los_transform = TRANSFORMS[instrument]
    los_transform.set_equilibrium(PLASMA.equilibrium)
    los_integral = los_transform.integrate_on_los(
        phantom_emission, phantom_emission.t.values
    )

    z = los_transform.z.mean("beamlet")
    R = los_transform.R.mean("beamlet")
    dl = los_transform.dl
    has_data = np.logical_not(np.isnan(los_integral.isel(t=0).data))
    rho_equil = los_transform.equilibrium.rhop.interp(t=los_integral.t)
    input_dict = dict(
        brightness=los_integral.data.T,
        dl=dl,
        t=los_integral.t.data,
        R=R.data,
        z=z.data,
        rho_equil=dict(
            R=rho_equil.R.data,
            z=rho_equil.z.data,
            t=rho_equil.t.data,
            rho=rho_equil.data,
        ),
        has_data=has_data,
        debug=False,
    )
    if emissivity is not None:
        input_dict["emissivity"] = emissivity

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)
    tomo()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rhop", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rhop", tomo.rho_grid_centers)],
    )
    inverted_emissivity = inverted_emissivity.assign_coords(
        error=(inverted_emissivity.dims, inverted_error.data)
    )

    data_tomo = los_integral
    bckc_tomo = DataArray(tomo.backprojection.T, coords=data_tomo.coords)

    if plot:
        plt.ioff()
        los_transform.plot()
        tomo.show_reconstruction()
        plt.show()

    return inverted_emissivity, data_tomo, bckc_tomo


def example_fit_ts(
    fit_R_shift: bool = False,
    verbose: bool = False,
    plot: bool = True,
):

    models = {"ts": ThomsonScattering}
    model_reader = ModelReader(models)
    model_reader.set_plasma(PLASMA)
    model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
    bckc = model_reader()

    te_data = bckc["ts"]["te"]
    te_err = bckc["ts"]["te"].error
    ne_data = bckc["ts"]["ne"]
    ne_err = bckc["ts"]["ne"].error

    if not fit_R_shift:
        R_shift = xr.full_like(te_data.t, 0.0)
    else:
        R_shift = None

    ts_R = te_data.transform.R
    ts_z = te_data.transform.z
    equilibrium = te_data.transform.equilibrium
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

    if plot:
        time = te_data.t
        cols = CM(np.linspace(0.1, 0.75, len(time), dtype=float))

        plt.ioff()
        plt.figure()
        _R_shift = [f"{(ne_fit.R_shift.sel(t=t).values * 100):.1f}" for t in time]
        for i, t in enumerate(time.values):
            if i % NPLOT:
                continue
            plt.errorbar(
                ne_data.transform.rhop.sel(t=t),
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
        plt.title("TS Ne data & fits")
        plt.xlim(0, 1.1)
        plt.ylim(0, np.nanmax(ne_fit) * 1.2)
        plt.legend()

        plt.figure()
        for i, t in enumerate(time.values):
            if i % NPLOT:
                continue
            plt.errorbar(
                te_data.transform.rhop.sel(t=t),
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
        plt.title("TS Te data & fits")
        plt.xlim(0, 1.1)
        plt.ylim(0, np.nanmax(te_fit) * 1.2)
        plt.legend()
        plt.show()

    return te_data, ne_data, te_fit, ne_fit
