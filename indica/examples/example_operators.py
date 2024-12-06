import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.operators.spline_fit import fit_profile
from indica.operators.tomo_asymmetry import InvertPoloidalAsymmetry
from indica.readers.modelreader import ModelReader
from indica.utilities import set_axis_sci

PLASMA = load_default_objects("st40", "plasma")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
TRANSFORMS = load_default_objects("st40", "geometry")
PLASMA.set_equilibrium(EQUILIBRIUM)


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
    debug: bool = True,
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

    if debug:
        print("\n Running asymmetry inference...")
    invert_asymm = InvertPoloidalAsymmetry()
    bckc_emission, bckc, profile, asymmetry = invert_asymm(
        los_integral,
        los_transform,
        debug=debug,
    )
    if debug:
        print("...and finished \n")

    if plot:
        los_transform.plot()
        for t in bckc.t:
            plt.ioff()

            kwargs_phantom = dict(
                marker="o", label="Phantom", color="r", zorder=1, alpha=0.8
            )
            kwargs_bckc = dict(
                marker="x",
                label="Back-calculated",
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
            plt.title("LOS integrals")
            plt.legend()
            set_axis_sci()

            plt.figure()
            phantom_emission.attrs = {"long_name": "Emissivity", "units": "W/m$^3$"}
            phantom_emission.sel(t=t).plot()
            plt.title("Phantom")
            plt.axis("equal")

            plt.figure()
            bckc_emission.attrs = {"long_name": "Emissivity", "units": "W/m$^3$"}
            bckc_emission.sel(t=t).plot()
            plt.title("2D back-calculated")
            plt.axis("equal")

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
            plt.title("Midplane cut (z=0)")
            set_axis_sci()
            plt.legend()
            plt.show()


def example_tomo_1D(
    instrument: str = "sxrc_xy1",
    asymmetric_profile: bool = False,
    element: str = "ar",
    reg_level_guess: float = 0.8,
    debug: bool = True,
    plot: bool = True,
):

    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry()
    else:
        rho_2d = PLASMA.equilibrium.rhop.interp(t=PLASMA.t.values)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)

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
        tomo.show_reconstruction()
        plt.show()

    return inverted_emissivity, data_tomo, bckc_tomo


def example_spline_fit(
    machine: str = "st40",
    quantity: str = "te",
    R_shift: float = 0.0,
    knots: list = None,
    plot: bool = True,
):
    _reader = ModelReader(machine, instruments=["ts"])
    _reader.set_geometry_transforms(TRANSFORMS["ts"])
    PLASMA.set_equilibrium(EQUILIBRIUM)
    _reader.set_plasma(PLASMA)
    raw_data = _reader()

    if quantity == "te" and knots is None:
        knots = [0, 0.3, 0.6, 0.8, 1.1]
    if quantity == "ne" and knots is None:
        knots = [0, 0.3, 0.6, 0.8, 0.95, 1.1]
    data_all = raw_data["ts"][quantity]
    t = data_all.t
    transform = data_all.transform
    transform.convert_to_rho_theta(t=data_all.t)

    R = transform.R
    Rmag = transform.equilibrium.rmag.interp(t=t)

    # Fit all available TS data
    ind = np.full_like(data_all, True)
    rho = xr.where(ind, transform.rho, np.nan)
    data = xr.where(ind, data_all, np.nan)
    err = xr.where(ind, data_all.error, np.nan)
    fit = fit_profile(rho, data, err, knots=knots, virtual_knots=False)

    # Use only HFS channels
    ind = R <= Rmag
    rho_hfs = xr.where(ind, transform.rho, np.nan)
    data_hfs = xr.where(ind, data_all, np.nan)
    err_hfs = xr.where(ind, data_all.error, np.nan)
    fit_hfs = fit_profile(rho_hfs, data_hfs, err_hfs, knots=knots, virtual_knots=True)

    # Use only LFS channels
    ind = R >= Rmag
    rho_lfs = xr.where(ind, transform.rho, np.nan)
    data_lfs = xr.where(ind, data_all, np.nan)
    err_lfs = xr.where(ind, data_all.error, np.nan)
    fit_lfs = fit_profile(rho_lfs, data_lfs, err_lfs, knots=knots, virtual_knots=True)

    if plot:
        for t in data_all.t:
            plt.ioff()
            plt.errorbar(
                rho_hfs.sel(t=t),
                data_hfs.sel(t=t),
                err_hfs.sel(t=t),
                marker="o",
                label="data HFS",
                color="blue",
            )
            plt.errorbar(
                rho_lfs.sel(t=t),
                data_lfs.sel(t=t),
                err_lfs.sel(t=t),
                marker="o",
                label="data LFS",
                color="red",
            )
            fit.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="black", label="spline fit all"
            )
            fit_lfs.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="red", label="spline fit LFS"
            )
            fit_hfs.sel(t=t).plot(
                linewidth=5, alpha=0.5, color="blue", label="spline fit HFS"
            )
            plt.legend()
            plt.show()

    return data_all, fit
