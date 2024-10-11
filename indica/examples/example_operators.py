import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.defaults.load_defaults import load_default_objects
from indica.operators import tomo_1D
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_2d_map
from indica.operators.centrifugal_asymmetry import centrifugal_asymmetry_parameter
from indica.operators.tomo_asymmetry import InvertPoloidalAsymmetry
from indica.utilities import set_axis_sci

PLASMA = load_default_objects("st40", "plasma")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
GEOMETRY = load_default_objects("st40", "geometry")
LOS_TRANSFORM = GEOMETRY["sxrc_xy2"]
LOS_TRANSFORM.set_equilibrium(EQUILIBRIUM)
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
    asymmetric_profile: bool = True,
    debug: bool = True,
    plot: bool = True,
    element: str = "ar",
):

    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry()
    else:
        rho_2d = PLASMA.equilibrium.rho.interp(t=PLASMA.t)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)

    imp_density_2d = ion_density_2d.sel(element=element).drop_vars("element")
    el_density_2d = PLASMA.electron_density.interp(rhop=imp_density_2d.rhop)
    lz_tot_2d = (
        PLASMA.lz_tot[element].sum("ion_charge").interp(rhop=imp_density_2d.rhop)
    )
    phantom_emission = el_density_2d * imp_density_2d * lz_tot_2d
    los_integral = LOS_TRANSFORM.integrate_on_los(
        phantom_emission, phantom_emission.t.values
    )

    if debug:
        print("\n Running asymmetry inference...")
    invert_asymm = InvertPoloidalAsymmetry()
    bckc_emission, bckc, profile, asymmetry = invert_asymm(
        los_integral,
        LOS_TRANSFORM,
        debug=debug,
    )
    if debug:
        print("...and finished \n")

    if plot:
        LOS_TRANSFORM.plot()
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
                    PLASMA.equilibrium.rho.sel(t=t, method="nearest")
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
    plot: bool = True,
    asymmetric_profile: bool = False,
    element: str = "ar",
    reg_level_guess: float = 0.8,
):

    if asymmetric_profile:
        ion_density_2d = example_poloidal_asymmetry()
    else:
        rho_2d = PLASMA.equilibrium.rho.interp(t=PLASMA.t)
        ion_density_2d = PLASMA.ion_density.interp(rhop=rho_2d)

    imp_density_2d = ion_density_2d.sel(element=element).drop_vars("element")
    el_density_2d = PLASMA.electron_density.interp(rhop=imp_density_2d.rhop)
    lz_tot_2d = (
        PLASMA.lz_tot[element].sum("ion_charge").interp(rhop=imp_density_2d.rhop)
    )
    phantom_emission = el_density_2d * imp_density_2d * lz_tot_2d
    los_integral = LOS_TRANSFORM.integrate_on_los(
        phantom_emission, phantom_emission.t.values
    )

    z = LOS_TRANSFORM.z.mean("beamlet")
    R = LOS_TRANSFORM.R.mean("beamlet")
    dl = LOS_TRANSFORM.dl
    has_data = np.logical_not(np.isnan(los_integral.isel(t=0).data))
    rho_equil = LOS_TRANSFORM.equilibrium.rho.interp(t=los_integral.t)
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


if __name__ == "__main__":
    example_tomo_asymmetry(debug=False, asymmetric_profile=False)
