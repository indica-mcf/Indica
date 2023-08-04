"""Inverts line of sight integrals to estimate local emissivity."""

import getpass
from typing import Callable
from typing import Dict
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.models.diode_filters import BremsstrahlungDiode
from indica.models.diode_filters import example_run as brems_example
from indica.models.plasma import example_run as example_plasma
from indica.models.sxr_camera import example_run as sxr_example
from indica.models.sxr_camera import SXRcamera
from indica.operators import tomo_1D
from indica.readers.read_st40 import ReadST40
from indica.utilities import save_figure
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams

DataArrayCoords = Tuple[DataArray, DataArray]

set_plot_rcparams("profiles")

PHANTOMS: Dict[str, Callable] = {
    "sxrc_xy2": sxr_example,
    "sxr_camera_4": sxr_example,
    "pi": brems_example,
}

PULSES: Dict[str, int] = {
    "sxrc_xy2": 10821,
    "sxr_camera_4": 9229,
    "pi": 10821,
}

SXR_MODEL = SXRcamera("sxrc_xy2")
BREMS_MODEL = BremsstrahlungDiode("pi")


def phantom_examples(
    instrument: str = "sxrc_xy2",
    reg_level_guess: float = 0.5,
    plot: bool = True,
):
    plasma, model, bckc = PHANTOMS[instrument]()
    los_transform = model.los_transform
    emissivity = model.emissivity
    brightness = bckc["brightness"]
    z = los_transform.z
    R = los_transform.R
    dl = los_transform.dl

    has_data = np.logical_not(np.isnan(brightness.isel(t=0).data))
    rho_equil = plasma.equilibrium.rho.interp(t=brightness.t)
    input_dict = dict(
        brightness=brightness.data,
        dl=dl,
        t=brightness.t.data,
        R=R,
        z=z,
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
    if plot:
        model.los_transform.plot()
        tomo.show_reconstruction()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    inverted_emissivity.attrs["error"] = inverted_error

    data_tomo = brightness
    bckc_tomo = DataArray(tomo.backprojection, coords=data_tomo.coords)

    return inverted_emissivity, data_tomo, bckc_tomo


def sxrc_example(
    pulse: int = None,
    instrument: str = "sxrc_xy2",
    reg_level_guess: float = 0.5,
    phantom_data: bool = True,
    plot: bool = True,
):
    if pulse is None:
        pulse = PULSES[instrument]

    tstart = 0.02
    tend = 0.1
    dt = 0.01
    st40 = ReadST40(pulse, tstart, tend, dt=dt)
    st40(instruments=[instrument, "efit"])
    equilibrium = Equilibrium(st40.raw_data["efit"])
    quantity = list(st40.binned_data[instrument])[0]
    los_transform = st40.binned_data[instrument][quantity].transform
    los_transform.set_equilibrium(equilibrium, force=True)
    model = SXR_MODEL
    model.set_los_transform(los_transform)

    if phantom_data:
        plasma = example_plasma(
            pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
        )
        plasma.build_atomic_data()
        plasma.set_equilibrium(equilibrium)
        model.set_plasma(plasma)
        bckc = model()
        emissivity = model.emissivity
        brightness = bckc["brightness"]
    else:
        emissivity = None
        brightness = st40.binned_data[instrument]["brightness"]

    z = los_transform.z
    R = los_transform.R
    dl = los_transform.dl

    data_t0 = brightness.isel(t=0).data
    has_data = np.logical_not(np.isnan(brightness.isel(t=0).data)) & (data_t0 > 0)
    rho_equil = equilibrium.rho.interp(t=brightness.t)
    input_dict = dict(
        brightness=brightness.data,
        dl=dl,
        t=brightness.t.data,
        R=R,
        z=z,
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

    if plot:
        model.los_transform.plot()
        tomo.show_reconstruction()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    inverted_emissivity.attrs["error"] = inverted_error
    data_tomo = brightness
    bckc_tomo = DataArray(tomo.backprojection, coords=data_tomo.coords)

    return inverted_emissivity, data_tomo, bckc_tomo


def pi_bremsstrahlung_example(
    pulse: int = None,
    instrument: str = "pi",
    reg_level_guess: float = 0.5,
    phantom_data: bool = True,
    plot: bool = True,
):
    if pulse is None:
        pulse = PULSES[instrument]

    tstart = 0.02
    tend = 0.1
    dt = 0.01
    st40 = ReadST40(pulse, tstart, tend, dt=dt)
    st40(instruments=[instrument, "efit"])
    equilibrium = Equilibrium(st40.raw_data["efit"])
    quantity = list(st40.binned_data[instrument])[0]
    los_transform = st40.binned_data[instrument][quantity].transform
    los_transform.set_equilibrium(equilibrium, force=True)
    model = BREMS_MODEL
    model.set_los_transform(los_transform)
    attrs = st40.binned_data[instrument]["spectra"].attrs
    background, brightness = BREMS_MODEL.integrate_spectra(
        st40.binned_data[instrument]["spectra"]
    )
    background.attrs = attrs
    brightness.attrs = attrs
    st40.binned_data[instrument]["background"] = background
    st40.binned_data[instrument]["brightness"] = brightness

    if phantom_data:
        plasma = example_plasma(
            pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
        )
        plasma.build_atomic_data()
        plasma.set_equilibrium(equilibrium)
        model.set_plasma(plasma)
        bckc = model()
        emissivity = model.emissivity
        brightness = bckc["brightness"]
    else:
        emissivity = None
        brightness = st40.binned_data[instrument]["brightness"]

    z = los_transform.z
    R = los_transform.R
    dl = los_transform.dl

    data_t0 = brightness.isel(t=0).data
    has_data = np.logical_not(np.isnan(brightness.isel(t=0).data)) & (data_t0 > 0)
    rho_equil = equilibrium.rho.interp(t=brightness.t)
    input_dict = dict(
        brightness=brightness.data,
        dl=dl,
        t=brightness.t.data,
        R=R,
        z=z,
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

    if plot:
        model.los_transform.plot()
        tomo.show_reconstruction()

    inverted_emissivity = DataArray(
        tomo.emiss, coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)]
    )
    inverted_error = DataArray(
        tomo.emiss_err,
        coords=[("t", tomo.tvec), ("rho_poloidal", tomo.rho_grid_centers)],
    )
    inverted_emissivity.attrs["error"] = inverted_error
    data_tomo = brightness
    bckc_tomo = DataArray(tomo.backprojection, coords=data_tomo.coords)

    return inverted_emissivity, data_tomo, bckc_tomo


def sxrc_xy(
    pulse: int = 10821,
    tstart: float = 0.02,
    tend: float = 0.11,
    dt: float = 0.01,
    debug=False,
    exclude_bad_points=True,
    plot=True,
    reg_level_guess: float = 0.5,
    input_dict: dict = None,
    channels=slice(0, 15),
    save_fig: bool = False,
    instrument="sxrc_xy2",
):

    if input_dict is None:
        st40 = ReadST40(pulse, tstart, tend, dt=dt)
        st40(instruments=[instrument], map_diagnostics=False)
        data = st40.binned_data[instrument]["brightness"].sel(channel=channels)
        data.transform.set_dl(data.transform.dl)
        dl = data.transform.dl
        z = data.transform.z.sel(channel=channels)
        R = data.transform.R.sel(channel=channels)

        equil = st40.equilibrium
        rho, theta = data.transform.convert_to_rho_theta(t=data.t)
        # rho = rho.sel(channel=channels)

        brightness = data

        data.transform.plot()

        data_R = data.transform.impact_parameter.R.sel(channel=channels)
        data = data.assign_coords(R=("channel", data_R)).swap_dims({"channel": "R"})

        fig_path = f"/home/{getpass.getuser()}/figures/Indica/time_evolution/"
        plt.figure()
        surf = data.T.plot()
        set_axis_sci(plot_object=surf)
        data.transform.equilibrium.rmag.plot(
            linestyle="dashed", color="w", label="R$_{mag}$"
        )
        plt.ylabel("R [m]")
        plt.xlabel("t [s]")
        plt.legend()
        if save_fig:
            save_figure(
                fig_path,
                f"{pulse}_{instrument}_surface_plot",
                save_fig=save_fig,
            )

        plt.figure()
        data.sel(R=0.4, method="nearest").plot(label="R=0.4 m")
        data.sel(R=0.47, method="nearest").plot(label="R=0.47 m")
        data.sel(R=0.55, method="nearest").plot(label="R=0.55 m")
        plt.title("")
        set_axis_sci()
        plt.legend()
        if save_fig:
            save_figure(
                fig_path,
                f"{pulse}_{instrument}_channel_evolution",
                save_fig=save_fig,
            )

        data_t0 = brightness.isel(t=0).data
        if exclude_bad_points:
            has_data = np.logical_not(np.isnan(data_t0)) & (data_t0 >= 1.0e3)
        else:
            has_data = np.logical_not(np.isnan(data_t0))

        rho_equil = equil.rho.interp(t=brightness.t, method="nearest")
        input_dict = dict(
            brightness=brightness.data,
            dl=dl,
            t=brightness.t.data,
            R=R.data,
            z=z.data,
            rho_equil=dict(
                R=rho_equil.R.data,
                z=rho_equil.z.data,
                t=rho_equil.t.data,
                rho=rho_equil.data,
            ),
            debug=debug,
            has_data=has_data,
        )
        # return input_dict

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)

    tomo()

    if plot:
        plt.ioff()
        tomo.show_reconstruction()

    return input_dict


def old_camera(
    pulse: int = 9229,
    tstart: float = 0.02,
    tend: float = 0.12,
    debug=False,
    exclude_bad_points=True,
    plot=True,
    reg_level_guess: float = 0.5,
    input_dict: dict = None,
):

    if input_dict is None:
        st40 = ReadST40(pulse, tstart, tend)
        # return st40
        st40(instruments=["sxr_camera_4"], map_diagnostics=False)
        data = st40.binned_data["sxr_camera_4"]["brightness"]
        equil = st40.equilibrium
        z = data.transform.z - 0.02
        R = data.transform.R
        dl = data.transform.dl

        brightness = data
        data_t0 = brightness.isel(t=0).data
        if exclude_bad_points:
            has_data = np.logical_not(np.isnan(data_t0)) & (data_t0 >= 1.0e3)
        else:
            has_data = np.logical_not(np.isnan(data_t0))

        rho_equil = equil.rho.interp(t=brightness.t, method="nearest")
        input_dict = dict(
            brightness=brightness.data,
            dl=dl,
            t=brightness.t.data,
            R=R.data,
            z=z.data,
            rho_equil=dict(
                R=rho_equil.R.data,
                z=rho_equil.z.data,
                t=rho_equil.t.data,
                rho=rho_equil.data,
            ),
            debug=debug,
            has_data=has_data,
        )
        # return input_dict

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)

    tomo()

    if plot:
        plt.ioff()
        tomo.show_reconstruction()

    return input_dict
