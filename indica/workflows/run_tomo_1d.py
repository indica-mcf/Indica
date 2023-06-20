"""Inverts line of sight integrals to estimate local emissivity."""

from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.operators import tomo_1D
from indica.readers.read_st40 import ReadST40

DataArrayCoords = Tuple[DataArray, DataArray]


def sxrc_xy(
    pulse: int = 10619,  # 10605
    tstart: float = 0.02,
    tend: float = 0.11,
    debug=False,
    exclude_bad_points=True,
    plot=True,
    reg_level_guess: float = 0.5,
    input_dict: dict = None,
    channels=slice(0, 15),
):

    if input_dict is None:
        st40 = ReadST40(pulse, tstart, tend)
        st40(instruments=["sxrc_xy2"], map_diagnostics=False)
        data = st40.binned_data["sxrc_xy2"]["brightness"].sel(channel=channels)
        equil = st40.equilibrium
        z = data.transform.z.sel(channel=channels)
        R = data.transform.R.sel(channel=channels)
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
        impact_paramaters = data.transform.impact_parameter

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
            impact_parameters=impact_paramaters,
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


def fake_data(
    pulse: int = 9229,
    plasma=None,
    model=None,
    bckc=None,
    debug=False,
    exclude_bad_points=True,
    plot=True,
    reg_level_guess: float = 0.5,
    nchannels=12,
    input_dict: dict = None,
):
    from indica.models.bolometer_camera import example_run

    if input_dict is None:
        if plasma is None or model is None or bckc is None:
            plasma, model, bckc = example_run(pulse=pulse, nchannels=nchannels)

            tstart = plasma.t.min().values
            tend = plasma.t.max().values
            st40 = ReadST40(pulse, tstart, tend)
            st40(instruments=["sxrc_xy2", "sxr_camera_4"])

            try:
                data = st40.binned_data["sxrc_xy2"]["brightness"]
            except KeyError:
                data = st40.binned_data["sxr_camera_4"]["brightness"]
            model.set_los_transform(data.transform)
            model.los_transform.set_equilibrium(plasma.equilibrium, force=True)
            model()

        z = model.los_transform.z
        R = model.los_transform.R
        dl = model.los_transform.dl

        impact_paramaters = model.los_transform.impact_parameter

        brightness = model.bckc["brightness"]
        data_t0 = brightness.isel(t=0).data
        if exclude_bad_points:
            has_data = np.logical_not(np.isnan(data_t0)) & (data_t0 >= 1.0e3)
        else:
            has_data = np.logical_not(np.isnan(data_t0))

        rho_equil = plasma.equilibrium.rho.interp(t=brightness.t)
        input_dict = dict(
            emissivity=model.emissivity,
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
            impact_parameters=impact_paramaters,
            debug=debug,
            has_data=has_data,
        )

        # return input_dict

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)

    tomo()

    if plot:
        tomo.show_reconstruction()

    return input_dict
