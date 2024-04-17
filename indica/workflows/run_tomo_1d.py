"""Inverts line of sight integrals to estimate local emissivity."""


import numpy as np
from xarray import DataArray

from indica.converters.time import convert_in_time_dt
from indica.defaults.default_geometries import load_default_equilibrium
from indica.defaults.default_geometries import load_default_transforms
from indica.equilibrium import Equilibrium
from indica.models.plasma import example_plasma
from indica.operators import tomo_1D
from indica.readers import ST40Reader
from indica.readers.modelreader import ModelReader


def example_tomo(
    pulse: int,
    tstart: float = 0.02,
    tend: float = 0.1,
    dt: float = 0.01,
    dl: float = 0.02,
    instrument: str = "sxrc_xy2",
    reg_level_guess: float = 0.8,
    plot: bool = True,
    phantom_data: bool = False,
):
    machine = "st40"
    if phantom_data:
        modelreader = ModelReader(machine, instruments=[instrument])

        transforms = load_default_transforms(machine)
        modelreader.set_geometry_transforms(transforms)

        equilibrium = load_default_equilibrium(machine)
        tstart = equilibrium.t.min() + dt
        tend = equilibrium.t.max() - dt
        plasma = example_plasma("st40", tstart=tstart, tend=tend)
        plasma.set_equilibrium(equilibrium)
        modelreader.set_plasma(plasma)

        data = modelreader.get("", instrument)
        brightness = data["brightness"]
        los_transform = brightness.transform
        emissivity = modelreader.models[instrument].emissivity
    else:
        st40reader = ST40Reader(pulse, tstart - dt, tend + dt)
        data = st40reader.get("", instrument, dl=dl)
        brightness = convert_in_time_dt(tstart, tend, dt, data["brightness"])
        los_transform = brightness.transform
        efit = st40reader.get("", "efit")
        equilibrium = Equilibrium(efit)
        los_transform.set_equilibrium(equilibrium, force=True)
        emissivity = None

    z = los_transform.z.mean("beamlet")
    R = los_transform.R.mean("beamlet")
    dl = los_transform.dl

    has_data = np.logical_not(np.isnan(brightness.isel(t=0).data))
    rho_equil = los_transform.equilibrium.rho.interp(t=brightness.t)
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
        has_data=has_data,
        debug=False,
    )
    if emissivity is not None:
        input_dict["emissivity"] = emissivity

    tomo = tomo_1D.SXR_tomography(input_dict, reg_level_guess=reg_level_guess)
    tomo()
    if plot:
        los_transform.plot()
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
