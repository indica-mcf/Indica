import numpy as np
import xarray as xr
import indica.models.interferometry as interferometry
from indica.models.plasma import example_run as example_plasma
from indica.models.plasma import Plasma

from indica.readers import ST40Reader
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates


def match_interferometer_los_int(
    models: dict,
    data: dict,
    t: float,
    optimize_for: dict = {"smmh1": ["ne"]},
    optimize: dict = {"Ne_prof": ["y0"]},
    guesses: dict = {"Ne_prof": {"y0": 5.0e19}},
    bounds:dict = {"Ne_prof": {"y0": (1.e17, 1.e21)}},
    niter: int = 3,
):

    list_data = []
    list_model = []
    list_quantity = []
    for instrument in optimize_for.keys():
        if instrument in models:
            list_model.append(models[instrument])
            for quantity in optimize_for[instrument]:
                key = f"{instrument}_{quantity}"
                list_data[key] = data[instrument][quantity]
                list_quantity.append(quantity)

    plasma = models
    Ne_prof = plasma.Ne_prof
    const = 1.0
    for j in range(niter):
        ne0 *= const
        ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
        Ne_prof.set_parameters(y0=ne0)

        list_const = []
        for _model, _data, _quantity in zip(list_model, list_data, list_quantity):
            _bckc = _model(Ne_prof(), t=t)
            list_const.append(
                (
                    _data[_quantity].sel(t=t, method="nearest")
                    / _bckc[_quantity].sel(t=t, method="nearest")
                ).values
            )
        const = np.array(list_const).mean()

    plasma.electron_density.loc[dict(t=t)] = Ne_prof().values

    return plasma


def example_run():
    instrument = "smmh1"
    plasma = example_plasma()

    pulse = 9229
    reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)

    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    _data = reader.get("interferom", instrument, 0)
    trans = _data[list(_data)[0]].transform
    data = {}
    data[instrument] = _data

    model = interferometry.Interferometry(instrument)
    model.set_transform(trans)
    model.set_flux_transform(plasma.flux_transform)
    model.set_plasma(plasma)

    return plasma, model, data
