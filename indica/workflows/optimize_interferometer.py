import numpy as np
import xarray as xr
from xarray import DataArray
from indica.provenance import get_prov_attribute
from indica.readers.manage_data import initialize_bckc_dataarray
from indica.utilities import print_like
from indica.profiles_gauss import Profiles
from indica.models.interferometry import Interferometry
from indica.numpy_typing import LabeledArray


def match_interferometer_los_int(
    models: dict,
    Ne_prof: Profiles,
    data: dict,
    t: float,
    instruments: list = ["smmh1"],
    quantities: list = ["ne"],
    ne0=5.0e19,
    niter: int = 3,
):
    """
    Rescale density profiles to match the interferometer measurements

    Parameters
    ----------
    model
        Model to simulate an interferometer
    Ne_prof
        Profile object to build electron density profile for optimization
    data
        Dictionary of Dataarrays of interferometer data as returned by ST40reader
    t
        Time for which optimisation must be performed
    quantities
        Measurement identifiers to be optimised
    bckc
        Dictionary where back-calculated values are to be saved (same structure as data)
    ne0
        Initial guess of central density
    niter
         Number of iterations

    Returns
    -------

    """

    list_data = []
    list_model = []
    list_quantity = []
    for instrument in instruments:
        if instrument in models:
            for quantity in quantities:
                list_model.append(models[instrument])
                list_data.append(data[instrument])
                list_quantity.append(quantity)

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

    return Ne_prof
