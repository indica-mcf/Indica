import numpy as np
import xarray as xr
from xarray import DataArray
from indica.provenance import get_prov_attribute
from hda.manage_data import initialize_bckc_dataarray
from hda.utils import print_like
from hda.profiles import Profiles
from hda.models.interferometer import Interferometer
from indica.numpy_typing import LabeledArray


def match_interferometer_los_int(
    model: Interferometer,
    Ne_prof: Profiles,
    data: dict,
    t:float,
    quantities:list=["ne"],
    bckc:dict=None,
    ne0=5.e19,
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

    data_value = {}
    for quantity in quantities:
        data_value[quantity] = data[quantity].sel(t=t).values

    const = 1.0
    for j in range(niter):
        ne0 *= const
        ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
        Ne_prof.y0 = ne0
        Ne_prof.build_profile()
        bckc_tmp, _ = model.integrate_on_los(Ne_prof.yspl, t=t)

        const = []
        for quantity in quantities:
            _const = (data_value[quantity] / bckc_tmp[quantity]).values
            const.append(_const)
        const = np.array(const).mean()

    if bckc is None:
        bckc = bckc_tmp
    else:
        for quantity in quantities:
            bckc[quantity].loc[dict(t=t)] = bckc_tmp[quantity]

    return bckc, Ne_prof
