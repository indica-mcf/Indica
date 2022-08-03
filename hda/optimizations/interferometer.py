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
    interferometer: Interferometer,
    data: DataArray,
    Ne_prof: Profiles,
    niter: int = 3,
    t: LabeledArray = None,
):
    """
    Rescale density profiles to match the interferometer measurements

    Parameters
    ----------
    interferometer
        Model to simulate an interferometer
    data
        Interferometer line of sight integral of the electron density
    Ne_prof
        Profile object to build electron density profile for optimization
    t
        If None, run optimization for all time-points

    Returns
    -------

    """

    print_like(f"Re-calculating density profiles to match interferometer measurement")

    if t is None:
        t = data.t

    bckc = initialize_bckc_dataarray(data)
    Ne = []
    for time in t:
        const = 1.0
        for j in range(niter):
            ne0 = Ne_prof.yspl.sel(rho_poloidal=0) * const
            ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
            Ne_prof.y0 = ne0.values
            Ne_prof.build_profile()
            los_integral, _ = interferometer.line_integrated_density(Ne_prof.yspl, t=time)
            bckc.loc[dict(t=time)] = los_integral
            const = (data.sel(t=time) / bckc.sel(t=time)).values
        Ne.append(Ne_prof.yspl)
    Ne = xr.concat(Ne, "t").assign_coords({"t":t})

    return bckc, Ne
