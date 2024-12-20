from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline
import xarray as xr
from xarray import DataArray

from indica.profilers.profiler_gauss import ProfilerGauss
from indica.utilities import set_plot_colors

CMAP, COLORS = set_plot_colors()


def sawtooth_crash(pre: ProfilerGauss, rho_inv: float, volume: DataArray = None):
    """
    Model a sawtooth crash, without resetting the internal quantities

    If volume in not None, then volume integral is conserved adapting the edge shape

    Parameters
    ----------
    rho_inv
        Inversion radius
    volume
        Plasma volume

    """
    post = deepcopy(pre)

    if volume is None:
        volume = DataArray(0.85 * pre.xspl**3, coords=[("rhop", pre.xspl)])
    vol = volume.interp(rhop=pre.ydata.rhop)
    vol_int_pre = np.trapz(pre.ydata, vol)

    x = pre.x[np.where(pre.x <= 1.0)[0]]

    rho = post.ydata.rhop
    inv_ind = np.max(np.where(rho <= rho_inv)[0])
    for rind in np.arange(inv_ind, rho.size):
        y = xr.where(rho <= rho[rind], post.ydata.sel(rhop=rho[inv_ind]), post.ydata)
        vol_int_post = np.trapz(y, vol)
        if vol_int_post >= vol_int_pre:
            break

    y = xr.where(rho != rho[rind], y, (y[rind] + y[rind + 1]) / 2)
    y = y.interp(rhop=x)

    x = np.append(x, pre.xend)
    y = np.append(y, pre.yend)
    cubicspline = CubicSpline(
        x,
        y,
        0,
        "clamped",
        False,
    )
    post.ydata.values = cubicspline(pre.xspl)
    vol_int_post = np.trapz(post.ydata, vol)

    print(f"Vol-int: {vol_int_pre}, {vol_int_post}")

    return post
