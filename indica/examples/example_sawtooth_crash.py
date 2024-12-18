from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from indica.operators.sawtooth_crash import sawtooth_crash
from xarray import DataArray
from indica.utilities import set_plot_colors
from indica.profilers.profiler_gauss import ProfilerGauss

CMAP, COLORS = set_plot_colors()

def density_crash(
    los_avrg=2.8e19,
    rhop=np.linspace(0, 1, 20),
    rho_inv=0.4,
    identifier="density",
):
    volume = DataArray(0.85 * rhop**3, coords=[("rhop", rhop)])

    pre = ProfilerGauss(f"electron_{identifier}", xspl=rhop)
    pre.wcenter = rho_inv / 1.5
    pre()

    plt.figure()

    scan = np.linspace(1.0, 5.0, 5)
    cols = CMAP(np.linspace(0.1, 0.75, len(scan), dtype=float))
    for i, s in enumerate(scan):
        pre.peaking = s
        pre.y0 *= los_avrg / pre.ydata.mean().values
        pre()

        post = sawtooth_crash(deepcopy(pre), rho_inv, volume)

        pre.ydata.plot(color=cols[i])
        post.ydata.plot(linestyle="dashed", color=cols[i])

    return pre, post
