import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from xarray import DataArray


def default_profiles():
    el_temp = build_profile(1.5e3, 50, default="temperature")
    el_dens = build_profile(5.0e19, 1.0e19, default="density")
    vrot = build_profile(150.0e3, 10.0e3, default="rotation")

    return el_temp, el_dens, vrot


def build_profile(
    y0,
    y1,
    wped=4,
    wcenter=0.4,
    peaking=1.5,
    y0_fix=False,
    default=None,
    plot=False,
    coord="rho_poloidal",
):
    """

    Parameters
    ----------
    y0
        value at the centre
    y1
        value at separatrix
    wped
        parameter governing the pedestal peaking [2, 20]
    wcenter
        HFHM of gaussian used for central peaking
    peaking
        peaking factor: new central value = y0 * peaking
    y0_fix
        re-scale new profile to have central value = y0

    Returns
    -------

    """

    def gaussian(x, A, B, x_0, w):
        return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B

    parameters = {
        "density": (12, 0.5, 1.3),
        "temperature": (4, 0.4, 1.5),
        "rotation": (4, 0.4, 1.4),
    }
    if default is not None:
        wped, wcenter, peaking = parameters[default]

    xspl = np.linspace(0, 1.0, 30)

    # nominal profile, no peaking
    x = np.linspace(0, 1, 15) ** 0.7
    y = (y0 - y1) * (1 - x ** wped) + y1

    # add central peaking
    if peaking > 1:
        sigma = wcenter / (np.sqrt(2 * np.log(2)))
        ycenter = gaussian(x, y0 * (peaking - 1), 0, 0, sigma)
        y += ycenter

    if y0_fix:
        y -= y1
        y /= y[0]
        y *= y0 - y1
        y += y1

    x = np.append(x, 1.05)
    y = np.append(y, 0.0)

    cubicspline = CubicSpline(
        x,
        y,
        0,
        "clamped",
        False,
    )
    yspl = cubicspline(xspl)

    if plot:
        plt.plot(x, y, "o")
        plt.plot(xspl, yspl)

    result = DataArray(yspl, coords=[(coord, xspl)])

    return result
