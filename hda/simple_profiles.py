import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from xarray import DataArray


def default_profiles():
    el_temp, _ = build_profile(1.5e3, 50, default="temperature")
    el_dens, _ = build_profile(5.0e19, 1.0e19, default="density")
    vrot, _ = build_profile(150.0e3, 10.0e3, default="rotation")

    return el_temp, el_dens, vrot


def build_profile(
    y0,
    y1,
    wped=4,
    wcenter=0.4,
    peaking=1.5,
    peaking2=1,
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
    peaking2
        additional peaking factor for modification of peaked profile
        new central value = y0 * peaking * peaking2
    y0_fix
        re-scale new profile to have central value = y0

    Returns
    -------

    """

    def gaussian(x, A, B, x_0, w):
        return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B

    if default is not None:
        wped, wcenter, peaking = get_defaults(default)

    params = (y0, y1, wped, wcenter, peaking, y0_fix)

    xspl = np.linspace(0, 1.0, 30)

    # nominal profile, no peaking
    x = np.linspace(0, 1, 15) ** 0.7
    y = (y0 - y1) * (1 - x ** wped) + y1

    # add central peaking
    if peaking > 1:
        sigma = wcenter / (np.sqrt(2 * np.log(2)))
        ycenter = gaussian(x, y0 * (peaking - 1), 0, 0, sigma)
        y += ycenter

    if peaking2 > 1:
        sigma = wcenter / (np.sqrt(2 * np.log(2)))
        ycenter = gaussian(x, y[0] * (peaking2 - 1), 0, 0, sigma)
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

    return result, params

def get_defaults(default):

    parameters = {
        "density": (12, 0.4, 1.3),
        "temperature": (4, 0.4, 1.5),
        "rotation": (4, 0.4, 1.4),
    }

    if default in parameters.keys():
        return parameters[default]