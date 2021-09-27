import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from xarray import DataArray
from copy import deepcopy


class Profiles:
    def __init__(self, identifier="temperature", datatype:tuple=("", "")):
        # Radial arrays for building profiles and smoothing using splines
        self.x = np.linspace(0, 1, 15) ** 0.7
        self.xspl = np.linspace(0, 1.0, 30)

        self.params = get_defaults(prof_type)
        self.build_profile(1.5e3, 50, datatype=datatype)

    def set_parameters(self, wped, wcenter, peaking, peaking2=1):
        """

        Parameters
        ----------
        wped
            parameter governing the pedestal peaking [2, 20]
        wcenter
            HFHM of gaussian used for central peaking
        peaking
            peaking factor: new central value = y0 * peaking
        peaking2
            additional peaking factor for modification of peaked profile
            new central value = y0 * peaking * peaking2

        """
        self.params = {
            "wped": wped,
            "wcenter": wcenter,
            "peaking": peaking,
            "peaking2": peaking2,
        }

    def build_profile(
        self,
        y0,
        y1,
        y0_fix=False,
        plot=False,
        coord="rho_poloidal",
        datatype=None,
    ):
        """
        Builds the profiles using the parameters set

        Parameters
        ----------
        y0
            value at the centre
        y1
            value at separatrix
        y0_fix
            re-scale new profile to have central value = y0

        Returns
        -------

        """

        def gaussian(x, A, B, x_0, w):
            return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y = (y0 - y1) * (1 - x ** self.params["wped"]) + y1

        # add central peaking
        if self.params["peaking"] > 1:
            sigma = self.params["wcenter"] / (np.sqrt(2 * np.log(2)))
            ycenter = gaussian(x, y0 * (self.params["peaking"] - 1), 0, 0, sigma)
            y += ycenter

        # add additional peaking
        if self.params["peaking2"] > 1:
            sigma = self.params["wcenter"] / (np.sqrt(2 * np.log(2)))
            ycenter = gaussian(x, y[0] * (self.params["peaking2"] - 1), 0, 0, sigma)
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
        yspl = cubicspline(self.xspl)

        if plot:
            plt.plot(x, y, "o")
            plt.plot(xspl, yspl)

        result = DataArray(yspl, coords=[(coord, self.xspl)])

        if datatype is not None:
            attrs = {"datatype": datatype}
            name = datatype[1] + "_" + datatype[0]
            result.name = name
            result.attrs = attrs

        self.profile = result

def get_defaults(identifier):

    parameters = {
        "density": (12, 0.4, 1.3),
        "temperature": (4, 0.4, 1.5),
        "rotation": (4, 0.4, 1.4),
    }

    if identifier not in parameters.keys():
        print(
            f"\n {prof_type} not in parameters' keys \n Using 'temperature' as default \n"
        )
        identifier = "temperature"

    params = parameters[identifier]
    return {"wped": params[0], "wcenter": params[1], "peaking": params[2], "peaking2": 1.0}
