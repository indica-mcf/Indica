import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import CubicSpline
from xarray import DataArray
from copy import deepcopy


class Profiles:
    def __init__(self, datatype=("temperature", "electron"), xspl=None):
        # Radial arrays for building profiles and smoothing using splines
        self.x = np.linspace(0, 1, 15) ** 0.7
        if xspl is None:
            xspl = np.linspace(0, 1.0, 30)
        self.xspl = xspl

        self.datatype = datatype
        self.params, values = get_defaults(datatype[0])
        self.build_profile(values[0], values[1])

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
        y0_ref=None,
        wcenter_exp=0.05,
        plot=False,
        coord="rho_poloidal",
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

        centre = deepcopy(y0)
        edge = deepcopy(y1)
        wcenter = deepcopy(self.params["wcenter"])
        wped = deepcopy(self.params["wped"])
        peaking = deepcopy(self.params["peaking"])

        def gaussian(x, A, B, x_0, w):
            return (A - B) * np.exp(-((x - x_0) ** 2) / (2 * w ** 2)) + B

        peaking2 = 1.
        if y0_ref is not None:
            if y0_ref > y0:
                print("\n Reference central value must be y0_ref < y0 ")
            else:
                peaking2 = y0 / y0_ref

        if peaking2 > 1:
            centre = y0_ref
            wcenter = wcenter - (peaking2 ** wcenter_exp - 1)

        centre = centre/peaking

        x = self.x[np.where(self.x <= 1.0)[0]]

        # baseline profile shape
        y = (centre - edge) * (1 - x ** wped) + y1

        # add central peaking
        if peaking > 1:
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y += gaussian(x, centre * (peaking - 1), 0, 0, sigma)

        # add additional peaking
        if peaking2 > 1:
            sigma = wcenter / (np.sqrt(2 * np.log(2)))
            y += gaussian(x, y[0] * (peaking2 - 1), 0, 0, sigma)

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

        yspl = DataArray(yspl, coords=[(coord, self.xspl)])
        attrs = {"datatype": self.datatype}
        name = self.datatype[1] + "_" + self.datatype[0]
        yspl.name = name
        yspl.attrs = attrs

        if plot:
            yspl.plot()

        self.yspl = yspl

def get_defaults(identifier):

    parameters = {
        "density": (12, 0.4, 1.3),
        "temperature": (4, 0.4, 1.5),
        "rotation": (4, 0.4, 1.4),
    }

    values = {
        "density": (5.e19, 1.e19),
        "temperature": (1.5e3, 50),
        "rotation": (200.e3, 10.e3),
    }

    if identifier not in parameters.keys():
        print(
            f"\n {prof_type} not in parameters' keys \n Using 'temperature' as default \n"
        )
        identifier = "temperature"

    params = parameters[identifier]
    params = {"wped": params[0], "wcenter": params[1], "peaking": params[2], "peaking2": 1.0}

    vals = values[identifier]
    return params, vals
