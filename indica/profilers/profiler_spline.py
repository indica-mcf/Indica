import os
import matplotlib.pylab as plt
import yaml
from scipy.interpolate import CubicSpline, PchipInterpolator
import indica
from indica.profilers.profiler_base import ProfilerBase
import numpy as np
from indica.utilities import format_coord, format_dataarray


class ProfilerMonoSpline(ProfilerBase):
    def __init__(self,
                 datatype: str = "electron_temperature",
                 coord="poloidal",
                 xspl: np.ndarray = None,
                 parameters: dict = None,
                ):

        super().__init__(parameters)
        self.coord = f"rho_{coord}"
        self.datatype = datatype

        if xspl is None:
            self.xspl = format_coord(np.linspace(0, 1.0, 30), self.coord)
        else:
            self.xspl = xspl

        if parameters is None:
            self.set_parameters(**get_defaults_for_profiler_spline(datatype=datatype, config_name="profiler_monospline"))
        else:
            self.set_parameters(**parameters)

        self.x = self.parameters["xknots"]

        if len(self.x) != len(self.parameters)-1:  # don't count xknot in length of y params
            raise ValueError(f"number of y parameters: {len(self.parameters)-1} != number of xknots: {len(self.x)}")

    def __call__(self, *args, **kwargs):

        self.y = [_y for key, _y in dict(sorted(self.parameters.items())).items() if "xknot" not in key]
        self.spline = PchipInterpolator(self.x, self.y)
        _yspl = self.spline(self.xspl)
        coords = {self.coord: self.xspl}
        self.ydata = format_dataarray(_yspl, self.datatype, coords=coords)
        return self.ydata


class ProfilerCubicSpline(ProfilerBase):
    def __init__(self,
                 datatype: str = "electron_temperature",
                 coord="poloidal",
                 xspl: np.ndarray = None,
                 parameters: dict = None,
                ):

        super().__init__(parameters)
        self.coord = f"rho_{coord}"
        self.datatype = datatype

        if xspl is None:
            self.xspl = format_coord(np.linspace(0, 1.0, 30), self.coord)
        else:
            self.xspl = xspl

        if parameters is None:
            self.set_parameters(**get_defaults_for_profiler_spline(datatype=datatype, config_name="profiler_cubicspline"))
        else:
            self.set_parameters(**parameters)

        self.x = self.parameters["xknots"]

        if len(self.x) != len(self.parameters)-1:  # don't count xknot in length of y params
            raise ValueError(f"number of y parameters: {len(self.parameters)-1} != number of xknots: {len(self.x)}")

    def __call__(self, *args, **kwargs):

        _yshape = [_y for key, _y in dict(sorted(self.parameters.items())).items() if "shape" in key]

        y = [self.parameters["y0"]]
        for index, shapevalue in enumerate(_yshape):
            y.append(shapevalue*y[index])
        y.append(self.parameters["y1"])

        self.spline = CubicSpline(self.x, y, bc_type="clamped")
        _yspl = self.spline(self.xspl)
        coords = {self.coord: self.xspl}
        self.ydata = format_dataarray(_yspl, self.datatype, coords=coords)
        return self.ydata


def get_defaults_for_profiler_spline(
    datatype="electron_temperature", config_name="profiler_monospline"
):
    """
    Loads config for default parameter values
    """
    path = os.path.join(
        os.path.dirname(indica.__file__), f"configs/profilers/{config_name}.yaml"
    )
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    return cfg[datatype]


if __name__ == "__main__":

    plt.figure()
    xknots_mspline = [0.0, 0.15, 0.3, 1.0, 1.05]
    xknots_cspline = [0.0, 0.15, 0.3,  1.0, ]

    _prof = ProfilerMonoSpline(datatype="ion_temperature", parameters={"y0":1.1, "y1":0.8, "y2":0.6,
                          "y3":0.01, "y4":0.001, "xknots":xknots_mspline}, ).plot(fig=False, color="red", marker="o", label="monospline1")

    _prof = ProfilerMonoSpline(datatype="ion_temperature", parameters={"y0":1, "y1":1.05, "y2":0.8,
                          "y3":0.01, "y4":0.001, "xknots":[0.0, 0.2, 0.4, 1.0, 1.05]}, ).plot(fig=False, color="red", marker="x", label="monospline1")

    _prof = ProfilerCubicSpline(datatype="ion_temperature",
                                parameters={"y0":1.1, "y1":0.01, "shape1":0.8, "shape2":0.75,  "xknots":xknots_cspline},
                                ).plot(fig=False, color="blue", marker="o", label="cubicspline1")

    _prof = ProfilerCubicSpline(datatype="ion_temperature",
                                parameters={"y0":1, "y1":0.01, "shape1":1.05, "shape2":0.8,
                            "xknots":xknots_cspline}, ).plot(fig=False, color="blue", marker="x", label="cubicspline1")
    plt.legend()
    plt.show(block=True)
