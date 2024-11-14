
import matplotlib.pyplot as plt

from indica.profilers.profiler_spline import ProfilerMonoSpline, ProfilerCubicSpline

xknots_mspline = [0.0, 0.15, 0.3, 1.0, 1.05]
xknots_cspline = [
    0.0,
    0.15,
    0.3,
    1.0,
]

_prof = ProfilerMonoSpline(
    datatype="ion_temperature",
    parameters={
        "y0": 1.1,
        "y1": 0.8,
        "y2": 0.6,
        "y3": 0.01,
        "y4": 0.001,
        "xknots": xknots_mspline,
    },
)
_prof.plot(fig=False, color="red", marker="o", label="monospline1")

_prof = ProfilerMonoSpline(
    datatype="ion_temperature",
    parameters={
        "y0": 1,
        "y1": 1.05,
        "y2": 0.8,
        "y3": 0.01,
        "y4": 0.001,
        "xknots": [0.0, 0.2, 0.4, 1.0, 1.05],
    },
)
_prof.plot(fig=False, color="red", marker="x", label="monospline1")

_prof = ProfilerCubicSpline(
    datatype="ion_temperature",
    parameters={
        "y0": 1.1,
        "y1": 0.01,
        "shape1": 0.8,
        "shape2": 0.75,
        "xknots": xknots_cspline,
    },
)
_prof.plot(fig=False, color="blue", marker="o", label="cubicspline1")

_prof = ProfilerCubicSpline(
    datatype="ion_temperature",
    parameters={
        "y0": 1,
        "y1": 0.01,
        "shape1": 1.05,
        "shape2": 0.8,
        "xknots": xknots_cspline,
    },
)
_prof.plot(fig=False, color="blue", marker="x", label="cubicspline1")
plt.legend()
plt.show(block=True)
