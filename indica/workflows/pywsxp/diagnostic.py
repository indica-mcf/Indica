from copy import deepcopy
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Union

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from xarray import concat
from xarray import DataArray
from xarray import Dataset

from indica.converters import LineOfSightTransform
from indica.converters.abstractconverter import CoordinateTransform
from indica.converters.transect import TransectCoordinates
from indica.datatypes import DATATYPES
from indica.datatypes import UNITS
from indica.models import PinholeCamera
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.plasma import Plasma


@dataclass
class Diagnostic:
    instrument: str
    quantity: str
    measurement: DataArray
    model: AbstractDiagnostic
    channels: Optional[list[int]] = None
    weight: Optional[float] = 1.0
    rescale_factor: float = 1.0
    ignored_channels: Optional[List[int]] = None


def calc_closest_approach(transform: CoordinateTransform, t: float):
    """Calculate the closest approach to the magnetic axis"""
    if not hasattr(transform, "equilibrium"):
        raise Exception("Set equilibrium object to calculate impact parameter")
    if isinstance(transform, TransectCoordinates):
        rhop, theta, _ = transform.equilibrium.flux_coords(
            transform.R, transform.z, t=t
        )
        return Dataset(
            {
                "x": transform.x,
                "y": transform.y,
                "z": transform.z,
                "R": transform.R,
                "rhop": rhop,
                "theta": theta,
            }
        )
    Rmag = transform.equilibrium.rmag.interp(t=t)
    zmag = transform.equilibrium.zmag.interp(t=t)
    impact = []
    index = []
    x = []
    y = []
    z = []
    R = []
    rhop = []
    theta = []
    for ch in transform.x1:
        x_mean = transform.x.sel(channel=ch).mean("beamlet")
        y_mean = transform.y.sel(channel=ch).mean("beamlet")
        z_mean = transform.z.sel(channel=ch).mean("beamlet")
        R_mean = np.sqrt(x_mean**2 + y_mean**2)
        distance = np.sqrt((R_mean - Rmag) ** 2 + (z_mean - zmag) ** 2)
        _index = np.unravel_index(distance.argmin(), distance.shape)
        _index_temp = distance.argmin()
        index.append(_index_temp)
        impact.append(distance[_index])
        x.append(x_mean[_index])
        y.append(y_mean[_index])
        z.append(z_mean[_index])
        R.append(R_mean[_index])
        rhop_mean, theta_mean, _ = transform.equilibrium.flux_coords(
            R_mean[_index], z_mean[_index], t=t
        )
        rhop.append(rhop_mean)
        theta.append(theta_mean)

    impact = Dataset(
        {
            "index": concat(index, "channel"),
            "value": concat(impact, "channel"),
            "x": concat(x, "channel"),
            "y": concat(y, "channel"),
            "z": concat(z, "channel"),
            "R": concat(R, "channel"),
            "rhop": concat(rhop, "channel"),
            "theta": concat(theta, "channel"),
        }
    )

    return impact


def calc_weighted_rho(diagnostic: PinholeCamera, *args, **kwargs):
    r"""Calculate weighted rho of emission for a :py:`PinholeCamera`

    :latex:`\bar{\rho} = \frac{\int{\rho\psi}dl}{\int{\psi}dl}`
    :latex:`\delta_{\rho} = \frac{\int{\rho^{2}\psi}dl}{\int{psi}dl} - \left(\bar{\rho}^{2}\right)`
    """
    try:
        assert isinstance(diagnostic, PinholeCamera)
    except AssertionError:
        raise UserWarning(
            f"Weighted Rho currently only works for PinholeCamera"
            f", not {type(diagnostic)}"
        )
    transform = getattr(diagnostic, "transform")
    try:
        assert isinstance(transform, LineOfSightTransform)
    except AssertionError:
        raise UserWarning(
            f"Weighted Rho only makes sense for LineOfSightTransform"
            f", not {type(transform)}"
        )

    if not hasattr(diagnostic, "emissivity_element"):
        diagnostic(*args, **kwargs)  # type: ignore
    emissivity_mapped = transform.map_profile_to_los(
        diagnostic.emissivity_element, t=diagnostic.t, calc_rho=False
    ).assign_coords({"element": diagnostic.emissivity_element.element})
    rho_los = transform.convert_to_rho_theta(t=diagnostic.t)[0]
    weighted_rho = (
        (rho_los * emissivity_mapped).sum(("los_position", "beamlet"))
        * transform.dl
        * transform.passes
    ) / (
        emissivity_mapped.sum(("los_position", "beamlet"))
        * transform.dl
        * transform.passes
    )
    delta_weighted_rho = np.sqrt(
        (
            (
                ((rho_los**2) * emissivity_mapped).sum(("los_position", "beamlet"))
                * transform.dl
                * transform.passes
            )
            / (
                emissivity_mapped.sum(("los_position", "beamlet"))
                * transform.dl
                * transform.passes
            )
        )
        - (weighted_rho**2)
    )

    return weighted_rho, delta_weighted_rho, rho_los


def plot_los(
    plasma: Plasma,
    diagnostic: Diagnostic,
    time: float,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    transform = deepcopy(diagnostic.measurement.transform)
    rhop_bnd = transform.get_equilibrium_boundaries(time)[-1]
    if diagnostic.quantity == "brightness":
        diagnostic.model.set_plasma(plasma)
        diagnostic.model(t=time)
        emissivity = diagnostic.model.emissivity.assign_attrs(  # type: ignore
            {
                "long_name": DATATYPES[diagnostic.quantity][0],
                "units": UNITS[DATATYPES[diagnostic.quantity][1]],
            }
        )
        if "rhop" in emissivity.dims:
            emissivity = emissivity.interp(rhop=rhop_bnd)
        emissivity.where(emissivity.rhop <= 1.0).plot(x="R", cmap=cmap, ax=ax)
    elif diagnostic.quantity == "conc":
        (
            plasma.ion_density.sel(element=diagnostic.model.element)  # type: ignore
            / plasma.electron_density
        ).assign_attrs(
            {
                "long_name": DATATYPES["concentration"][0],
                "units": UNITS[DATATYPES["concentration"][1]],
            }
        ).interp(
            t=time, rhop=rhop_bnd
        ).plot(
            x="R", cmap=cmap, ax=ax
        )
    elif diagnostic.quantity == "zeff_avrg":
        plasma.zeff.sum("element").interp(t=time, rhop=rhop_bnd).plot(
            x="R", cmap=cmap, ax=ax
        )
    if "channel" in diagnostic.measurement.coords:
        transform.R = transform.R.sel(channel=diagnostic.measurement.channel)
        transform.z = transform.z.sel(channel=diagnostic.measurement.channel)
    transform.plot(orientation="Rz", figure=False)
    ax.set_title(diagnostic.instrument)
    return ax


def plot_measurement_model_comparison(
    plasma: Plasma,
    diagnostic: Diagnostic,
    time: float,
    rescale_factor: Union[int, float],
    ax: Optional[Axes] = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    closest_approach = calc_closest_approach(diagnostic.measurement.transform, t=time)
    rhop, theta = closest_approach["rhop"], closest_approach["theta"]
    if diagnostic.instrument.endswith("h"):
        pos = theta >= 0
    else:
        pos = (theta <= np.pi / 2) & (theta >= -np.pi / 2)
    x = rhop.where(pos, other=-1.0 * rhop)
    diagnostic.model.set_plasma(plasma)
    measurement = diagnostic.measurement.interp(t=time) * rescale_factor
    if "channel" not in measurement.dims:
        measurement = measurement.expand_dims({"channel": [0]}, axis=0)
    measurement = measurement.assign_coords(
        {"rhop": ("channel", x.sel(channel=measurement.channel).data)}
    )
    model = diagnostic.model(t=time)[diagnostic.quantity]
    if "channel" not in model.dims:
        model = model.expand_dims({"channel": [0]}, axis=0)
    model = model.assign_coords({"rhop": ("channel", x.data)})
    try:
        data = measurement.sortby("rhop")
        ax.errorbar(
            x=data.rhop,
            y=data.values,
            yerr=data.error,
            marker="+" if np.all(data.error == 0.0) else " ",
            linestyle="-",
            capsize=5.0,
            label=f"t={float(time):.2f}s, s={rescale_factor:.2f}",
        )
        model.sortby("rhop").plot.line(
            x="rhop",
            marker="x",
            linestyle="--",
            ax=ax,
        )
    except Exception as e:
        print(e)
        print(f"{diagnostic.instrument=}, t={time:.3f}")
    ax.legend()
    ax.set_title("{}: {}".format(diagnostic.instrument, diagnostic.quantity))
    return ax
