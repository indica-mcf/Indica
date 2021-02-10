"""A subclass of :py:class:`indica.equilibrium.Equilibrium` which fakes
the implementation."""

from itertools import product
from unittest.mock import MagicMock

from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
import numpy as np
from xarray import DataArray

from indica.equilibrium import Equilibrium


FLUX_TYPES = ["poloidal", "toroidal"]


@composite
def flux_types(draw):
    return draw(sampled_from(FLUX_TYPES))


class FakeEquilibrium(Equilibrium):
    """A class which fakes the behaviour of an Equilibrium object.  Flux
    surface and magnetic fields are taken to vary in an elliptical profile
    away from the magnetif axis.

    Fluxes have form $$r^2 = \\frac{(R-R_{mag})^2}{a^2} +
    \\frac{(z-z_{mag})^2}{b^2},$$ where $r = \\rho^n(1 + \\alpha t)$ and $a$, $b$,
    $n$ and $\\alpha$ are parameters specified by the user at instantiation.

    $B_{tot}$ varies according to a different equation:
    $$B_{tot} = \\frac{(1 + \\alpha t) a}{1 + bR} + (z - z_{mag}).$$

    Paramter values may be specified at instantiation using
    keyword-arguments of the constructor. There are also default
    values available for flux kinds ``poloidal``, ``toroidal``, and
    the total magnetic field strength.

    Parameters
    ----------
    Rmag : float
        Major radius of the magnetic axis
    zmag : float
        Vertical position of the magnetic axis
    kwargs : Dict[str, float]
        Values for parameters describing the equilibrium profile. Keys take the
        form ``<flux_type>_<parameter_name>``. The ``<flux_type>`` may be any
        string which will be accpeted as a ``kind`` argument in methods such as
        :py:meth:`flux_coords``, or ``Btot`` if the paremeter is describing the
        profile of total magnetic field strength. The ``<parameter_name>`` may
        be ``a``, ``b``, ``n``, or ``alpha``.

    """

    DEFAULT_PARAMS = {
        "poloidal_a": 0.5,
        "poloidal_b": 1.0,
        "poloidal_n": 1,
        "poloidal_alpha": 0.01,
        "toroidal_a": 0.7,
        "toroidal_b": 1.4,
        "toroidal_n": 1,
        "toroidal_alpha": -0.00005,
        "Btot_a": 1.0,
        "Btot_b": 1.0,
        "Btot_alpha": 0.001,
    }

    def __init__(
        self,
        Rmag=3.0,
        zmag=0.0,
        default_t=DataArray([0.0, 5e3], dims="t"),
        Bmax=1.0,
        **kwargs
    ):
        ones = DataArray(np.ones_like(default_t), coords=[("t", default_t)])
        self.rmag = np.abs(Rmag) * ones
        self.zmag = zmag * ones
        self.parameters = kwargs
        for k, v in self.DEFAULT_PARAMS.items():
            if k not in self.parameters:
                self.parameters[k] = v
        self.default_t = default_t
        self.prov_id = MagicMock()
        self.provenance = MagicMock()
        self._session = MagicMock()

    def Btot(self, R, z, t=None):
        if t is None:
            t = self.default_t
            zmag = self.zmag
        else:
            zmag = self.zmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
        return (
            (1 + self.parameters["Btot_alpha"] * t)
            * self.parameters["Btot_a"]
            / (1 + self.parameters["Btot_b"] * R)
            + z
            - zmag,
            t,
        )

    def enclosed_volume(self, rho, t=None, kind="poloidal"):
        if t is None:
            t = self.default_t
            rmag = self.rmag
        else:
            rmag = self.rmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
        a = self.parameters[kind + "_a"]
        b = self.parameters[kind + "_b"]
        n = self.parameters[kind + "_n"]
        alpha = self.parameters[kind + "_alpha"]
        vol = 2 * np.pi ** 2 * a * b * rho ** (2 * n) * (1 + alpha * t) ** 2 * rmag
        return vol, t

    def invert_enclosed_volume(self, vol, t=None, kind="poloidal"):
        if t is None:
            t = self.default_t
            rmag = self.rmag
        else:
            rmag = self.rmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
        a = self.parameters[kind + "_a"]
        b = self.parameters[kind + "_b"]
        n = self.parameters[kind + "_n"]
        alpha = self.parameters[kind + "_alpha"]
        rho = (vol / (2 * np.pi ** 2 * a * b * (1 + alpha * t) ** 2 * rmag)) ** (
            0.5 / n
        )
        return rho, t

    def minor_radius(self, rho, theta, t=None, kind="poloidal"):
        if t is None:
            t = self.default_t
        r = rho ** self.parameters[kind + "_n"] * (
            1 + self.parameters[kind + "_alpha"] * t
        )
        return r, t

    def flux_coords(self, R, z, t=None, kind="poloidal"):
        if t is None:
            t = self.default_t
            rmag = self.rmag
            zmag = self.zmag
        else:
            rmag = self.rmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
            zmag = self.zmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
        rho = (
            (
                (R - rmag) ** 2 / self.parameters[kind + "_a"] ** 2
                + (z - zmag) ** 2 / self.parameters[kind + "_b"] ** 2
            )
            / (1 + self.parameters[kind + "_alpha"] * t) ** 2
        ) ** (1 / (2 * self.parameters[kind + "_n"]))
        theta = np.arctan2((z - zmag), (R - rmag))
        return rho, theta, t

    def spatial_coords(self, rho, theta, t=None, kind="poloidal"):
        if t is None:
            t = self.default_t
            rmag = self.rmag
            zmag = self.zmag
        else:
            rmag = self.rmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
            zmag = self.zmag.interp(
                t=t, method="nearest", kwargs={"fill_value": "extrapolate"}
            )
        tan_theta = np.tan(theta)
        dR = np.sign(np.cos(theta)) * np.sqrt(
            (
                rho ** self.parameters[kind + "_n"]
                * (1 + self.parameters[kind + "_alpha"] * t)
            )
            ** 2
            / (
                1 / self.parameters[kind + "_a"] ** 2
                + (tan_theta / self.parameters[kind + "_b"]) ** 2
            )
        )
        dz = tan_theta * dR
        return rmag + dR, zmag + dz, t

    def convert_flux_coords(
        self, rho, theta, t=None, from_kind="poloidal", to_kind="toroidal"
    ):
        R, z, t = self.spatial_coords(rho, theta, t, from_kind)
        return self.flux_coords(R, z, t, to_kind)


@composite
def fake_equilibria(
    draw,
    Rmag,
    zmag,
    default_t=DataArray([0.0, 500.0], dims="t"),
    flux_types=FLUX_TYPES,
    **kwargs
):
    """Generate instances of the FakeEquilibrium class, with the specified
    flux types. Parameters will be drawn from the ``floats`` strategy,
    unless explicitely specified as a keyword arguments. These
    parameters should take the form ``<flux_type>_a``,
    ``<flux_type>_b``, ``<flux_type>_n`` and ``<flux_type>_alpha``. In
    addition to the flux types specified as an argument, you may
    specify the parameter values for ``Btot``.

    """
    param_types = {
        "a": floats(-0.9, 9.0).map(lambda x: x + 1.0),
        "b": floats(-0.9, 9.0).map(lambda x: x + 1.0),
        "n": one_of(sampled_from([1, 2, 0.5]), floats(0.2, 4.0)),
        "alpha": floats(-0.02, 0.09).map(lambda x: x + 0.01),
    }
    param_values = kwargs
    for flux, param in product(flux_types, param_types):
        param_name = flux + "_" + param
        if param_name not in param_values:
            param_values[param_name] = draw(param_types[param])
    if "Btot_a" in flux_types and "Btot_a" not in param_values:
        param_values["Btot_a"] = draw(floats(1e-3, 1e3))
    if "Btot_b" in flux_types and "Btot_b" not in param_values:
        param_values["Btot_b"] = draw(floats(1e-5, 1e-2))
    if "Btot_alpha" in flux_types and "Btot_alpha" not in param_values:
        param_values["Btot_alpha"] = draw(floats(-1e-3, 1e-3))
    return FakeEquilibrium(Rmag, zmag, default_t, **param_values)
