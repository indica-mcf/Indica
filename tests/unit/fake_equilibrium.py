"""A subclass of :py:class:`src.equilibrium.Equilibrium` which fakes
the implementation."""

from itertools import product

from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
import numpy as np

from src.equilibrium import Equilibrium


FLUX_TYPES = ["poloidal", "toroidal"]


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
    B_coeff : float
        Coefficient on B term on magnetic field variation
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
        self, Rmag=3.0, zmag=0.0, Bmax=1.0, B_coeff=1.0, B_alpha=0.001, **kwargs
    ):
        self.Rmag = Rmag
        self.zmag = zmag
        self.parameters = kwargs
        for k, v in self.DEFAULT_PARAMS.items():
            if k not in self.parameters:
                self.parameters[k] = v
        self.default_t = 0.0

    def Btot(self, R, z, t=None):
        return (
            (1 + self.parameters["B_alpha"] * t)
            * self.parameters["Bmax"]
            / (1 + self.parameters["Bcoeff"] * R)
        )

    def minor_radius(self, rho, theta, t=None, kind="toroidal"):
        if not t:
            t = self.default_t
        r = rho ** self.parameters[kind + "_n"] * (
            1 + self.parameters[kind + "_alpha"] * t
        )
        return r, t

    def flux_coords(self, R, z, t=None, kind="toroidal"):
        if not t:
            t = self.default_t
        rho = (
            (
                (R - self.Rmag) ** 2 / self.parameters[kind + "_a"] ** 2
                + (z - self.zmag) ** 2 / self.parameters[kind + "_b"] ** 2
            )
            / (1 + self.parameters[kind + "_alpha"] * t)
        ) ** (1 / (2 * self.parameters[kind + "_n"]))
        theta = np.atan((z - self.zmag) / (R - self.Rmag))
        return rho, theta, t

    def spatial_coords(self, rho, theta, t=None, kind="toroidal"):
        if not t:
            t = self.default_t
        tan_theta = np.tan(theta)
        dR = np.sign(tan_theta) * np.sqrt(
            (
                rho ** self.parameters[kind + "_n"]
                * (1 + self.parameters[kind + "_alpha"])
            )
            ** 2
            / (
                1 / self.parameters[kind + "_a"] ** 2
                + (tan_theta / self.parameters[kind + "_b"]) ** 2
            )
        )
        dz = tan_theta * dR
        return self.Rmag + dR, self.zmag + dz

    def convert_flux_coords(
        self, rho, theta, t=None, from_kind="toroidal", to_kind="poloidal"
    ):
        R, z, t = self.spatial_coords(rho, theta, t, from_kind)
        return self.flux_coords(R, z, t, to_kind)


@composite
def fake_equilibria(draw, Rmag, zmag, flux_types=FLUX_TYPES, **kwargs):
    """Generate instances of the FakeEquilibrium class, with the specified
    flux types. Parameters will be drawn from the ``floats`` strategy,
    unless explicitely specified as a keyword arguments. These
    parameters should take the form ``<flux_type>_a``,
    ``<flux_type>_b``, ``<flux_type>_n`` and ``flux_type>_alpha``. In
    addition to the flux types specified as an argument, you may
    specify the parameter values for ``Btot``.

    """
    param_types = {
        "a": floats(0.01, 10.0),
        "b": floats(0.01, 10.0),
        "n": one_of(sampled_from([1, 2, 0.5]), floats(0.2, 4.0)),
        "alpha": floats(-0.01, 0.1),
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
    return FakeEquilibrium(Rmag, zmag, **param_values)
