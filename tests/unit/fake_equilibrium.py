"""A subclass of :py:class:`src.equilibrium.Equilibrium` which fakes
the implementation."""

import numpy as np

from src.equilibrium import Equilibrium


class FakeEquilibrium(Equilibrium):
    """A class which fakes the behaviour of an Equilibrium object.  Flux
    surface and magnetic fields are taken to vary in an elliptical profile
    away from the magnetif axis.

    """

    DEFAULT_PARAMS = {
        "poloidal_a": 0.5,
        "poloidal_b": 1.0,
        "poloidal_n": 1,
        "poloidal_alpha": 0.01,
        "toroidal_a": 0.7,
        "toroidal_b": 1.44,
        "toroidal_n": 0.5,
        "toroidal_alpha": -0.00005,
        "Btot_a": 0.6,
        "Btot_b": 1.21,
        "Btot_n": 1,
        "Btot_alpha": 0.001,
    }

    def __init__(self, Rmag=3.0, zmag=0.0, **kwargs):
        self.Rmag = Rmag
        self.zmag = zmag
        self.parameters = kwargs
        for k, v in self.DEFAULT_PARAMS.items():
            if k not in self.parameters:
                self.parameters[k] = v
        self.default_t = 0.0

    def Btot(self, R, z, t=None):
        return self.flux_coords(R, z, t, "Btot")

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
