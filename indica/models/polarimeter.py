import math

import matplotlib.pylab as plt
import numpy as np
from scipy.constants import electron_mass
from scipy.constants import elementary_charge
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light
from xarray import DataArray

from indica import Equilibrium
from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams


class Polarimeter(AbstractDiagnostic):
    """
    Object representing a polarimeter diagnostic
    """

    Bl: DataArray
    Ne: DataArray
    Dphi: DataArray
    ne_remapped: DataArray
    los_integral_dphi: DataArray

    def __init__(self, name: str, instrument_method="get_polarimetry"):
        self.transform: LineOfSightTransform
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "location": self.transform.origin,
            "direction": self.transform.direction,
            "dphi": self.los_integral_dphi,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        laser_wavelength: float,
        Ne: DataArray = None,
        t: LabeledArray = None,
        equilibrium: Equilibrium = None,
        calc_rho=False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile
        t

        Returns
        -------

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            equilibrium = self.plasma.equilibrium
            Ne = self.plasma.electron_density.interp(t=t)
        if Ne is None or equilibrium is None:
            raise ValueError("Give inputs or assign plasma class!")
        self.t: DataArray = t
        self.Ne: DataArray = Ne
        self.equilibrium: Equilibrium = equilibrium

        ne_remapped = self.transform.map_profile_to_los(
            self.Ne,
            t=self.t,
            calc_rho=calc_rho,
        )
        ne_remapped = ne_remapped.assign_coords(
            {
                "R": (("channel", "beamlet", "los_position"), self.transform.R.data),
                "z": (("channel", "beamlet", "los_position"), self.transform.z.data),
            }
        )
        self.ne_remapped = ne_remapped

        Br, Bz, _, _ = equilibrium.Bfield(self.transform.R, self.transform.z, t=t)
        unit_factor = np.linalg.norm(self.transform.direction, axis=1)
        Bl = (Br * self.transform.direction_x / unit_factor) + (
            Bz * self.transform.direction_z / unit_factor
        )
        Bl = Bl.assign_coords(
            {
                "R": (("channel", "beamlet", "los_position"), self.transform.R.data),
                "z": (("channel", "beamlet", "los_position"), self.transform.z.data),
            }
        )
        Bl.name = "Longitudinal Magnetic Field (T)"
        self.Bl = Bl

        Dphi = (
            (
                (elementary_charge**3)
                / (
                    2**3
                    * math.pi**2
                    * epsilon_0
                    * electron_mass**2
                    * speed_of_light**3
                )
            )
            * (laser_wavelength**2)
            * self.ne_remapped
            * self.Bl
        )
        Dphi.name = "Faraday Rotation (rad)"
        self.Dphi = Dphi
        self.los_integral_dphi = (
            self.Dphi.sum(("los_position", "beamlet"))
            * self.transform.dl
            * self.transform.passes
        )

        self._build_bckc_dictionary()
        return self.bckc

    def plot(self, nplot: int = 1):
        set_plot_rcparams("profiles")
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        # Line-of-sight information
        self.transform.plot()
        plt.figure()
        _value = self.bckc["dphi"]
        if "beamlet" in _value.dims:
            plt.fill_between(
                _value.t,
                _value.max("beamlet"),
                _value.min("beamlet"),
                alpha=0.5,
            )
            value = _value.mean("beamlet")
        else:
            value = _value
        value.plot()
        set_axis_sci()
        plt.title(self.name.upper())
        plt.xlabel("Time (s)")
        plt.ylabel("Measured Faraday Rotation (rad)")
        plt.legend()
