import math

import matplotlib.pylab as plt
import numpy as np
from scipy.constants import electron_mass
from scipy.constants import elementary_charge
from scipy.constants import epsilon_0
from scipy.constants import speed_of_light
import xarray as xr
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
            {"R": self.transform.R, "z": self.transform.z}
        )
        self.ne_remapped = ne_remapped

        Br, Bz, Bt, _ = equilibrium.Bfield(self.transform.R, self.transform.z, t=t)
        Bx = Br * np.cos(self.transform.phi) - Bt * np.sin(self.transform.phi)
        By = Br * np.sin(self.transform.phi) + Bt * np.cos(self.transform.phi)
        Bx_l = xr.zeros_like(Bx).transpose(
            self.transform.x1_name, "beamlet", self.transform.x2_name
        )
        By_l = xr.zeros_like(By).transpose(
            self.transform.x1_name, "beamlet", self.transform.x2_name
        )
        Bz_l = xr.zeros_like(Bz).transpose(
            self.transform.x1_name, "beamlet", self.transform.x2_name
        )
        Bl = xr.zeros_like(self.ne_remapped).transpose(
            self.transform.x1_name, "beamlet", self.transform.x2_name
        )
        for i, x1 in enumerate(self.transform.x1):
            for j, beamlet in enumerate(self.transform.beamlets):
                _bx = Bx.sel({self.transform.x1_name: x1, "beamlet": beamlet})
                _by = By.sel({self.transform.x1_name: x1, "beamlet": beamlet})
                _bz = Bz.sel({self.transform.x1_name: x1, "beamlet": beamlet})
                dx = self.transform.beamlet_direction_x[i, j]
                dy = self.transform.beamlet_direction_y[i, j]
                dz = self.transform.beamlet_direction_z[i, j]
                uf = np.linalg.norm((dx, dy, dz))
                dx /= uf
                dy /= uf
                dz /= uf
                Bx_l.loc[i, j, :] = dx.data * _bx.data
                By_l.loc[i, j, :] = dy.data * _by.data
                Bz_l.loc[i, j, :] = dz.data * _bz.data
                Bl.loc[i, j, :] = np.dot((dx, dy, dz), (_bx, _by, _bz))
        # Bl = (Bx * dx) + (By * dy) + (Bz * dz)
        Bl = Bl.assign_coords({"R": self.transform.R, "z": self.transform.z})
        Bl.name = "Longitudinal Magnetic Field (T)"
        self.Br = Br
        self.Bz = Bz
        self.Bt = Bt
        self.Bx = Bx
        self.By = By
        self.Bx_l = Bx_l
        self.By_l = By_l
        self.Bz_l = Bz_l
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
        los_integral_dphi = (
            self.Dphi.sum(("los_position", "beamlet"))
            * self.transform.dl
            * self.transform.passes
        )
        los_integral_dphi.name = "Faraday Rotation Integrated (rad)"
        self.los_integral_dphi = los_integral_dphi

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
