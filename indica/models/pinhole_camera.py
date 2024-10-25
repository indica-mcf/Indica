import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.utilities import build_dataarrays
from indica.utilities import set_axis_sci


class PinholeCamera(AbstractDiagnostic):
    """
    Object representing a pinhole camera diagnostic e.g. bolometer or SXR camera
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_radiation",
    ):
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
            "brightness": self.los_integral,
        }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Ne: DataArray = None,
        Nion: DataArray = None,
        Lz: dict = None,
        t: LabeledArray = None,
        calc_rho=False,
        sum_beamlets: bool = True,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Nion
            Ion density profiles (dims = "rho", "t", "element")
        Lz
            Cooling factor dictionary of DataArrays of each element to be included
        t
            Time (s) for remapping on equilibrium reconstruction

        Returns
        -------
        Dictionary of back-calculated quantities (as abstractreader.py)

        """
        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.interp(t=t)
            _Lz = self.plasma.lz_tot
            Lz = {}
            for elem in _Lz.keys():
                Lz[elem] = _Lz[elem].interp(t=t)
            Nion = self.plasma.ion_density.interp(t=t)
        else:
            if Ne is None or Nion is None or Lz is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t: DataArray = t
        self.Ne: DataArray = Ne
        self.Nion: DataArray = Nion
        self.Lz: dict = Lz

        elements = self.Nion.element.values

        _emissivity = []
        for ielem, elem in enumerate(elements):
            _emissivity.append(
                self.Lz[elem].sum("ion_charge") * self.Nion.sel(element=elem) * self.Ne
            )
        self.emissivity_element = xr.concat(_emissivity, "element")
        self.emissivity = self.emissivity_element.sum("element")

        self.los_integral = self.transform.integrate_on_los(
            self.emissivity,
            t=t,
            calc_rho=calc_rho,
            sum_beamlets=sum_beamlets,
        )

        self._build_bckc_dictionary()

        return self.bckc

    def plot(self, nplot: int = 1):
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        # Line-of-sight information
        self.transform.plot(np.mean(self.t))

        # Back-calculated profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.t), dtype=float))
        plt.figure()
        for i, t in enumerate(np.array(self.t)):
            if i % nplot:
                continue

            _brightness = self.bckc["brightness"].sel(t=t, method="nearest")
            if "beamlet" in _brightness.dims:
                plt.fill_between(
                    _brightness.channel,
                    _brightness.max("beamlet"),
                    _brightness.min("beamlet"),
                    color=cols_time[i],
                    alpha=0.5,
                )
                brightness = _brightness.mean("beamlet")
            else:
                brightness = _brightness
            brightness.plot(label=f"t={t:1.2f} s", color=cols_time[i])
        set_axis_sci()
        plt.title(self.name.upper())
        plt.xlabel("Channel")
        plt.ylabel("Measured brightness (W/m^2)")
        plt.legend()

        # Local emissivity profiles
        plt.figure()
        for i, t in enumerate(np.array(self.t)):
            if i % nplot:
                continue
            plt.plot(
                self.emissivity.rhop,
                self.emissivity.sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        set_axis_sci()
        plt.xlabel("rho")
        plt.ylabel("Local radiated power (W/m^3)")
        plt.legend()
