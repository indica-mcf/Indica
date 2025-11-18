import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
from indica.operators import PowerLoss
from indica.utilities import assign_datatype
from indica.utilities import build_dataarrays
from indica.utilities import set_axis_sci


class PinholeCamera(AbstractDiagnostic):
    """
    Object representing a pinhole camera diagnostic e.g. bolometer or SXR camera
    """

    def __init__(
        self,
        name: str,
        power_loss: dict[str, PowerLoss],
        instrument_method: str = "get_radiation",
    ):
        self.transform: LineOfSightTransform
        self.name = name
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]
        self.power_loss = power_loss

        self.t: DataArray
        self.Te: DataArray
        self.Ne: DataArray
        self.Nion: DataArray
        self.Nh: DataArray
        self.Lz: dict
        self.fz: dict

    def _build_bckc_dictionary(self):
        bckc = {
            "t": self.t,
            "channel": np.arange(len(self.transform.x1)),
            "location": self.transform.origin,
            "direction": self.transform.direction,
            "brightness": self.los_integral,
        }
        if "beamlet" in self.los_integral.coords:
            bckc["beamlet"] = self.los_integral.beamlet

        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
        self,
        Te: DataArray = None,
        Ne: DataArray = None,
        Nion: DataArray = None,
        Nh: DataArray = None,
        fz: dict = None,
        t: LabeledArray = None,
        calc_rho=False,
        sum_beamlets: bool = True,
        full_run: bool = False,
        **kwargs,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Te
            Electron temperature profile (dims = "rho", "t")
        Ne
            Electron density profile (dims = "rho", "t")
        Nion
            Ion density profiles (dims = "rho", "t", "element")
        Nh
            Neutral main ion density profiles (dims = "rho", "t")
        fz
            Fractional abundance dictionary for each element to be included
        t
            Time (s) for remapping on equilibrium reconstruction

        Returns
        -------
        Dictionary of back-calculated quantities (as datareader.py)

        """

        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Ne = self.plasma.electron_density.interp(t=t)
            Nion = self.plasma.ion_density.interp(t=t)
            Nh = self.plasma.neutral_density.interp(t=t)
            Te = self.plasma.electron_temperature.interp(t=t)
            fz = self.plasma.fz
        else:
            if Te is None or Ne is None or Nion is None or fz is None or t is None:
                raise ValueError("Give inputs or assign plasma class!")

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nion = Nion
        self.Nh = Nh
        self.fz = fz

        _isfinite = np.isfinite(self.Ne.interp(rhop=self.Nion.rhop)) * np.isfinite(
            self.Te.interp(rhop=self.Nion.rhop)
        )

        self.Lz = {}
        emissivity_element = []
        elements = self.Nion.element.values

        for elem in elements:
            Lz = []
            for _t in t:
                fz = self.fz[elem].sel(t=_t)

                _Lz = self.power_loss[elem](
                    Te.sel(t=_t),
                    fz,
                    Ne=Ne.sel(t=_t),
                    Nh=Nh.sel(t=_t),
                )
                Lz.append(_Lz)

            self.Lz[elem] = xr.concat(Lz, "t")

            _emissivity = (
                self.Lz[elem].sum("ion_charge").interp(rhop=self.Nion.rhop)
                * self.Nion.sel(element=elem)
                * self.Ne.interp(rhop=self.Nion.rhop)
            )
            emissivity_element.append(xr.where(_isfinite, _emissivity, np.nan))

        self.emissivity_element = xr.concat(emissivity_element, "element")
        emissivity = self.emissivity_element.sum("element")
        self.emissivity = xr.where(
            np.isfinite(Ne.interp(rhop=self.Nion.rhop)), emissivity, np.nan
        )
        assign_datatype(self.emissivity, "total_radiation")

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
        plt.legend()

        # Local emissivity profiles
        plt.figure()
        for i, t in enumerate(np.array(self.t)):
            if i % nplot:
                continue
            if hasattr(self.emissivity, "rhop"):
                plt.plot(
                    self.emissivity.rhop,
                    self.emissivity.sel(t=t),
                    color=cols_time[i],
                    label=f"t={t:1.2f} s",
                )
            else:
                self.emissivity.sel(t=t).plot(
                    label=f"t={t:1.2f} s",
                )
                self.transform.plot(orientation="Rz", figure=False, t=t)
                break

        set_axis_sci()
        if hasattr(self.emissivity, "rhop"):
            plt.xlabel("Rho-poloidal")
            plt.ylabel("(W/m^3)")
        else:
            plt.axis("equal")
        plt.title("Local radiated power")
        plt.legend()
