import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.utilities import assign_datatype
from indica.utilities import set_axis_sci


class Bolometer(DiagnosticModel):
    """
    Object representing a bolometer camera diagnostic
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_radiation",
    ):

        self.name = name
        self.instrument_method = instrument_method
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

    def _build_bckc_dictionary(self):
        self.bckc = {}

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if quant == "brightness":
                quantity = quant
                self.bckc[quantity] = self.los_integral
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "transform": self.los_transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
                assign_datatype(self.bckc[quantity], datatype)
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

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
                t = self.plasma.t
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

        self.los_integral = self.los_transform.integrate_on_los(
            self.emissivity,
            t=t,
            calc_rho=calc_rho,
            sum_beamlets=sum_beamlets,
        )

        self._build_bckc_dictionary()

        return self.bckc

    def plot(self, tplot: float = None, nplot: int = 1):
        if len(self.bckc) == 0:
            print("No model results to plot")
            return

        if tplot is not None:
            tplot = float(self.t.sel(t=tplot, method="nearest"))
        else:
            tplot = float(self.t.sel(t=self.t.mean(), method="nearest"))

        # Line-of-sight information
        self.los_transform.plot(tplot)

        # Back-calculated profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.t), dtype=float))
        plt.figure()
        for i, t in enumerate(self.t.values):
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
        for i, t in enumerate(self.t.values):
            if i % nplot:
                continue
            plt.plot(
                self.emissivity.rho_poloidal,
                self.emissivity.sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        set_axis_sci()
        plt.xlabel("rho")
        plt.ylabel("Local radiated power (W/m^3)")
        plt.legend()


def example_run(
    pulse: int = None,
    diagnostic_name: str = "bolo_xy",
    origin: LabeledArray = None,
    direction: LabeledArray = None,
    plasma=None,
    plot=False,
    tplot=None,
    nchannels: int = 11,
):

    if plasma is None:
        from indica.equilibrium import fake_equilibrium

        plasma = example_plasma(pulse=pulse)
        machine_dims = plasma.machine_dimensions
        equilibrium = fake_equilibrium(
            tstart=plasma.tstart,
            tend=plasma.tend,
            dt=plasma.dt / 2.0,
            machine_dims=machine_dims,
        )
        plasma.set_equilibrium(equilibrium)

    # return plasma
    # Create new interferometers diagnostics
    if origin is None or direction is None:
        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.0
        los_end[:, 1] = np.linspace(-0.2, -1, nchannels)
        los_end[:, 2] = 0.0
        los_start = np.array([[1.5, 0, 0]] * los_end.shape[0])
        origin = los_start
        direction = los_end - los_start

        # los_end = np.full((nchannels, 3), 0.0)
        # los_end[:, 0] = 0.17
        # los_end[:, 1] = 0.0
        # los_end[:, 2] = np.linspace(0.6, -0.6, nchannels)
        # los_start = np.array([[1.0, 0, 0]] * los_end.shape[0])
        # origin = los_start
        # direction = los_end - los_start

    los_transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
        beamlets=16,
        spot_width=0.03,
    )
    los_transform.set_equilibrium(plasma.equilibrium)
    model = Bolometer(
        diagnostic_name,
    )
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    bckc = model(sum_beamlets=False)

    if plot:
        model.plot(tplot=tplot)

    return plasma, model, bckc


if __name__ == "__main__":
    plt.ioff()
    example_run(plot=True)
    plt.show()
