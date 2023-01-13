import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


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
                self.bckc[quantity] = self.los_integral_radiation
                error = xr.full_like(self.bckc[quantity], 0.0)
                stdev = xr.full_like(self.bckc[quantity], 0.0)
                self.bckc[quantity].attrs = {
                    "datatype": datatype,
                    "transform": self.los_transform,
                    "error": error,
                    "stdev": stdev,
                    "provenance": str(self),
                }
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

    def __call__(
        self,
        Ne: DataArray = None,
        Nimp: DataArray = None,
        Lz: dict = None,
        t: LabeledArray = None,
        calc_rho=False,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile (dims = "rho", "t")
        Nimp
            Impurity density profiles (dims = "rho", "t", "element")
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
            Nimp = self.plasma.impurity_density.interp(t=t)
        else:
            if Ne is None or Nimp is None or Lz is None:
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Ne = Ne
        self.Nimp = Nimp
        self.Lz = Lz

        elements = Nimp.element.values

        _emission = []
        for ielem, elem in enumerate(elements):
            _emission.append(
                self.Lz[elem].sum("ion_charges") * self.Nimp.sel(element=elem) * self.Ne
            )
        emission = xr.concat(_emission, "element")
        los_integral = self.los_transform.integrate_on_los(
            emission.sum("element"),
            t=t,
            calc_rho=calc_rho,
        )

        self.emission = emission
        self.los_integral_radiation = los_integral

        self._build_bckc_dictionary()

        return self.bckc


def example_run(
    pulse:int=None,
    diagnostic_name: str = "bolo_Rz",
    origin: LabeledArray = None,
    direction: LabeledArray = None,
    plasma=None,
    plot=False,
):

    if plasma is None:
        plasma = example_plasma(pulse=pulse)

    # Create new interferometers diagnostics
    if origin is None or direction is None:
        nchannels = 11
        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.17
        los_end[:, 1] = 0.0
        los_end[:, 2] = np.linspace(0.53, -0.53, nchannels)
        los_start = np.array([[1.0, 0, 0]] * los_end.shape[0])
        origin = los_start
        direction = los_end - los_start

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
    )
    los_transform.set_equilibrium(plasma.equilibrium)
    model = Bolometer(
        diagnostic_name,
    )
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    bckc = model()

    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]

        model.los_transform.plot_los(tplot, plot_all=True)

        # Plot back-calculated profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            bckc["brightness"].sel(t=t, method="nearest").plot(
                label=f"t={t:1.2f} s", color=cols_time[i]
            )
        plt.xlabel("Channel")
        plt.ylabel("Measured brightness (W/m^2)")
        plt.legend()

        # Plot the radiation profiles
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.emission.rho_poloidal,
                model.emission.sum("element").sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("Local radiated power (W/m^3)")
        plt.legend()

    return plasma, model, bckc
