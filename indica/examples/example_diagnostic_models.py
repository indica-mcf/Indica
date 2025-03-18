from typing import Callable

import matplotlib.pylab as plt

from indica.defaults.load_defaults import load_default_objects
from indica.models import BremsstrahlungDiode
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.operators.atomic_data import default_atomic_data


def run_example_diagnostic_model(
    machine: str, instrument: str, model: Callable, plot: bool = False
):
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)

    kwargs = {}
    if model == PinholeCamera:
        _, power_loss = default_atomic_data(plasma.elements)
        kwargs["power_loss"] = power_loss

    model = model(instrument, **kwargs)
    model.set_transform(transform)
    model.set_plasma(plasma)

    bckc = model(sum_beamlets=False)

    if plot and hasattr(model, "plot"):
        plt.ioff()
        model.plot()
        plt.show()

    return plasma, model, bckc


def example_thomson_scattering(
    plot=False,
):
    machine = "st40"
    instrument = "ts"
    _model = ThomsonScattering
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_bolometer(
    plot=False,
):
    machine = "st40"
    instrument = "blom_xy1"
    _model = PinholeCamera
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_axuv_unfiltered(
    plot=False,
):
    machine = "st40"
    instrument = "sxrc_xy1"
    _model = PinholeCamera
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_charge_exchange(
    plot=False,
):
    machine = "st40"
    instrument = "cxff_pi"
    _model = ChargeExchangeSpectrometer
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_helike_spectroscopy(
    plot=False,
):
    machine = "st40"
    instrument = "xrcs"
    _model = HelikeSpectrometer
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_interferometer(
    plot=False,
):
    machine = "st40"
    instrument = "smmh"
    _model = Interferometer
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_diode_filter(
    plot=False,
):
    machine = "st40"
    instrument = "sxrc_xy1"
    _model = BremsstrahlungDiode
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_equilibrium(
    plot=False,
):
    machine = "st40"
    instrument = "efit"
    _model = EquilibriumReconstruction
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


if __name__ == "__main__":
    example_helike_spectroscopy(plot=True)
