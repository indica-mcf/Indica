from typing import Callable

import matplotlib.pylab as plt

from indica.defaults.read_write_defaults import load_default_objects
from indica.models import Bolometer
from indica.models import ChargeExchange
from indica.models import HelikeSpectrometer
from indica.models import Interferometry
from indica.models import ThomsonScattering


def run_example_diagnostic_model(
    machine: str, instrument: str, model: Callable, plot: bool = False
):
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)
    model = model(instrument)
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
    _model = Bolometer
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_axuv_unfiltered(
    plot=False,
):
    machine = "st40"
    instrument = "sxrc_xy1"
    _model = Bolometer
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_charge_exchange(
    plot=False,
):
    machine = "st40"
    instrument = "cxff_pi"
    _model = ChargeExchange
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
    _model = Interferometry
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


if __name__ == "__main__":
    example_helike_spectroscopy(plot=True)
