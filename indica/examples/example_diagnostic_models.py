from typing import Callable

import matplotlib.pylab as plt

from indica.defaults.load_defaults import load_default_objects
from indica.models import (
    BolometerCamera,
    BremsstrahlungDiode,
    ChargeExchangeSpectrometer,
    EquilibriumReconstruction,
    HelikeSpectrometer,
    Interferometer,
    ThomsonScattering,
)


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
    _model = BolometerCamera
    return run_example_diagnostic_model(machine, instrument, _model, plot=plot)


def example_axuv_unfiltered(
    plot=False,
):
    machine = "st40"
    instrument = "sxrc_xy1"
    _model = BolometerCamera
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
    instrument = "brems"
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
