from typing import Callable

import matplotlib.pylab as plt
import numpy as np

from indica.defaults.load_defaults import load_default_objects
from indica.models import BremsstrahlungDiode
from indica.models import ChargeExchangeSpectrometer
from indica.models import EquilibriumReconstruction
from indica.models import HelikeSpectrometer
from indica.models import Interferometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.models.passive_spectrometer import format_pecs
from indica.models.passive_spectrometer import PassiveSpectrometer
from indica.models.passive_spectrometer import read_adf15s
from indica.operators.atomic_data import default_atomic_data


def run_example_diagnostic_model(
    machine: str, instrument: str, model: Callable, plot: bool = False, **kwargs
):
    transforms = load_default_objects(machine, "geometry")
    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")

    plasma.set_equilibrium(equilibrium)
    transform = transforms[instrument]
    transform.set_equilibrium(equilibrium)

    model = model(instrument, **kwargs)
    model.set_transform(transform)
    model.set_plasma(plasma)

    bckc = model(
        sum_beamlets=False,
    )

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
    _, power_loss = default_atomic_data(["h", "ar", "c", "he"])

    return run_example_diagnostic_model(
        machine, instrument, _model, plot=plot, power_loss=power_loss
    )


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


def example_passive_spectroscopy(
    plot=False,
):
    machine = "st40"
    instrument = "sxrc_xy1"  # placeholder
    _model = PassiveSpectrometer
    wlower, wupper = (400, 550)

    pecs = read_adf15s(
        [
            "he",
            "c",
            "ar",
        ],
    )
    pec_database = format_pecs(pecs, wavelength_bounds=slice(wlower, wupper))

    window = np.linspace(wlower, wupper, 1000)
    return run_example_diagnostic_model(
        machine,
        instrument,
        _model,
        plot=plot,
        pecs=pec_database,
        window=window,
    )


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
    example_bolometer(plot=True)
