"""Example script for running :class:`NbiFidasim` with default ST40 objects.

This example is intended as a smoke test for the new abstract-NBI interface:
1) build default transform/plasma/equilibrium objects,
2) run one FIDASIM call through NbiFidasim,
3) read refactor_output dict and plot the four returned rhop profiles.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from indica.configs.operators.nbi_configs import get_default_nbi_transform_config
from indica.converters.line_of_sight import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects
from indica.operators import NbiFidasim


def run_nbi_operator_example(show_plots: bool = True):
    """Build default inputs, run NbiFidasim once, and plot output profiles."""
    machine = "st40"

    # Build NBI transform from editable default config and attach equilibrium.
    nbi_cfg = get_default_nbi_transform_config()
    nbi_transform = LineOfSightTransform(**nbi_cfg)

    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")
    plasma.set_equilibrium(equilibrium)
    nbi_transform.set_equilibrium(equilibrium)

    # Instantiate the new abstract-NBI implementation for FIDASIM.
    nbi_op = NbiFidasim(
        name="hnbi",
        energy=52.0e3,  # eV
        power=0.5e6,  # W
        nbi_element="d",
        current_fractions=(0.5, 0.35, 0.15),
    )
    nbi_op.set_transform(nbi_transform)

    # Run one time point through prepare -> run -> refactor_output.
    result = nbi_op(
        Ti=plasma.ion_temperature,
        Te=plasma.electron_temperature,
        Ne=plasma.electron_density,
        Nn=plasma.neutral_density,
        Vtor=plasma.toroidal_rotation,
        Zeff=plasma.zeff,
        MeanZ=plasma.meanz,
        target_element="d",
        t=float(plasma.t[5]),
        file_name="nbiop_example",
        machine=machine,
    )

    print("NBI result keys:", list(result.keys()))
    for key, value in result.items():
        print(f"{key}: dims={value.dims}, shape={value.shape}")

    # Plot the four Indica-native outputs vs rhop at the single returned time.
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    keys = [
        "neutral_density",
        "fast_ion_density",
        "parallel_fast_ion_pressure",
        "perpendicular_fast_ion_pressure",
    ]

    for ax, key in zip(axs.flat, keys):
        da = result[key]
        ax.plot(da.rhop.values, da.isel(t=0).values)
        ax.set_title(key)
        ax.set_xlabel("rhop")
        ax.set_ylabel(key)

    fig.tight_layout()
    if show_plots:
        plt.show()

    return result


if __name__ == "__main__":
    run_nbi_operator_example(show_plots=True)
