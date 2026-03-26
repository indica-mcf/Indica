"""Example script for running :class:`NbiFidasim` with default ST40 objects.

This example is intended as a smoke test for the new abstract-NBI interface:
1) build default transform/plasma/equilibrium objects,
2) run one FIDASIM call through NbiFidasim,
3) read refactor_output dict and plot the four returned rhop profiles.
"""

from __future__ import annotations

from indica.configs.operators.nbi_configs import get_default_nbi_transform_config
from indica.converters.line_of_sight import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects
from indica.operators import NbiFidasim


def run_nbi_operator_example(
    show_plots: bool = True,
    reuse_existing_outputs: bool = False,
    overwrite: bool = True,
    save_plots: bool = True,
    plot_dir: str | None = None,
):
    """Build default inputs, run/reuse NbiFidasim outputs, and use operator plotting."""
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
    # For fast reruns, set reuse_existing_outputs=True and overwrite=False.
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
        prepare_kwargs={
            "overwrite": overwrite,
            "reuse_existing_outputs": reuse_existing_outputs,
        },
        run_kwargs={"reuse_existing_outputs": reuse_existing_outputs},
    )

    print("NBI result keys:", list(result.keys()))
    for key, value in result.items():
        print(f"{key}: dims={value.dims}, shape={value.shape}")

    plot_out = nbi_op.plot(
        result=result,
        show=show_plots,
        save_plots=save_plots,
        plot_dir=plot_dir,
    )
    for key, path in plot_out["saved_paths"].items():
        print(f"Saved {key} plot: {path}")

    return result


if __name__ == "__main__":
    run_nbi_operator_example(
        show_plots=False,
        reuse_existing_outputs=True,
        overwrite=False,
        save_plots=True,
        plot_dir="/home/jussi.hakosalo/Indica/indica/examples/fidasimtestplots"
    )
