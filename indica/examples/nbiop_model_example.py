"""Example script for running :class:`NbiFidasim` via :class:`NbiOperatorModel`.

This shows the diagnostic-model style workflow:
1) build default transform/plasma/equilibrium objects,
2) attach Plasma + transform to NbiOperatorModel,
3) call model without passing profile input args explicitly.
"""

from __future__ import annotations

import os

from indica.configs.operators.nbi_configs import get_default_nbi_transform_config
from indica.converters.line_of_sight import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects
from indica.models import NbiOperatorModel
from indica.operators import NbiFidasim


def run_nbi_model_example(
    fi_dist_file: str,
    num_cores: int | None = None,
    show_plots: bool = True,
    reuse_existing_outputs: bool = False,
    overwrite: bool = True,
    save_plots: bool = True,
    plot_dir: str | None = None,
):
    """Build defaults, run/reuse NbiOperatorModel output, and plot via operator."""
    machine = "st40"

    nbi_cfg = get_default_nbi_transform_config()
    nbi_transform = LineOfSightTransform(**nbi_cfg)

    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")
    plasma.set_equilibrium(equilibrium)
    nbi_transform.set_equilibrium(equilibrium)

    nbi_op = NbiFidasim(
        name="hnbi",
        energy=52.0e3,  # eV
        power=0.5e6,  # W
        nbi_element="d",
        current_fractions=(0.5, 0.35, 0.15),
    )

    nbi_model = NbiOperatorModel(
        operator=nbi_op,
        target_element="d",
        file_name="nbiop_model_example",
        machine=machine,
    )
    nbi_model.set_transform(nbi_transform)
    nbi_model.set_plasma(plasma)

    result = nbi_model(
        t=float(plasma.t[5]),
        prepare_kwargs={
            "fi_dist_file": fi_dist_file,
            "overwrite": overwrite,
            "reuse_existing_outputs": reuse_existing_outputs,
        },
        run_kwargs={
            "num_cores": num_cores,
            "reuse_existing_outputs": reuse_existing_outputs,
        },
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
    fi_dist_file = os.environ.get("FIDASIM_FI_DIST_FILE")
    if not fi_dist_file:
        raise EnvironmentError(
            "Set FIDASIM_FI_DIST_FILE to run this example "
            "(explicit fi_dist_file is required)."
        )
    run_nbi_model_example(
        fi_dist_file=fi_dist_file,
        num_cores=None,
        show_plots=False,
        reuse_existing_outputs=False,
        overwrite=True,
        save_plots=True,
        plot_dir="/home/jussi.hakosalo/Indica/indica/examples/fidasimtestplots",
    )
