"""Example script for running :class:`NBIOperator` with default ST40 objects."""

from indica.configs.operators.nbi_configs import get_default_nbi_transform_config
from indica.converters.line_of_sight import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects
from indica.operators import nbioperator


def run_nbi_operator_example():
    """Build default inputs and run the NBI operator once."""
    machine = "st40"

    # Build an NBI transform from editable placeholder config.
    nbi_cfg = get_default_nbi_transform_config()
    nbi_transform = LineOfSightTransform(**nbi_cfg)

    equilibrium = load_default_objects(machine, "equilibrium")
    plasma = load_default_objects(machine, "plasma")
    plasma.set_equilibrium(equilibrium)
    nbi_transform.set_equilibrium(equilibrium)



    nbi_op = nbioperator.NBIOperator(
        name="hnbi",
        einj=52.0,  # keV
        pinj=0.5,  # MW
        current_fractions=[0.5, 0.35, 0.15],
        ab=2.014,
        nbi_model="FIDASIM",
    )
    nbi_op.set_transform(nbi_transform)

    neutrals_by_time = nbi_op(
        ion_temperature=plasma.ion_temperature,
        electron_temperature=plasma.electron_temperature,
        electron_density=plasma.electron_density,
        neutral_density=plasma.neutral_density,
        toroidal_rotation=plasma.toroidal_rotation,
        zeff=plasma.zeff,
        t=plasma.t[5],
        file_name="nbiop_example",
    )
    print(neutrals_by_time)
    return neutrals_by_time


if __name__ == "__main__":
    run_nbi_operator_example()
