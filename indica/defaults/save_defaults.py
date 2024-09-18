from pathlib import Path
import pickle

from indica.defaults.load_defaults import get_filename_default_objects
from indica.equilibrium import Equilibrium
from indica.examples.example_plasma import example_plasma
from indica.operators.atomic_data import default_atomic_data
from indica.readers import ST40Conf
from indica.readers import ST40Reader

PROJECT_PATH = Path(__file__).parent.parent
DEFAULTS_PATH = f"{PROJECT_PATH}/defaults/"


def save_default_objects(
    machine: str,
    pulse: int,
    tstart: float = 0.02,
    tend: float = 0.1,
    dt: float = 0.01,
    dl: float = 0.005,
    equilibrium_instrument: str = "efit",
):
    """
    Write geometries for specified machine to file for future use as defaults
    """
    if machine == "st40":
        _reader = ST40Reader(pulse, tstart - dt, tend + dt)
        _conf = ST40Conf()
    else:
        raise ValueError(f"Machine {machine} currently not supported")

    plasma_file = get_filename_default_objects(machine)["plasma"]
    equilibrium_file = get_filename_default_objects(machine)["equilibrium"]
    geometry_file = get_filename_default_objects(machine)["geometry"]
    # Diagnostic geometry transform objects
    transforms: dict = {}
    for instr in _conf.INSTRUMENT_METHODS.keys():
        try:
            data = _reader.get("", instr, 0, dl=dl)
            if hasattr(data[list(data)[0]], "transform"):
                _transform = data[list(data)[0]].transform
                transforms[instr] = _transform
        except Exception as e:
            print(f"Error reading {instr}: {e}")
    print(f"\n Writing geometry to: {geometry_file}. \n")
    pickle.dump(transforms, open(geometry_file, "wb"))

    # Equilibrium object
    equilibrium_data = _reader.get("", equilibrium_instrument, 0)
    equilibrium_object = Equilibrium(equilibrium_data)
    print(f"\n Writing equilibrium data to: {equilibrium_file}. \n")
    pickle.dump(equilibrium_object, open(equilibrium_file, "wb"))

    # Plasma Equilibrium object
    if machine == "st40":
        conf = ST40Conf()
        machine_dimensions = conf.MACHINE_DIMS
    else:
        raise Exception("\n st40 is currently only the only supported machine \n")

    # Plasma object
    plasma = example_plasma(
        machine=machine,
        pulse=pulse,
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion="h",
        impurities=("c", "ar", "he"),
        load_from_pkl=False,
        machine_dimensions=machine_dimensions,
        impurity_concentration=(0.02, 0.001),  # should be deleted!
        full_run=False,
        n_rad=41,
        n_R=100,
        n_z=100,
    )
    fract_abu, power_loss_tot, power_loss_sxr = default_atomic_data(plasma.elements)
    plasma.fract_abu = fract_abu
    plasma.power_loss_tot = power_loss_tot
    plasma.power_loss_sxr = power_loss_sxr
    print(f"\n Writing plasma object to: {plasma_file}. \n")
    pickle.dump(plasma, open(plasma_file, "wb"))

    return plasma


if __name__ == "__main__":
    save_default_objects("st40", 11560)
