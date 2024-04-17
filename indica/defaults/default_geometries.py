from pathlib import Path
import pickle

from indica.readers import ST40Conf
from indica.readers import ST40Reader
from indica.equilibrium import Equilibrium

PROJECT_PATH = Path(__file__).parent.parent
GEOMETRIES_PATH = f"{PROJECT_PATH}/defaults/"


def write_default_geometries(
    machine: str,
    pulse: int,
    tstart: float = 0.01,
    tend: float = 0.1,
    dl: float = 0.005,
    equilibrium_instrument:str = "efit",
):
    """
    Write geometries for specified machine to file for future use as defaults
    """
    if machine == "st40":
        _reader = ST40Reader(pulse, tstart, tend)
        _conf = ST40Conf()
    else:
        raise ValueError(f"Machine {machine} currently not supported")

    transforms: dict = {}
    for instr in _conf.INSTRUMENT_METHODS.keys():
        try:
            data = _reader.get("", instr, 0, dl=dl)
        except Exception as e:
            print(f"Error reading {instr}: {e}")

        if hasattr(data[list(data)[0]], "transform"):
            _transform = data[list(data)[0]].transform
            transforms[instr] = _transform

    filename = geometry_filename(machine)
    print(f"\n Writing geometry to: {filename}. \n")
    pickle.dump(transforms, open(filename, "wb"))

    equilibrium_data = _reader.get("", equilibrium_instrument, 0)
    equilibrium_object = Equilibrium(equilibrium_data)
    filename = equilibrium_filename(machine)
    print(f"\n Writing equilibrium data to: {filename}. \n")
    pickle.dump(equilibrium_object, open(filename, "wb"))


def load_default_transforms(machine: str):
    return pickle.load(open(geometry_filename(machine), "rb"))

def load_default_equilibrium(machine: str):
    return pickle.load(open(equilibrium_filename(machine), "rb"))

def geometry_filename(machine: str):
    return GEOMETRIES_PATH + f"{machine}_default_geometry_transforms.pkl"

def equilibrium_filename(machine: str):
    return GEOMETRIES_PATH + f"{machine}_default_equilibrium_data.pkl"
