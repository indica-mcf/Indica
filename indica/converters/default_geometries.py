from pathlib import Path
import pickle
from typing import List
from indica.readers import ST40Reader, ST40Conf

PROJECT_PATH = Path(__file__).parent.parent
GEOMETRIES_PATH = f"{PROJECT_PATH}/data/"

def load_default_geometries(machine: str):
    """
    Load default geometries for specified machine

    Parameters
    ----------
    machine
        Machine name
    pulse
        Pulse number

    Returns
    -------
    dictionary of instrument coordinate transforms
    """

    geometries_file = geometry_filename(machine)

    return pickle.load(open(geometries_file, "rb"))


def write_default_geometries(
    machine:str,
    pulse: int,
    tstart: float = 0.01,
    tend: float = 0.1,
    dl: float = 0.02,
):
    """
    Write geometries for specified machine to file for future use as defaults

    Parameters
    ----------
    machine
        Machine name
    pulse
        Pulse number

    Returns
    -------
    dictionary of instrument coordinate transforms
    """
    if machine=="st40":
        _reader = ST40Reader(pulse, tstart, tend)
        _conf = ST40Conf()
    else:
        raise ValueError(f"Machine {machine} currently not supported")

    transforms: dict = {}
    for instr in _conf.INSTRUMENT_METHODS.keys():
        try:
            data = _reader.get("", instr, 0, dl=dl)
            transforms[instr] = data[list(data)[0]].transform
        except Exception as e:
            print(f"Error reading {instr}: {e}")
            raise e

    filename = geometry_filename(machine)
    print(f"\n Writing geometry file to: {filename}. \n")
    pickle.dump(transforms, open(filename, "wb"))

def geometry_filename(machine: str):
    return GEOMETRIES_PATH + f"{machine}_default_geometries.pkl"
