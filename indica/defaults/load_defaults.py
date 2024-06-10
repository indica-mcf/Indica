from pathlib import Path
import pickle


PROJECT_PATH = Path(__file__).parent.parent
DEFAULTS_PATH = f"{PROJECT_PATH}/defaults/"

def get_filename_default_objects(machine: str):
    _files = {}
    _files["geometry"] = (
        DEFAULTS_PATH + f"{machine}_default_geometry_transform_objects.pkl"
    )
    _files["equilibrium"] = DEFAULTS_PATH + f"{machine}_default_equilibrium_object.pkl"
    _files["plasma"] = DEFAULTS_PATH + f"{machine}_default_plasma_object.pkl"

    return _files


def load_default_objects(machine: str, identifier: str = "geometry"):
    """
    Load default objects from local pickle files

    Parameters
    ----------
    machine - e.g. "st40"
    identifier - "geometry" or "equilibrium" or "plasma"
    """
    _file = get_filename_default_objects(machine)[identifier]

    try:
        return pickle.load(open(_file, "rb"))
    except FileNotFoundError:
        to_print = f"""

************************************************************
The following file does not exist:
{_file}

Create your defaults file:
    python indica/defaults/read_write_defaults.py

or, to choose specific kwargs (e.g. machine or pulse):
    save_default_objects("st40", 11419)

************************************************************
            """
        raise Exception(to_print)
