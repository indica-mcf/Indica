from pathlib import Path
import pickle

import numpy as np

from indica.equilibrium import Equilibrium
from indica.models import Plasma
from indica.models.plasma import PlasmaProfiles
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

    kwargs = dict(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion="h",
        impurities=("c", "ar", "he"),
        machine_dimensions=machine_dimensions,
        impurity_concentration=(0.02, 0.001),  # should be deleted!
        full_run=False,
        n_rad=41,
        n_R=100,
        n_z=100,
    )
    # Plasma object
    plasma = Plasma(**kwargs)
    fract_abu, power_loss_tot, power_loss_sxr = default_atomic_data(plasma.elements)
    plasma.fract_abu = fract_abu
    plasma.power_loss_tot = power_loss_tot
    plasma.power_loss_sxr = power_loss_sxr

    # Assign profiles to time-points
    update_profiles = PlasmaProfiles(plasma)
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 2, nt)
    te_peaking = np.linspace(1, 2, nt)
    _y0 = update_profiles.profilers["toroidal_rotation"].y0
    vrot0 = np.linspace(
        _y0 * 1.1,
        _y0 * 2.5,
        nt,
    )
    vrot_peaking = np.linspace(1, 2, nt)

    _y0 = update_profiles.profilers["ion_temperature"].y0
    ti0 = np.linspace(_y0 * 1.1, _y0 * 2.5, nt)

    _y0 = update_profiles.profilers[f"impurity_density:{plasma.impurities[0]}"].y0
    nimp_y0 = _y0 * 5 * np.linspace(1, 8, nt)
    nimp_peaking = np.linspace(1, 5, nt)
    nimp_wcenter = np.linspace(0.4, 0.1, nt)
    for i, t in enumerate(plasma.t):
        parameters = {
            "electron_temperature.peaking": te_peaking[i],
            "ion_temperature.peaking": te_peaking[i],
            "ion_temperature.y0": ti0[i],
            "toroidal_rotation.peaking": vrot_peaking[i],
            "toroidal_rotation.y0": vrot0[i],
            "electron_density.peaking": ne_peaking[i],
            f"impurity_density:{plasma.impurities[1]}.peaking": nimp_peaking[i],
            f"impurity_density:{plasma.impurities[1]}.y0": nimp_y0[i],
            f"impurity_density:{plasma.impurities[1]}.wcenter": nimp_wcenter[i],
        }
        update_profiles(parameters, t=t)
    print(f"\n Writing plasma object to: {plasma_file}. \n")
    pickle.dump(plasma, open(plasma_file, "wb"))

    return plasma


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


if __name__ == "__main__":
    save_default_objects("st40", 11419)
