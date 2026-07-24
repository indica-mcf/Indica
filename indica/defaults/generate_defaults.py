from pathlib import Path
import pickle
from typing import Tuple

import numpy as np
from xarray import DataArray

from indica import Equilibrium
from indica.configs.readers import ST40Conf
from indica.defaults.load_defaults import get_filename_default_objects
from indica.examples.example_plasma import example_plasma
from indica.operators import FractionalAbundanceAdas
from indica.operators import PowerLoss
from indica.profilers import ProfilerGauss
from indica.readers import ST40Reader

PROJECT_PATH = Path(__file__).parent.parent
DEFAULTS_PATH = f"{PROJECT_PATH}/defaults/"


def default_atomic_data(
    elements: Tuple[str, ...],
    Te: DataArray = None,
    Ne: DataArray = None,
    Nn: DataArray = None,
    tau: DataArray = None,
):
    """
    Initialises atomic data classes with default ADAS files and runs the
    __call__ with default plasma parameters
    """
    if Te is None or Ne is None:
        Te, Ne, Nn, tau = default_profiles()

    fract_abu, power_loss_tot = {}, {}
    for elem in elements:
        fract_abu[elem] = FractionalAbundanceAdas(element=elem)
        fz_t = fract_abu[elem](Ne, Te, Nn, tau=tau)

        power_loss_tot[elem] = PowerLoss(element=elem)
        _ = power_loss_tot[elem](Te, fz_t, Ne=Ne, Nn=Nn)

    return fract_abu, power_loss_tot


def default_profiles(n_rad: int = 20):
    """
    Set default plasma profiles to calculate atomic data
    """
    xend = 1.02
    rho_end = 1.01
    rho = np.abs(np.linspace(rho_end, 0, n_rad) ** 1.8 - rho_end - 0.01)
    rho_coord = DataArray(rho, coords={"rhop": rho}, dims="rhop").coords
    Te = DataArray(np.linspace(10, 10e3, n_rad), coords=rho_coord)
    Ne = DataArray(np.logspace(16, 21, n_rad), coords=rho_coord)

    # TODO: fix FractionalAbundance so that it does 2d interp of Nn and Te
    params = {
        "y0": 1e14,
        "y1": 5e15,
        "yend": 5e15,
        "wcenter": 0.01,
        "wped": 18,
        "peaking": 1,
    }
    Nn_prof = ProfilerGauss(
        datatype="neutral_density", xspl=rho, xend=xend, parameters=params
    )
    Nn = Nn_prof()
    tau = None
    return Te, Ne, Nn, tau


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
        _reader = ST40Reader(pulse, tstart - dt, tend + dt, dt=dt)
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

    # Plasma object
    plasma = example_plasma(
        machine=machine,
        tstart=tstart,
        tend=tend,
        dt=dt,
    )
    fract_abu, power_loss_tot = default_atomic_data(plasma.elements)
    plasma.fract_abu = fract_abu
    plasma.power_loss_tot = power_loss_tot
    print(f"\n Writing plasma object to: {plasma_file}. \n")
    pickle.dump(plasma, open(plasma_file, "wb"))

    return plasma


if __name__ == "__main__":
    # save_default_objects("st40", 11560)
    save_default_objects("st40", 12857)
