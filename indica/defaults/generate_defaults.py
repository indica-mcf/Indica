from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
from xarray import DataArray

from indica.operators import FractionalAbundanceAdas
from indica.operators import PowerLoss
from indica.profilers import ProfilerGauss

PROJECT_PATH = Path(__file__).parent.parent
DEFAULTS_PATH = f"{PROJECT_PATH}/defaults/"


def default_atomic_data(
    elements: Tuple[str, ...],
):
    """
    Initialises atomic data classes with default ADAS files and runs the
    __call__ with default plasma parameters
    """
    n_rad = 41
    rho_end = 1.02
    rho = np.abs(np.linspace(rho_end, 0, n_rad) ** 1.8 - rho_end - 0.01)
    rho_coord = DataArray(rho, coords={"rhop": rho}, dims="rhop").coords
    Te = DataArray(np.linspace(20, 9e3, n_rad), coords=rho_coord)
    Ne = DataArray(np.logspace(18, 20, n_rad), coords=rho_coord)
    Nn = DataArray(np.logspace(15, 12, n_rad), coords=rho_coord)
    tau = None

    fract_abu, power_loss_tot = {}, {}
    for elem in elements:
        print(elem)
        Fz_adas = FractionalAbundanceAdas(elem)
        fz_t = Fz_adas(
            Te,
            Ne,
            Nn=Nn,
            tau=tau,
        )
        fract_abu[elem] = deepcopy(Fz_adas)

        Ploss = PowerLoss(element=elem)
        _ = Ploss(Te, fz_t, Ne=Ne, Nn=Nn)
        power_loss_tot[elem] = deepcopy(Ploss)

    return fract_abu, power_loss_tot


def default_profiles_gauss(n_rad: int = 41):
    """
    Genarate default plasma profiles
    """
    rhop_end = 1.02
    rhop = np.abs(np.linspace(rhop_end, 0, n_rad) ** 1.8 - rhop_end - 0.01)

    Te = ProfilerGauss(datatype="electron_temperature", xspl=rhop)
    Ne = ProfilerGauss(datatype="electron_density", xspl=rhop)
    Nn = ProfilerGauss(datatype="neutral_density", xspl=rhop)
    tau = None

    return Te, Ne, Nn, tau
