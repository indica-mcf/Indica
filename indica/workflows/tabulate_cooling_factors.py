from typing import List

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import xarray as xr

from indica.operators.atomic_data import default_profiles
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers.adas import ADASReader
from indica.readers.adas import ADF11
from indica.utilities import DATA_PATH


def cooling_factor_corona(
    elements: List[str],
    write_to_file: bool = False,
):
    """
    Initialises atomic data classes with default ADAS files and runs the
    __call__ with default plasma parameters
    """
    Te, Ne, Nh, tau = default_profiles(n_rad=100)
    Nh = xr.zeros_like(Nh)
    Ne = xr.full_like(Ne, 5.0e19)

    _size = Te.size
    # _size = 50
    te_array = (-np.log(np.linspace(1, 0.8, _size))) ** 2
    te_array -= -np.min(te_array)
    te_array /= np.max(te_array)
    te_array = te_array * 14995 + 5
    Te.values = te_array

    fract_abu, power_loss_tot, cooling_factor = {}, {}, {}
    files = ""
    adas_reader = ADASReader()
    _to_write = {"Te": Te.data, "Ne": Ne.data, "Nh": Nh.data}
    plt.figure()
    for elem in elements:
        print(elem)
        _scd = adas_reader.get_adf11("scd", elem, ADF11[elem]["scd"])
        _acd = adas_reader.get_adf11("acd", elem, ADF11[elem]["acd"])
        _ccd = adas_reader.get_adf11("ccd", elem, ADF11[elem]["ccd"])

        fract_abu[elem] = FractionalAbundance(_scd, _acd, ccd=_ccd)
        _fz = fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau)

        _plt = adas_reader.get_adf11("plt", elem, ADF11[elem]["plt"])
        _prb = adas_reader.get_adf11("prb", elem, ADF11[elem]["prb"])
        _prc = adas_reader.get_adf11("prc", elem, ADF11[elem]["prc"])
        power_loss_tot[elem] = PowerLoss(_plt, _prb, prc=_prc)
        _power_loss = power_loss_tot[elem](Te, _fz, Ne=Ne, Nh=Nh)

        files += (
            f"{_scd.filename} {_acd.filename} {_ccd.filename} "
            f"{_plt.filename} {_prb.filename} {_prc.filename} "
        )

        _cooling_factor = _power_loss.sum("ion_charge")
        _cooling_factor = (
            _cooling_factor.assign_coords(electron_temperature=("rho_poloidal", Te))
            .swap_dims({"rho_poloidal": "electron_temperature"})
            .drop_vars("rho_poloidal")
        )

        cooling_factor[elem] = _cooling_factor
        _cooling_factor.plot(label=elem)

        _to_write[elem] = cooling_factor[elem].data

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    _to_write["atomic_data_files"] = files

    if write_to_file:
        if any(Nh > 0):
            file_name = f"{DATA_PATH}corona_cooling_factors_Nh.csv"
        else:
            file_name = f"{DATA_PATH}corona_cooling_factors.csv"
        df = pd.DataFrame(_to_write)
        df.to_csv(file_name)

    return cooling_factor, _to_write, fract_abu
