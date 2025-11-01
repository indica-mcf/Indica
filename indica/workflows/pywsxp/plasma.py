from copy import deepcopy
from functools import partial

import numpy as np
from xarray import DataArray

from indica.plasma import Plasma


def _ion_density_2d(
    ion_density: DataArray,
    asymmetry_parameter: DataArray,
    R_0: DataArray,
) -> DataArray:
    rho_2d = asymmetry_parameter.rhop
    assert rho_2d is not None
    return ion_density.interp(rhop=rho_2d).drop_vars("rhop") * np.exp(
        asymmetry_parameter * (rho_2d.R**2 - R_0**2)
    )


def make_plasma_2d(
    plasma: Plasma,
    asymmetry_parameter: DataArray,
    R_0: DataArray,
) -> Plasma:
    plasma = deepcopy(plasma)
    ion_density = plasma.ion_density
    plasma.Ion_density.operator = partial(
        _ion_density_2d,
        ion_density,
        asymmetry_parameter,
        R_0,
    )
    plasma.Ion_density.__call__.cache_clear()
    return plasma
