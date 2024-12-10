from copy import deepcopy

import numpy as np
from xarray import DataArray

from indica import Equilibrium
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.utilities import get_element_info


def centrifugal_asymmetry_parameter(
    ion_density: DataArray,
    ion_temperature: DataArray,
    electron_temperature: DataArray,
    toroidal_rotation: DataArray,
    meanz: DataArray,
    zeff: DataArray,
    main_ion: str,
):
    """
    Indica-native wrapper to calculate the centrifugal asymmetry parameter a-la Wesson
    """

    elements = ion_density.element
    zeff = zeff.sum("element")

    asymmetry_parameter = deepcopy(ion_density)
    for elem in elements.data:
        _elem = str(elem)
        main_ion_mass = get_element_info(main_ion)[1]
        mass = get_element_info(_elem)[1]
        asymmetry_parameter.loc[dict(element=_elem)] = ph.centrifugal_asymmetry(
            ion_temperature,
            electron_temperature,
            mass,
            meanz.sel(element=_elem).drop_vars("element"),
            zeff,
            main_ion_mass,
            toroidal_rotation=toroidal_rotation,
        )

    return asymmetry_parameter


def centrifugal_asymmetry_2d_map(
    profile_to_map: DataArray,
    asymmetry_parameter: DataArray,
    equilibrium: Equilibrium,
    t: LabeledArray = None,
):
    """Map centrifugal asymmetric profiles to 2D"""

    if t is None:
        t = profile_to_map.t.values

    if "t" in profile_to_map.dims:
        _profile_to_map = profile_to_map.interp(t=t)
    else:
        _profile_to_map = profile_to_map
    if "t" in asymmetry_parameter.dims:
        _asymmetry_parameter = asymmetry_parameter.interp(t=t)
    else:
        _asymmetry_parameter = asymmetry_parameter

    rho_2d = equilibrium.rhop.interp(t=t)
    R_0 = equilibrium.rmjo.interp(t=t).interp(rhop=rho_2d).drop_vars("rhop")

    _profile_2d = _profile_to_map.interp(rhop=rho_2d).drop_vars("rhop")
    profile_2d = _profile_2d * np.exp(
        _asymmetry_parameter.interp(rhop=rho_2d) * (rho_2d.R**2 - R_0**2)
    )

    return profile_2d
