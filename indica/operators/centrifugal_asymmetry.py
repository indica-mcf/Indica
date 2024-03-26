from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
from xarray import DataArray

from indica.equilibrium import Equilibrium
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.utilities import assign_data
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
    for elem in elements.values:
        main_ion_mass = get_element_info(main_ion)[1]
        mass = get_element_info(elem)[1]
        asymmetry_parameter.loc[dict(element=elem)] = ph.centrifugal_asymmetry(
            ion_temperature.sel(element=elem).drop_vars("element"),
            electron_temperature,
            mass,
            meanz.sel(element=elem).drop_vars("element"),
            zeff,
            main_ion_mass,
            toroidal_rotation=toroidal_rotation.sel(element=elem).drop_vars("element"),
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

    rho_2d = equilibrium.rho.interp(t=t)
    R_0 = (
        equilibrium.rmjo.drop_vars("z")
        .interp(t=t)
        .interp(rho_poloidal=rho_2d)
        .drop_vars("rho_poloidal")
    )

    _profile_2d = _profile_to_map.interp(rho_poloidal=rho_2d).drop_vars("rho_poloidal")
    profile_2d = _profile_2d * np.exp(
        _asymmetry_parameter.interp(rho_poloidal=rho_2d) * (rho_2d.R**2 - R_0**2)
    )

    return profile_2d


def example_run(plot: bool = False):

    plasma = example_plasma()

    asymmetry_parameter = centrifugal_asymmetry_parameter(
        plasma.ion_density,
        plasma.ion_temperature,
        plasma.electron_temperature,
        plasma.toroidal_rotation,
        plasma.meanz,
        plasma.zeff,
        plasma.main_ion,
    )

    ion_density_2d = centrifugal_asymmetry_2d_map(
        plasma.ion_density,
        asymmetry_parameter,
        plasma.equilibrium,
    )

    ion_density_2d = assign_data(
        ion_density_2d, ("density", "ion"), "$m^{-3}$", long_name="Ion density"
    )

    if plot:
        tplot = ion_density_2d.t[2]
        element = ion_density_2d.element[2]
        plot_2d = ion_density_2d.sel(t=tplot, element=element)
        plt.figure()
        plot_2d.plot()

        plt.figure()
        plot_2d.sel(z=0, method="nearest").plot(label="Midplane z=0")
        plt.legend()

    return plasma, ion_density_2d
