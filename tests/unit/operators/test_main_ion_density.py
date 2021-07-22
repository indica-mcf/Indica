import numpy as np
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters import FluxSurfaceCoordinates
from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.datatypes import ELEMENTS_BY_SYMBOL
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.main_ion_density import MainIonDensity
from indica.operators.mean_charge import MeanCharge
from indica.operators.spline_fit import Spline
from indica.readers.adas import ADASReader
from indica.utilities import broadcast_spline

# import unittest


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    if not isinstance(t, DataArray):
        if isinstance(t, np.ndarray):
            t = DataArray(data=t, coords={"t": t}, dims=["t"])
        else:
            t = DataArray(data=np.array([t]), coords={"t": np.array([t])}, dims=["t"])

    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    input_Ne = DataArray(
        data=np.tile(np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )
    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    rho = DataArray(
        data=np.linspace(0.0, 1.0, 20),
        coords=[("rho", np.linspace(0.0, 1.05, 20))],
        dims=["rho"],
    )

    dummy_coordinates = FluxSurfaceCoordinates("poloidal")

    input_Ne_spline = Spline(input_Ne, "rho", dummy_coordinates)
    input_Ne = broadcast_spline(
        input_Ne_spline.spline,
        input_Ne_spline.spline_dims,
        input_Ne_spline.spline_coords,
        rho,
    )

    input_Te_spline = Spline(input_Te, "rho", dummy_coordinates)
    input_Te = broadcast_spline(
        input_Te_spline.spline,
        input_Te_spline.spline_dims,
        input_Te_spline.spline_coords,
        rho,
    )

    example_frac_abundance = FractionalAbundance(
        SCD,
        ACD,
        input_Ne.isel(t=0),
        input_Te.isel(t=0),
    )

    F_z_tinf = example_frac_abundance.F_z_tinf

    # ignore with mypy since this is testing and inputs are known
    F_z_tinf = F_z_tinf.expand_dims({"t": t.size}, axis=-1)  # type: ignore

    return F_z_tinf


def test_main_ion_density():
    example_main_ion_density = MainIonDensity()

    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    t = np.linspace(77.5, 82.5, 6)
    rho = DataArray(
        data=np.linspace(0.0, 1.0, 20),
        coords=[("rho", np.linspace(0.0, 1.05, 20))],
        dims=["rho"],
    )

    electron_density = DataArray(
        data=np.tile(np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    dummy_coordinates = FluxSurfaceCoordinates("poloidal")

    electron_density_spline = Spline(electron_density, "rho", dummy_coordinates)
    electron_density = broadcast_spline(
        electron_density_spline.spline,
        electron_density_spline.spline_dims,
        electron_density_spline.spline_coords,
        rho,
    )

    beryllium_impurity_conc = 0.03 * electron_density
    neon_impurity_conc = 0.02 * electron_density
    nickel_impurity_conc = 0.0002 * electron_density
    tungsten_impurity_conc = 0.00005 * electron_density

    # be, ne, ni, w
    elements = [4, 10, 28, 74]
    elements = [ELEMENTS_BY_ATOMIC_NUMBER.get(i) for i in elements]

    impurity_densities = DataArray(
        data=np.ones((len(elements), *rho.shape, *t.shape)),
        coords=[("elements", elements), ("rho", rho), ("t", t)],
        dims=["elements", "rho", "t"],
    )

    impurity_densities.data[0] = beryllium_impurity_conc
    impurity_densities.data[1] = neon_impurity_conc
    impurity_densities.data[2] = nickel_impurity_conc
    impurity_densities.data[3] = tungsten_impurity_conc

    mean_charge = zeros_like(impurity_densities)

    F_z_tinf = fractional_abundance_setup("be", t)
    element_name = ELEMENTS_BY_SYMBOL.get("be")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[0] = result

    F_z_tinf = fractional_abundance_setup("ne", t)
    element_name = ELEMENTS_BY_SYMBOL.get("ne")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[1] = result

    F_z_tinf = fractional_abundance_setup("ni", t)
    element_name = ELEMENTS_BY_SYMBOL.get("ni")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[2] = result

    F_z_tinf = fractional_abundance_setup("w", t)
    element_name = ELEMENTS_BY_SYMBOL.get("w")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[3] = result

    nominal_inputs = {
        "impurity_densities": impurity_densities,
        "electron_density": electron_density,
        "mean_charge": mean_charge,
    }

    try:
        main_ion_density = example_main_ion_density(**nominal_inputs)
    except Exception as e:
        raise e

    assert main_ion_density is not None
