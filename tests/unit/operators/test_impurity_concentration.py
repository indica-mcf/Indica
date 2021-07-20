import unittest
from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters import FluxSurfaceCoordinates
from indica.converters.lines_of_sight import LinesOfSightTransform
from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.datatypes import ELEMENTS_BY_SYMBOL
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.impurity_concentration import ImpurityConcentration
from indica.operators.mean_charge import MeanCharge
from indica.readers.adas import ADASReader
from ..test_equilibrium_single import equilibrium_dat_and_te


class Exception_Impurity_Concentration_Test_Case(unittest.TestCase):
    def __init__(
        self,
        element,
        Zeff_diag,
        impurity_densities,
        electron_density,
        mean_charge,
        flux_surfaces,
        t,
    ):
        self.element = element
        self.Zeff_diag = Zeff_diag
        self.impurity_densities = impurity_densities
        self.electron_density = electron_density
        self.mean_charge = mean_charge
        self.flux_surfaces = flux_surfaces
        self.t = t

        self.nominal_inputs = [
            self.element,
            self.Zeff_diag,
            self.impurity_densities,
            self.electron_density,
            self.mean_charge,
            self.flux_surfaces,
            self.t,
        ]

    def call_type_check(
        self,
        element=None,
        Zeff_diag=None,
        impurity_densities=None,
        electron_density=None,
        mean_charge=None,
        flux_surfaces=None,
        t=None,
    ):
        inputs = [
            element,
            Zeff_diag,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        element,
        Zeff_diag,
        impurity_densities,
        electron_density,
        mean_charge,
        flux_surfaces,
        t = inputs

        with self.assertRaises(TypeError):
            example_ = ImpurityConcentration()
            example_(
                element,
                Zeff_diag,
                impurity_densities,
                electron_density,
                mean_charge,
                flux_surfaces,
                t,
            )

    def call_value_check(
        self,
        element=None,
        Zeff_diag=None,
        impurity_densities=None,
        electron_density=None,
        mean_charge=None,
        flux_surfaces=None,
        t=None,
    ):
        inputs = [
            element,
            Zeff_diag,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        element,
        Zeff_diag,
        impurity_densities,
        electron_density,
        mean_charge,
        flux_surfaces,
        t = inputs

        with self.assertRaises(ValueError):
            example_ = ImpurityConcentration()
            example_(
                element,
                Zeff_diag,
                impurity_densities,
                electron_density,
                mean_charge,
                flux_surfaces,
                t,
            )


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Te = np.logspace(4.6, 2.0, 10)

    input_Te = DataArray(
        data=input_Te, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Ne = DataArray(
        data=input_Ne, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )

    example_frac_abundance = FractionalAbundance(
        SCD,
        ACD,
        input_Ne,
        input_Te,
    )

    F_z_tinf = example_frac_abundance.F_z_tinf

    # ignore with mypy since this is testing and inputs are known
    F_z_tinf = F_z_tinf.expand_dims({"t": t.size}, axis=-1)  # type: ignore

    return F_z_tinf


def test_impurity_concentration():
    example_ = ImpurityConcentration()

    # R_arr = np.linspace(1.83, 3.9, 20)
    rho = np.linspace(0.0, 1.5, 10)
    t = np.linspace(77.5, 82.5, 5)
    # t = np.array([80.0])

    Zeff_diag = DataArray(
        data=np.ones(*t.shape) + 1.5,
        coords={"t": t},
        dims=["t"],
        attrs={
            "transform": LinesOfSightTransform(
                R_start=np.array([1.9]),
                z_start=np.array([0.2]),
                T_start=np.array([0.0]),
                R_end=np.array([3.8]),
                z_end=np.array([0.3]),
                T_end=np.array([0.0]),
                name="Zeff_diag",
            ),
            "Zeff_diag_coords": DataArray(
                data=np.array([0]), dims=["Zeff_diag_coords"]
            ),
        },
    )

    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0, 1.05])

    electron_density = DataArray(
        data=np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3, 0.0]),
        coords={"rho": rho_profile},
        dims=["rho"],
    )

    beryllium_impurity_conc = 0.03 * electron_density
    neon_impurity_conc = 0.02 * electron_density
    nickel_impurity_conc = 0.0002 * electron_density
    tungsten_impurity_conc = 0.00005 * electron_density

    # be, c, ne, w
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
    mean_charge = mean_charge.isel(R=0)
    mean_charge = mean_charge.drop_vars("R")

    F_z_tinf = fractional_abundance_setup("be", t)
    element_name = ELEMENTS_BY_SYMBOL.get("be")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[0] = result

    F_z_tinf = fractional_abundance_setup("c", t)
    element_name = ELEMENTS_BY_SYMBOL.get("c")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[1] = result

    F_z_tinf = fractional_abundance_setup("ne", t)
    element_name = ELEMENTS_BY_SYMBOL.get("ne")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[2] = result

    F_z_tinf = fractional_abundance_setup("w", t)
    element_name = ELEMENTS_BY_SYMBOL.get("w")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[3] = result

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    nominal_inputs = {
        "element": "beryllium",
        "Zeff_diag": Zeff_diag,
        "impurity_densities": impurity_densities,
        "electron_density": electron_density,
        "mean_charge": mean_charge,
        "flux_surfaces": flux_surfs,
        "t": DataArray(data=[80.0], coords={"t": [80.0]}, dims=["t"]),
    }

    try:
        concentration, t = example_(**nominal_inputs)
    except Exception as e:
        raise e

    test_case_impurity = Exception_Impurity_Concentration_Test_Case(**nominal_inputs)

    erroneous_input = {"element": 4}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * -1}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * -np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * np.nan}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"].expand_dims("blank")}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"] * -1}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"] * np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"] * -np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"] * np.nan}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_diag": nominal_inputs["Zeff_diag"].expand_dims("blank")}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"impurity_densities": nominal_inputs["impurity_densities"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"impurity_densities": nominal_inputs["impurity_densities"] * -1}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * np.inf
    }
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * -np.inf
    }
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * np.nan
    }
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"].expand_dims("blank")
    }
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * -1}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * -np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * np.nan}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "electron_density": nominal_inputs["electron_density"].expand_dims("blank")
    }
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * -1}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * -np.inf}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * np.nan}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {
        "mean_charge": nominal_inputs["mean_charge"].expand_dims("blank")
    }
    test_case_impurity.call_type_check(**erroneous_input)
