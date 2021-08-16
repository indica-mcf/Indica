from typing import Dict
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
from indica.operators.spline_fit import Spline
from indica.readers.adas import ADASReader
from indica.utilities import broadcast_spline
from ..test_equilibrium_single import equilibrium_dat_and_te


class Exception_Impurity_Concentration_Test_Case(unittest.TestCase):
    """Test case for testing type and value errors in ImpurityConcentration call."""

    def __init__(
        self,
        element,
        Zeff_LoS,
        impurity_densities,
        electron_density,
        mean_charge,
        flux_surfaces,
        t,
    ):
        """Initialise the test case with a set of nominal inputs."""
        self.element = element
        self.Zeff_LoS = Zeff_LoS
        self.impurity_densities = impurity_densities
        self.electron_density = electron_density
        self.mean_charge = mean_charge
        self.flux_surfaces = flux_surfaces
        self.t = t

        self.nominal_inputs = [
            self.element,
            self.Zeff_LoS,
            self.impurity_densities,
            self.electron_density,
            self.mean_charge,
            self.flux_surfaces,
            self.t,
        ]

    def call_type_check(
        self,
        element=None,
        Zeff_LoS=None,
        impurity_densities=None,
        electron_density=None,
        mean_charge=None,
        flux_surfaces=None,
        t=None,
    ):
        """Test TypeError for ImpurityConcentration call."""
        inputs = [
            element,
            Zeff_LoS,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            element,
            Zeff_LoS,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ) = inputs

        with self.assertRaises(TypeError):
            example_ = ImpurityConcentration()
            example_(*inputs)

    def call_value_check(
        self,
        element=None,
        Zeff_LoS=None,
        impurity_densities=None,
        electron_density=None,
        mean_charge=None,
        flux_surfaces=None,
        t=None,
    ):
        """Test ValueError for ImpurityConcentration call."""
        inputs = [
            element,
            Zeff_LoS,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ]
        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            element,
            Zeff_LoS,
            impurity_densities,
            electron_density,
            mean_charge,
            flux_surfaces,
            t,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ImpurityConcentration()
            example_(*inputs)


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    """Calculate and output Fractional abundance at t=infinity for calculating
    the mean charge in test_impurity_concentration()

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Times at which to define input_Ne and input_Te (also used for the output)

    Returns
    -------
    F_z_tinf
        Fractional abundance of the ionisation stages of the element at t=infinity.
    """
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


def nominal_output_checks(
    example_impurity_concetration: ImpurityConcentration,
    nominal_inputs: Dict,
    upper_limit: float,
):
    """Tests the output of nominal inputs for beryllium, neon, nickel and tungsten.

    Parameters
    ----------
    example_impurity_concentration
        Callable ImpurityConcentration object.
    nominal_inputs
        Dictionary of keyword arguments to pass to the ImpurityConcentration object.
    upper_limit
        Upper limit of concentration for the given element. (In fractional units)
        eg. 0.04 for beryllium (which translates to 4%%)
    """

    try:
        concentration, t = example_impurity_concetration(**nominal_inputs)
    except Exception as e:
        raise e

    element_name = nominal_inputs["element"]

    try:
        assert np.all(concentration > 0.0)
    except AssertionError:
        raise ValueError(
            f"Some concentration values for {element_name} are less than zero."
        )

    try:
        assert np.all(concentration <= upper_limit)
    except AssertionError:
        raise ValueError(
            f"Some concentration values for {element_name} are \
                more than {upper_limit * 100}%%."
        )


def test_impurity_concentration():
    """Test ImpurityConcentration.__call__."""
    example_ = ImpurityConcentration()

    t = np.linspace(75.0, 80.0, 5)

    Zeff_LoS = DataArray(
        data=np.ones(*t.shape) * 1.85,
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
                name="Zeff_LoS",
            ),
            "Zeff_LoS_coords": DataArray(data=np.array([0]), dims=["Zeff_LoS_coords"]),
        },
    )

    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    electron_density = DataArray(
        data=np.tile(np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(t), 1)).T,
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    beryllium_impurity_conc = 0.03 * electron_density
    neon_impurity_conc = 0.02 * electron_density
    nickel_impurity_conc = 0.0002 * electron_density
    tungsten_impurity_conc = 0.00005 * electron_density

    # be, ne, ni, w
    elements = [4, 10, 28, 74]
    elements = [ELEMENTS_BY_ATOMIC_NUMBER.get(i) for i in elements]

    impurity_densities = DataArray(
        data=np.ones((len(elements), *rho_profile.shape, *t.shape)),
        coords=[("elements", elements), ("rho", rho_profile), ("t", t)],
        dims=["elements", "rho", "t"],
    )
    impurity_densities.data[0] = beryllium_impurity_conc
    impurity_densities.data[1] = neon_impurity_conc
    impurity_densities.data[2] = nickel_impurity_conc
    impurity_densities.data[3] = tungsten_impurity_conc

    rho = np.linspace(0.0, 1.0, 20)
    rho = DataArray(data=rho, coords={"rho": rho}, dims=["rho"])

    dummy_coordinates = FluxSurfaceCoordinates("poloidal")

    electron_density_spline = Spline(electron_density, "rho", dummy_coordinates)
    electron_density = broadcast_spline(
        electron_density_spline.spline,
        electron_density_spline.spline_dims,
        electron_density_spline.spline_coords,
        rho,
    )

    electron_density = electron_density.transpose("rho", "t")

    impurity_densities_spline = Spline(impurity_densities, "rho", dummy_coordinates)
    impurity_densities = broadcast_spline(
        impurity_densities_spline.spline,
        impurity_densities_spline.spline_dims,
        impurity_densities_spline.spline_coords,
        rho,
    )

    impurity_densities = impurity_densities.transpose("elements", "rho", "t")

    mean_charge = zeros_like(impurity_densities)

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
        "Zeff_LoS": Zeff_LoS,
        "impurity_densities": impurity_densities,
        "electron_density": electron_density,
        "mean_charge": mean_charge,
        "flux_surfaces": flux_surfs,
        "t": DataArray(data=t, coords={"t": t}, dims=["t"]),
    }

    nominal_output_checks(example_, nominal_inputs, 0.04)

    nominal_inputs["element"] = "neon"

    nominal_output_checks(example_, nominal_inputs, 0.04)

    nominal_inputs["element"] = "nickel"

    nominal_output_checks(example_, nominal_inputs, 1e-3)

    nominal_inputs["element"] = "tungsten"

    nominal_output_checks(example_, nominal_inputs, 1e-4)

    # Input type and value checks
    test_case_impurity = Exception_Impurity_Concentration_Test_Case(**nominal_inputs)

    erroneous_input = {"element": 4}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * -1}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * -np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"] * np.nan}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"t": nominal_inputs["t"].expand_dims("blank")}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"] * -1}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"] * np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"] * -np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"] * np.nan}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"Zeff_LoS": nominal_inputs["Zeff_LoS"].expand_dims("blank")}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"impurity_densities": nominal_inputs["impurity_densities"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"impurity_densities": nominal_inputs["impurity_densities"] * -1}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * np.inf
    }
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * -np.inf
    }
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"] * np.nan
    }
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "impurity_densities": nominal_inputs["impurity_densities"].rename(
            {"elements": "elements", "rho": "theta", "t": "t"}
        )
    }
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * -1}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * -np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"electron_density": nominal_inputs["electron_density"] * np.nan}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "electron_density": nominal_inputs["electron_density"].expand_dims("blank")
    }
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"].data}
    test_case_impurity.call_type_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * -1}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * -np.inf}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {"mean_charge": nominal_inputs["mean_charge"] * np.nan}
    test_case_impurity.call_value_check(**erroneous_input)

    erroneous_input = {
        "mean_charge": nominal_inputs["mean_charge"].expand_dims("blank")
    }
    test_case_impurity.call_value_check(**erroneous_input)
