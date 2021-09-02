import unittest

from indica.operators.extrapolate_impurity_density import ExtrapolateImpurityDensity


class Exception_Impurity_Density_Test_Case(unittest.TestCase):
    def __init__(
        self,
        element,
        ion_radiation_loss,
        impurity_radiation_losses,
        sxr_emissivity,
        main_ion_density,
        impurity_densities,
        electron_density,
        electron_temperature,
        truncation_threshold,
        t,
    ):
        self.element = element
        self.ion_radiation_loss = ion_radiation_loss
        self.impurity_radiation_losses = impurity_radiation_losses
        self.sxr_emissivity = sxr_emissivity
        self.main_ion_density = main_ion_density
        self.impurity_densities = impurity_densities
        self.electron_density = electron_density
        self.electron_temperature = electron_temperature
        self.truncation_threshold = truncation_threshold
        self.t = t

        self.nominal_inputs = [
            self.element,
            self.ion_radiation_loss,
            self.impurity_radiation_losses,
            self.sxr_emissivity,
            self.main_ion_density,
            self.impurity_densities,
            self.electron_density,
            self.electron_temperature,
            self.truncation_threshold,
            self.t,
        ]

    def call_type_check(
        self,
        element=None,
        ion_radiation_loss=None,
        impurity_radiation_losses=None,
        sxr_emissivity=None,
        main_ion_density=None,
        impurity_densities=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        t=None,
    ):
        inputs = [
            element,
            ion_radiation_loss,
            impurity_radiation_losses,
            sxr_emissivity,
            main_ion_density,
            impurity_densities,
            electron_density,
            electron_temperature,
            truncation_threshold,
            t,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            element,
            ion_radiation_loss,
            impurity_radiation_losses,
            sxr_emissivity,
            main_ion_density,
            impurity_densities,
            electron_density,
            electron_temperature,
            truncation_threshold,
            t,
        ) = inputs

        with self.assertRaises(TypeError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)

    def call_value_check(
        self,
        element=None,
        ion_radiation_loss=None,
        impurity_radiation_losses=None,
        sxr_emissivity=None,
        main_ion_density=None,
        impurity_densities=None,
        electron_density=None,
        electron_temperature=None,
        truncation_threshold=None,
        t=None,
    ):
        inputs = [
            element,
            ion_radiation_loss,
            impurity_radiation_losses,
            sxr_emissivity,
            main_ion_density,
            impurity_densities,
            electron_density,
            electron_temperature,
            truncation_threshold,
            t,
        ]

        for i, iinput in enumerate(inputs):
            if iinput is None:
                inputs[i] = self.nominal_inputs[i]

        (
            element,
            ion_radiation_loss,
            impurity_radiation_losses,
            sxr_emissivity,
            main_ion_density,
            impurity_densities,
            electron_density,
            electron_temperature,
            truncation_threshold,
            t,
        ) = inputs

        with self.assertRaises(ValueError):
            example_ = ExtrapolateImpurityDensity()
            example_(*inputs)
