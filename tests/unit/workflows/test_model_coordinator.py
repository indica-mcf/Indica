from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import SXRcamera
from indica.models import ThomsonScattering
from indica.workflows.bda.model_coordinator import ModelCoordinator


def initialise_model_coordinator(model_settings=None):
    model_coordinator = ModelCoordinator(
        {
            "cxff_pi": ChargeExchangeSpectrometer,
            "xrcs": HelikeSpectrometer,
            "ts": ThomsonScattering,
            "sxrc_xy1": SXRcamera,
        },
        model_settings=model_settings,
    )
    return model_coordinator


def initialise_model_coordinator_and_setup(plasma, transforms, equilibrium, model_settings=None):
    model_coordinator = initialise_model_coordinator(model_settings=model_settings)
    model_coordinator.set_plasma(plasma)
    model_coordinator.set_transforms(transforms)
    model_coordinator.set_equilibrium(equilibrium)
    return model_coordinator


class TestModelCoordinator:
    def setup_class(self):
        self.plasma = load_default_objects("st40", "plasma")
        self.transforms = load_default_objects("st40", "geometry")
        self.equilibrium = load_default_objects("st40", "equilibrium")

    def test_initialise_models(self):
        model_coordinator = initialise_model_coordinator()
        for model_name, model in model_coordinator.models.items():
            assert hasattr(model, "name")

    def test_set_plasma(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_plasma(self.plasma)
        for model_name, model in model_coordinator.models.items():
            assert hasattr(model, "plasma")

    def test_set_transforms(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_transforms(self.transforms)
        for model_name, model in model_coordinator.models.items():
            assert hasattr(model, "transform")

    def test_set_equilibrium_without_transform(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_equilibrium(self.equilibrium)
        for model_name, model in model_coordinator.models.items():
            assert not hasattr(model, "equilibrium")

    def test_set_equilibrium_with_transform(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_transforms(self.transforms)
        model_coordinator.set_equilibrium(self.equilibrium)
        for model_name, model in model_coordinator.models.items():
            assert hasattr(model.transform, "equilibrium")

    def test_get_with_all_model_names(self):
        model_coordinator = initialise_model_coordinator_and_setup(self.plasma, self.transforms, self.equilibrium)
        for model_name, model in model_coordinator.models.items():
            assert model_coordinator.get(model_name)

    def test_get_with_wrong_name(self):
        model_coordinator = initialise_model_coordinator_and_setup(self.plasma, self.transforms, self.equilibrium)
        assert not model_coordinator.get("dummy")

    def test_default_call(self):
        model_coordinator = initialise_model_coordinator_and_setup(self.plasma, self.transforms, self.equilibrium)
        model_coordinator()
        assert model_coordinator.binned_data

    def test_call_with_nested_kwargs(self):
        model_coordinator = initialise_model_coordinator_and_setup(self.plasma, self.transforms, self.equilibrium)
        model_coordinator(["xrcs"], **{"xrcs": {"background": 0}})
        assert model_coordinator.binned_data.get("xrcs", {})
        assert model_coordinator.call_kwargs.get("xrcs", {}) == {"background": 0}

    def test_call_with_flat_kwargs(self):
        model_coordinator = initialise_model_coordinator_and_setup(self.plasma, self.transforms, self.equilibrium)
        model_coordinator(["xrcs"], flat_kwargs={"xrcs.background": 0})
        assert model_coordinator.binned_data.get("xrcs", {})
        assert model_coordinator.call_kwargs.get("xrcs", {}) == {"background": 0}


if __name__ == "__main__":
    test = TestModelCoordinator()
    test.setup_class()
    test.test_call_with_flat_kwargs()
