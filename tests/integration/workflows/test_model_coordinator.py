from indica.workflows.model_coordinator import ModelCoordinator
from indica.models import ChargeExchangeSpectrometer, HelikeSpectrometer, ThomsonScattering, SXRcamera
from indica.defaults.load_defaults import load_default_objects


def initialise_model_coordinator(model_settings=None):
    model_coordinator = ModelCoordinator({"cxff_pi": ChargeExchangeSpectrometer, "xrcs": HelikeSpectrometer,
                                          "ts": ThomsonScattering, "sxrc_xy1": SXRcamera},
                                         model_settings=model_settings)
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

    def test_get(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_plasma(self.plasma)
        model_coordinator.set_transforms(self.transforms)
        model_coordinator.set_equilibrium(self.equilibrium)

        for model_name, model in model_coordinator.models.items():
            assert model_coordinator.get(model_name)

    def test_get_with_wrong_name(self):
        model_coordinator = initialise_model_coordinator()
        model_coordinator.set_plasma(self.plasma)
        model_coordinator.set_transforms(self.transforms)
        model_coordinator.set_equilibrium(self.equilibrium)
        assert not model_coordinator.get("dummy")


if __name__ == "__main__":
    test = TestModelCoordinator()
    test.setup_class()
    test.test_get_with_wrong_name()
