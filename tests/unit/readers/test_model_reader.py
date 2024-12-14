from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.readers.modelreader import ModelReader


def initialise_model_reader(model_kwargs={}):
    model_reader = ModelReader(
        {
            "cxff_pi": ChargeExchangeSpectrometer,
            "xrcs": HelikeSpectrometer,
            "ts": ThomsonScattering,
            "sxrc_xy1": PinholeCamera,
        },
        model_kwargs=model_kwargs,
    )
    return model_reader


def initialise_model_reader_and_setup(plasma, transforms, equilibrium, model_kwargs={}):
    model_reader = initialise_model_reader(model_kwargs=model_kwargs)
    model_reader.set_plasma(plasma)
    model_reader.set_geometry_transforms(transforms, equilibrium)
    return model_reader


class TestModelReader:
    def setup_class(self):
        self.plasma = load_default_objects("st40", "plasma")
        self.transforms = load_default_objects("st40", "geometry")
        self.equilibrium = load_default_objects("st40", "equilibrium")

    def test_initialise_models(self):
        model_reader = initialise_model_reader()
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "name")

    def test_set_plasma(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(self.plasma)
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "plasma")

    def test_set_transforms(self):
        model_reader = initialise_model_reader()
        model_reader.set_geometry_transforms(self.transforms, self.equilibrium)
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "transform")

    def test_set_equilibrium_with_transform(self):
        model_reader = initialise_model_reader()
        model_reader.set_geometry_transforms(self.transforms, self.equilibrium)
        for model_name, model in model_reader.models.items():
            assert hasattr(model.transform, "equilibrium")

    def test_get_with_all_model_names(self):
        model_reader = initialise_model_reader_and_setup(
            self.plasma, self.transforms, self.equilibrium
        )
        for model_name, model in model_reader.models.items():
            assert model_reader.get(model_name)

    def test_get_with_wrong_name(self):
        model_reader = initialise_model_reader_and_setup(
            self.plasma, self.transforms, self.equilibrium
        )
        assert not model_reader.get("dummy")

    def test_default_call(self):
        model_reader = initialise_model_reader_and_setup(
            self.plasma, self.transforms, self.equilibrium
        )
        bckc = model_reader()
        assert bckc

    def test_call_with_kwargs(self):
        model_reader = initialise_model_reader_and_setup(
            self.plasma, self.transforms, self.equilibrium
        )
        bckc = model_reader(["xrcs"], **{"xrcs": {"background": 0}})
        assert bckc.get("xrcs", {})
