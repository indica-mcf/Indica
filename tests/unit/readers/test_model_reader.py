from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.operators.atomic_data import default_atomic_data
from indica.readers.modelreader import ModelReader


PLASMA = load_default_objects("st40", "plasma")
TRANSFORMS = load_default_objects("st40", "geometry")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
FZ, POWER_LOSS = default_atomic_data(PLASMA.elements)


def initialise_model_reader(model_kwargs={"sxrc_xy1": {"power_loss": POWER_LOSS}}):
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


class TestModelReader:
    def setup_class(self):
        return

    def test_initialise_models(self):
        model_reader = initialise_model_reader()
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "name")

    def test_set_plasma(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(PLASMA)
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "plasma")

    def test_set_transforms(self):
        model_reader = initialise_model_reader()
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        for model_name, model in model_reader.models.items():
            assert hasattr(model, "transform")

    def test_set_equilibrium_with_transform(self):
        model_reader = initialise_model_reader()
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        for model_name, model in model_reader.models.items():
            assert hasattr(model.transform, "equilibrium")

    def test_get_with_all_model_names(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(PLASMA)
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        for model_name, model in model_reader.models.items():
            assert model_reader.get(model_name)

    def test_get_with_wrong_name(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(PLASMA)
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        assert not model_reader.get("dummy")

    def test_default_call(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(PLASMA)
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        bckc = model_reader()
        assert bckc

    def test_call_with_kwargs(self):
        model_reader = initialise_model_reader()
        model_reader.set_plasma(PLASMA)
        model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
        bckc = model_reader(["xrcs"], **{"xrcs": {"background": 0}})
        assert bckc.get("xrcs", {})
