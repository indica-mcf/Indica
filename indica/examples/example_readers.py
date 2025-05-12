import matplotlib.pylab as plt
import numpy as np

from indica.configs.readers.adasconf import ADF11
from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.operators.atomic_data import default_atomic_data
from indica.readers import ADASReader
from indica.readers.modelreader import ModelReader
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_colors


CMAP, COLORS = set_plot_colors()
PLASMA = load_default_objects("st40", "plasma")
TRANSFORMS = load_default_objects("st40", "geometry")
EQUILIBRIUM = load_default_objects("st40", "equilibrium")
MODELS = {
    "cxff_pi": ChargeExchangeSpectrometer,
    "xrcs": HelikeSpectrometer,
    "ts": ThomsonScattering,
    "sxrc_xy1": PinholeCamera,
}


def example_model_reader(plot=False):
    _, power_loss = default_atomic_data(PLASMA.elements)

    model_reader = ModelReader(
        MODELS, model_kwargs={"sxrc_xy1": {"power_loss": power_loss}}
    )
    model_reader.set_plasma(PLASMA)
    model_reader.set_geometry_transforms(TRANSFORMS, EQUILIBRIUM)
    bckc = model_reader()

    if plot:
        plt.ioff()
        plt.figure()
        ti = bckc["cxff_pi"]["ti"]
        cols = CMAP(np.linspace(0.75, 0.1, len(ti.t), dtype=float))
        for i, t in enumerate(ti.t):
            PLASMA.ion_temperature.sel(t=t, method="nearest").plot(color=cols[i])
        PLASMA.ion_temperature.sel(t=t, method="nearest").plot(
            color=cols[i], label="Plasma Ti"
        )
        for i, t in enumerate(ti.t):
            plt.plot(ti.transform.rhop.sel(t=t), ti.sel(t=t), "o", color=cols[i])
        plt.plot(
            ti.transform.rhop.sel(t=t),
            ti.sel(t=t),
            "o",
            color=cols[i],
            label="CXRS measurement",
        )
        plt.title("CXRS ion temperature")
        plt.legend()
        set_axis_sci()
        ti.transform.plot()
        plt.show(block=True)

        plt.figure()
        spectra = bckc["xrcs"]["spectra_raw"]
        cols = CMAP(np.linspace(0.75, 0.1, len(spectra.t), dtype=float))
        for i, t in enumerate(spectra.t):
            plt.plot(spectra.wavelength, spectra.sel(t=t), color=cols[i])
        plt.title("XRCS spectra")
        plt.legend()
        set_axis_sci()

        spectra.transform.plot()
        plt.show(block=True)

    return bckc, model_reader


def example_adf11(
    element: str = "h",
    file_type: str = None,
    **kwargs,
):
    adas_reader = ADASReader()

    if element not in ADF11.keys():
        raise ValueError(f"{element} not in ADF11 dictionary")

    output = {}
    if file_type is None:
        file_types = ADF11[element].keys()
    else:
        file_types = [file_type]

    for file_type in file_types:
        try:
            tmp = adas_reader.get_adf11(file_type, element, ADF11[element][file_type])
        except Exception as e:
            print(f"Error reading ADAS file {element}:{file_type}: {e}")
            continue

        output[file_type] = tmp

    return output
