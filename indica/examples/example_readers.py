from indica.defaults.load_defaults import load_default_objects
from indica.models import ChargeExchangeSpectrometer
from indica.models import HelikeSpectrometer
from indica.models import PinholeCamera
from indica.models import ThomsonScattering
from indica.readers.modelreader import ModelReader
import matplotlib.pylab as plt
from indica.utilities import set_axis_sci, set_plot_colors
import numpy as np

CMAP, COLORS = set_plot_colors()

def example_model_reader(plot=False):
    plasma = load_default_objects("st40", "plasma")
    transforms = load_default_objects("st40", "geometry")
    equilibrium = load_default_objects("st40", "equilibrium")
    models = {
            "cxff_pi": ChargeExchangeSpectrometer,
            "xrcs": HelikeSpectrometer,
            "ts": ThomsonScattering,
            "sxrc_xy1": PinholeCamera,
    }

    model_reader = ModelReader(models)
    model_reader.set_plasma(plasma)
    model_reader.set_geometry_transforms(transforms, equilibrium)
    bckc = model_reader()

    if plot:
        plt.ioff()
        plt.figure()
        ti = bckc["cxff_pi"]["ti"]
        cols = CMAP(np.linspace(0.75, 0.1, len(ti.t), dtype=float))
        for i, t in enumerate(ti.t):
            plasma.ion_temperature.sel(t=t, method="nearest").plot(color=cols[i])
        plasma.ion_temperature.sel(t=t, method="nearest").plot(color=cols[i], label="Plasma Ti")
        for i, t in enumerate(ti.t):
            plt.plot(ti.transform.rhop.sel(t=t), ti.sel(t=t), "o", color=cols[i])
        plt.plot(ti.transform.rhop.sel(t=t), ti.sel(t=t), "o", color=cols[i], label="CXRS measurement")
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