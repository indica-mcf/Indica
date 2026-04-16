import matplotlib.pylab as plt

from indica.examples.example_readers import example_model_reader
from indica.plotters import DataPlotter
from indica.readers import SOLPSReader


def example_dataplotter():
    BCKC, MODEL_READER = example_model_reader(plot=False)

    plasma = MODEL_READER.plasma
    plotter = DataPlotter(0, plasma.t)

    plotter.plot(BCKC, "ts", "ne")
    plotter.plot(BCKC, "cxff_pi", "ti")
    plotter.plot(BCKC, "sxrc_xy1", "brightness")

    plotter.plot(plasma, "plasma", "electron_density", ylog=False)
    plotter.plot(plasma, "plasma", "zeff", ylog=False, sci=False)


def example_plot_solps():
    solps = SOLPSReader(13565, 0.16)
    processed = {}
    processed["solps"] = solps.get()

    t = processed["solps"]["ne"].t.values
    plotter = DataPlotter(13565, t, nplot=1)

    plotter.plot(processed, "solps", "nion", element="h", ion_charge=0)
    plotter.plot(processed, "solps", "nion", element="c", ion_charge=0)


if __name__ == "__main__":
    plt.ioff()
    _ = example_dataplotter()
    plt.show()
