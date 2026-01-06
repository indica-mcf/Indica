from indica.examples.example_readers import example_model_reader
from indica.plotters import DataPlotter

BCKC, MODEL_READER = example_model_reader(plot=False)


def test_dataplotter(plot=False):
    plasma = MODEL_READER.plasma
    plotter = DataPlotter(0, plasma.t)

    plotter.plot(BCKC, "ts", "ne")
    plotter.plot(BCKC, "sxrc_xy1", "brightness")

    plotter.plot(plasma, "plasma", "electron_density", ylog=False)
    plotter.plot(plasma, "plasma", "zeff", ylog=False, sci=False)

    # Below requires phantom equilibrium to calculate volume integral
    # plotter.plot(plasma, "plasma", "prad_tot", ylog=False)
