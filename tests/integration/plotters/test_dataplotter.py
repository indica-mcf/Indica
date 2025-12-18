from indica.examples.example_readers import example_model_reader
from indica.plotters import DataPlotter

BCKC, MODEL_READER = example_model_reader(plot=False)


def test_dataplotter(plot=False):
    plasma = MODEL_READER.plasma
    plotter = DataPlotter(0, plasma.t)

    plotter.plot_thomson_scattering("ts", BCKC["ts"])
    plotter.plot_radiation("sxrc_xy1", BCKC["sxrc_xy1"])

    plotter.plot_plasma_attribute(
        "electron_density", plasma, ylog=False, to_plot=("profiles",)
    )

    plotter.plot_plasma_attribute(
        "zeff", plasma, ylog=False, sci=False, to_plot=("profiles",)
    )
