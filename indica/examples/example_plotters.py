import matplotlib.pylab as plt

from indica.examples.example_readers import example_model_reader
from indica.plotters import DataPlotter
from indica.readers import SOLPSReader


def example_dataplotter():
    BCKC, MODEL_READER = example_model_reader(plot=False)

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

    plotter.plot_plasma_attribute(
        "fz", plasma, element="ar", ylog=False, sci=False, to_plot=("profiles",)
    )


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
