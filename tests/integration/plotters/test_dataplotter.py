import matplotlib.pylab as plt

from indica.examples import dataplotter


def _test_dataplotter():
    plt.ioff()
    dataplotter()
    plt.close("all")
