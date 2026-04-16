import matplotlib.pylab as plt

from indica.examples import example_dataplotter


def test_dataplotter():
    plt.ioff()
    example_dataplotter()
    plt.close("all")
