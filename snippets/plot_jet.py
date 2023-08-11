import matplotlib.pylab as plt
import numpy as np


def plot_jet_wall(filewall="snippets/wall_coords_jet.txt", divertor=True):
    """
    Plot JET first wall poloidal cross-section
    """
    data = np.genfromtxt(filewall)

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], color="black", linewidth=2.5)
    if divertor:
        plt.xlim(2.3, 3.0)
        plt.ylim(-1.8, -1.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("R (m)")
    plt.ylabel("z (m)")
