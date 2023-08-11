import pickle

import matplotlib.pylab as plt
import numpy as np

from indica.utilities import coord_array

pulse = 90279
server = "https://sal.jet.uk"


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


with open("stan_model_data.pkl", "rb") as pkl_file:
    pre_computed = pickle.load(pkl_file)

los_values = pre_computed["bolo_los_data"].los_values
los_transform = los_values.transform
kb5v_coords = los_values.bolo_kb5v_coords
kb5v_los_pos = coord_array(np.array([0, 1]), "bolo_kb5v_los_position")

starts_ends = los_transform.convert_to_Rz(kb5v_coords, kb5v_los_pos, None)
R_start = starts_ends[0].isel(bolo_kb5v_los_position=0)
R_end = starts_ends[0].isel(bolo_kb5v_los_position=1)
Z_start = starts_ends[1].isel(bolo_kb5v_los_position=0)
Z_end = starts_ends[1].isel(bolo_kb5v_los_position=1)


plot_jet_wall()
for i in range(len(R_start)):
    R_vals = np.array([R_start[i], R_end[i]])
    Z_vals = np.array([Z_start[i], Z_end[i]])
    plt.plot(R_vals, Z_vals, label=kb5v_coords[i])

machine_dims = ((1.83, 3.9), (-1.75, 2.0))
plt.xlim(machine_dims[0])
plt.ylim(machine_dims[1])
plt.legend()
plt.show()
