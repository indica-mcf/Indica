from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np

from indica.converters import LineOfSightTransform


# Dummy line-of-sight
machine_dims = ((0.15, 0.85), (-0.75, 0.75))
origin_x = np.array([1.0, 1.0, 1.0], dtype=float)
origin_y = np.array([0.0, 0.0, 0.0], dtype=float)
origin_z = np.array([0.0, 0.0, 0.0], dtype=float)
direction_x = np.array([-0.8, -0.8, -0.8], dtype=float)
direction_y = np.array([0.4, 0.1, 0.0], dtype=float)
direction_z = np.array([0.0, 0.0, 0.0], dtype=float)
name = "dummy_los"

# Optional inputs for the spot
beamlets = int(3 * 3)
focal_length = -100.0  # meter
spot_width = 0.01  # meter
spot_height = 0.01  # meter
spot_shape = "round"
div_width = 100.0 * 1e-3  # radians


los_transform = LineOfSightTransform(
    origin_x,
    origin_y,
    origin_z,
    direction_x,
    direction_y,
    direction_z,
    name=name,
    dl=0.01,
    spot_width=spot_width,
    # spot_height=spot_height,
    spot_shape=spot_shape,
    beamlets=beamlets,
    # div_height=div_height,
    div_width=div_width,
    focal_length=focal_length,
    machine_dimensions=machine_dims,
    passes=1,
)

# Set spot weightings
# spot_weights = SpotWeightings(
#     los_transform, "gaussian", sigma_w=0.003, sigma_v=0.003, p_w=2.0, p_v=2.0
# )
#
# los_transform.set_weightings(spot_weights.weightings)
# print(los_transform.weightings)

# Plotting...
cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(los_transform.x1), dtype=float))

plt.figure()

th = np.linspace(0, 2 * np.pi, 1000)
x_ivc = machine_dims[0][1] * np.cos(th)
y_ivc = machine_dims[0][1] * np.sin(th)
x_cc = machine_dims[0][0] * np.cos(th)
y_cc = machine_dims[0][0] * np.sin(th)

plt.plot(x_cc, y_cc, c="k", lw=2.0)
plt.plot(x_ivc, y_ivc, c="k", lw=2.0)

for x1 in los_transform.x1:
    for beamlet in range(los_transform.beamlets):
        x = los_transform.x.sel(channel=x1, beamlet=beamlet)
        y = los_transform.y.sel(channel=x1, beamlet=beamlet)

        plt.plot(x, y, c=cols[x1])

plt.tight_layout()

plt.figure()

plt.plot(
    [machine_dims[0][1], machine_dims[0][1]],
    [machine_dims[1][0], machine_dims[1][1]],
    c="k",
    lw=2.0,
)

plt.plot(
    [machine_dims[0][0], machine_dims[0][0]],
    [machine_dims[1][0], machine_dims[1][1]],
    c="k",
    lw=2.0,
)

for x1 in los_transform.x1:
    for beamlet in range(los_transform.beamlets):
        R = los_transform.R.sel(channel=x1, beamlet=beamlet)
        z = los_transform.z.sel(channel=x1, beamlet=beamlet)

        plt.plot(R, z, c=cols[x1])

plt.tight_layout()
plt.show(block=True)
