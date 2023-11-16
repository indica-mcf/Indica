import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm


# Add Indica to python path
path_to_indica = '/home/jonathan.wood/git_home/Indica/'
sys.path.append(path_to_indica)

# Import Indica things
from indica.converters import LineOfSightTransform


# Dummy line-of-sight
machine_dims = ((0.15, 0.85), (-0.75, 0.75))
origin_x = np.array([1.0], dtype=float)
origin_y = np.array([0.0], dtype=float)
origin_z = np.array([0.0], dtype=float)
direction_x = np.array([-1.0], dtype=float)
direction_y = np.array([0.0], dtype=float)
direction_z = np.array([0.0], dtype=float)
name = 'wowzers'

beamlets = int(3 * 3)
spot_width = 0.01
spot_height = 0.01
spot_shape = 'round'
div_w = 100 * 1e-3  # radians
div_h = 100 * 1e-3  # radians

los = LineOfSightTransform(
    origin_x,
    origin_y,
    origin_z,
    direction_x,
    direction_y,
    direction_z,
    name=name,
    dl=0.01,
    spot_width=spot_width,
    spot_height=spot_height,
    spot_shape=spot_shape,
    beamlets=beamlets,
    div_h=div_h,
    div_w=div_w,
    machine_dimensions=machine_dims,
)

cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(los.x1), dtype=float))

plt.figure()

th = np.linspace(0, 2*np.pi, 1000)
x_ivc = machine_dims[0][1] * np.cos(th)
y_ivc = machine_dims[0][1] * np.sin(th)
x_cc = machine_dims[0][0] * np.cos(th)
y_cc = machine_dims[0][0] * np.sin(th)

plt.plot(x_cc, y_cc, c='k', lw=2.0)
plt.plot(x_ivc, y_ivc, c='k', lw=2.0)

for x1 in los.x1:
    for beamlet in range(los.beamlets):
        x = los.x.sel(channel=x1, beamlet=beamlet)
        y = los.y.sel(channel=x1, beamlet=beamlet)

        plt.plot(x, y, c=cols[x1])

plt.tight_layout()

plt.figure()

plt.plot(
    [machine_dims[0][1], machine_dims[0][1]],
    [machine_dims[1][0], machine_dims[1][1]],
    c='k',
    lw=2.0,
)

plt.plot(
    [machine_dims[0][0], machine_dims[0][0]],
    [machine_dims[1][0], machine_dims[1][1]],
    c='k',
    lw=2.0,
)

for x1 in los.x1:
    for beamlet in range(los.beamlets):
        R = los.R.sel(channel=x1, beamlet=beamlet)
        z = los.z.sel(channel=x1, beamlet=beamlet)

        plt.plot(R, z, c=cols[x1])

plt.tight_layout()
plt.show(block=True)


print(los.spot_height)
print(los.spot_width)
print(los.spot_shape)
print(los.beamlets)

print(los.x)
print(los.x_start)
