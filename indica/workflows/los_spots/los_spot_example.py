import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm


# Add Indica to python path
# path_to_indica = '/home/jonathan.wood/git_home/Indica/'
# sys.path.append(path_to_indica)

# Import Indica things
from indica.converters import LineOfSightTransform
from indica.converters import SpotWeightings
from indica.models.plasma import example_run as example_plasma
from indica.readers.read_st40 import ReadST40


# Dummy line-of-sight
machine_dims = ((0.15, 0.85), (-0.75, 0.75))
origin_x = np.array([1.0], dtype=float)
origin_y = np.array([0.0], dtype=float)
origin_z = np.array([0.0], dtype=float)
direction_x = np.array([-1.0], dtype=float)
direction_y = np.array([0.0], dtype=float)
direction_z = np.array([0.0], dtype=float)
name = 'wowzers'

origin_x = np.array([1.0], dtype=float)
origin_y = np.array([0.0], dtype=float)
origin_z = np.array([0.0], dtype=float)
direction_x = np.array([-0.8], dtype=float)
direction_y = np.array([0.4], dtype=float)
direction_z = np.array([0.0], dtype=float)
name = 'wowzers'

origin_x = np.array([1.0, 1.0], dtype=float)
origin_y = np.array([0.0, 0.0], dtype=float)
origin_z = np.array([0.0, 0.0], dtype=float)
direction_x = np.array([-0.8, -0.8], dtype=float)
direction_y = np.array([0.4, 0.1], dtype=float)
direction_z = np.array([0.0, 0.0], dtype=float)
name = 'wowzers'

origin_x = np.array([1.0, 1.0, 1.0], dtype=float)
origin_y = np.array([0.0, 0.0, 0.0], dtype=float)
origin_z = np.array([0.0, 0.0, 0.0], dtype=float)
direction_x = np.array([-0.8, -0.8, -0.8], dtype=float)
direction_y = np.array([0.4, 0.1, 0.0], dtype=float)
direction_z = np.array([0.0, 0.0, 0.0], dtype=float)
name = 'wowzers'


beamlets = int(3 * 3)
spot_width = 0.01
spot_height = 0.01
spot_shape = 'round'
div_w = 70 * 1e-3  # radians
div_h = 70 * 1e-3  # radians

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
    spot_height=spot_height,
    spot_shape=spot_shape,
    beamlets=beamlets,
    div_h=div_h,
    div_w=div_w,
    machine_dimensions=machine_dims,
    passes=1,
)

# Set spot weightings
spot_weights = SpotWeightings(
    los_transform,
    "gaussian",
    sigma_w=0.003,
    sigma_v=0.003,
    p_w=2.,
    p_v=2.
)

los_transform.set_weightings(spot_weights.weightings)
print(los_transform.weightings)


# Plotting...
cols = cm.gnuplot2(np.linspace(0.3, 0.75, len(los_transform.x1), dtype=float))

plt.figure()

th = np.linspace(0, 2*np.pi, 1000)
x_ivc = machine_dims[0][1] * np.cos(th)
y_ivc = machine_dims[0][1] * np.sin(th)
x_cc = machine_dims[0][0] * np.cos(th)
y_cc = machine_dims[0][0] * np.sin(th)

plt.plot(x_cc, y_cc, c='k', lw=2.0)
plt.plot(x_ivc, y_ivc, c='k', lw=2.0)

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
    c='k',
    lw=2.0,
)

plt.plot(
    [machine_dims[0][0], machine_dims[0][0]],
    [machine_dims[1][0], machine_dims[1][1]],
    c='k',
    lw=2.0,
)

for x1 in los_transform.x1:
    for beamlet in range(los_transform.beamlets):
        R = los_transform.R.sel(channel=x1, beamlet=beamlet)
        z = los_transform.z.sel(channel=x1, beamlet=beamlet)

        plt.plot(R, z, c=cols[x1])

plt.tight_layout()
#plt.show(block=True)

'''
NEXT UP PLASMA
'''

# Inputs -- default tests equilibrium
run = 'RUN01'
pulse = 10009
tstart = 0.05
tend = 0.06
dt = 0.01

# Read ST40 data
st40 = ReadST40(tstart=tstart, tend=tend, dt=dt, pulse=pulse)
st40()

# Get plasma
plasma = example_plasma(tstart=tstart, tend=tend, dt=dt, pulse=pulse)
plasma.set_equilibrium(st40.equilibrium)
los_transform.set_equilibrium(plasma.equilibrium)

#
time = plasma.t

# Along LOS
Ti = plasma.ion_temperature.sel(element="c")
print(Ti)
Ti_along_los = los_transform.map_profile_to_los(Ti, t=time)
print(Ti_along_los)

for i_time in range(len(time)):
    plt.figure()
    for i_channel in range(len(los_transform.x1)):
        for i_beamlet in range(los_transform.beamlets):
            plt.plot(
                Ti_along_los.los_position,
                Ti_along_los.sel(channel=i_channel, beamlet=i_beamlet, t=time[i_time]),
                c = cols[i_channel],
            )

    plt.ylabel('Ti (eV)')

plt.show(block=True)


# Line integration
rho = los_transform.equilibrium.rho.interp(t=time)
R = rho.R
z = rho.z
b_tot, t = plasma.equilibrium.Btot(R, z, t=time)
b_tot_los_int = los_transform.integrate_on_los(b_tot, t=time)

b_tot_los_int_beamlets = los_transform.integrate_on_los(b_tot, t=time, sum_beamlet=False)
print(b_tot_los_int)
print(b_tot_los_int_beamlets)

plt.figure()
plt.contourf(
    R, z, b_tot.sel(t=time[0])
)
plt.colorbar()

plt.figure()
for i_beamlet in range(los_transform.beamlets):
    plt.plot(b_tot_los_int_beamlets.sel(channel=0, beamlet=i_beamlet))

plt.figure()
for i_channel in range(len(los_transform.x1)):
    plt.plot(b_tot_los_int.sel(channel=i_channel))


plt.show(block=True)
