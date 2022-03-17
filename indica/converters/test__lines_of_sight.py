import sys
sys.path.insert(0, "/home/jonathan.wood/git_home/Indica")
sys.path.remove("/home/marco.sertoli/python/Indica")
from indica.converters import lines_of_sight
from indica.converters import flux_surfaces
from indica import equilibrium
from hda_jw.read_st40 import ST40data
from matplotlib import pyplot as plt
from xarray import DataArray
import numpy as np

# Line of sight origin tuple
origin = (0.9, -0.2, 0.0)  # [xyz]

# Line of sight direction
direction = (-1.0, 0.0, 0.0)  # [xyz]

# machine dimensions
machine_dims = ((0.175, 1.0), (0.0, 0.0))

# name
name = "los_test"

# Equilibrium
st40_data = ST40data(pulse=9780, tstart=0.04, tend=0.085)
st40_data.get_efit()
equil = equilibrium.Equilibrium(st40_data.data['efit'])

# Flux surface coordinate
flux_coord = flux_surfaces.FluxSurfaceCoordinates("poloidal")
flux_coord.set_equilibrium(equil)

# Set-up line of sight class
los = lines_of_sight.LinesOfSightTransform(
    origin[0], origin[1], origin[2], direction[0], direction[1], direction[2],
    machine_dimensions=machine_dims, name=name
)

# Assign flux transform
los.assign_flux_transform(flux_coord)

# Convert_to_rho method
los.convert_to_rho(t=0.045)

# Check method #1: convert_to_Rz, inputs: "x1", "x2", "t"
x1 = 0.0  # does nothing
x2 = DataArray(np.linspace(0.0, 1.0, 350, dtype=float))  # index along line of sight, must be a DataArray
t = 0.0  # does nothing
r_, z_ = los.convert_to_Rz(x1, x2, t)
print(f'r_ = {r_}')
print(f'z_ = {z_}')

# Check method #2: convert_from_Rz, inputs: "R", "Z", "t"
# R_test = DataArray(np.linspace(0.8, 0.2, 200))
# Z_test = DataArray(np.linspace(0.0, 0.0, len(R_test)))
R_test = DataArray(0.5)  # Does not work as an array
Z_test = DataArray(0.0)  # Does not work as an array
x1_out1, x2_out2 = los.convert_from_Rz(R_test, Z_test, t)
print(f'x2_out2 = {x2_out2}')

# Check method #3: distance, inputs: "x1", "x2", "t"
dist = los.distance('dim_0', x1, x2, t)
print(f'dist = {dist}')

# Check!
print(f'los.dell = {los.dell}')
print(f'los.x2 = {los.x2}')
print(f'los.x = {los.x}')
print(f'los.y = {los.y}')
print(f'los.z = {los.z}')
print(f'los.r = {los.r}')
print(' ')
print(f'los.rho = {los.rho}')

# centre column
th = np.linspace(0.0, 2*np.pi, 1000)
x_cc = machine_dims[0][0] * np.cos(th)
y_cc = machine_dims[0][0] * np.sin(th)

# IVC
x_ivc = machine_dims[0][1] * np.cos(th)
y_ivc = machine_dims[0][1] * np.sin(th)


plt.figure()
plt.plot(los.x2, los.rho[0].sel(t=0.045, method='nearest'), 'b')
plt.ylabel('rho')


plt.figure()
plt.plot(x_cc, y_cc, 'k--')
plt.plot(x_ivc, y_ivc, 'k--')
plt.plot(los.x_start, los.y_start, 'ro', label='start')
plt.plot(los.x_end, los.y_end, 'bo', label='end')
plt.plot(los.x, los.y, 'g', label='los')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show(block=True)
