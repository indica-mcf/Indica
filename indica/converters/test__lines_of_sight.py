import sys
sys.path.insert(0, "/home/jonathan.wood/git_home/Indica")
sys.path.remove("/home/marco.sertoli/python/Indica")
from indica.converters import lines_of_sight
from matplotlib import pyplot as plt
import numpy as np

# Line of sight origin tuple
origin = (0.9, -0.1, 0.0)  # [xyz]

# Line of sight direction
direction = (-1.0, 0.0, 0.0)  # [xyz]

# machine dimensions
machine_dims = ((0.175, 1.0), (0.0, 0.0))

# name
name = "los_test"

# Set-up line of sight class
los = lines_of_sight.LinesOfSightTransform(origin, direction, machine_dimensions=machine_dims, name=name)

print(f'los.dell = {los.dell}')
print(f'los.x2 = {los.x2}')
print(f'los.x = {los.x}')
print(f'los.y = {los.y}')
print(f'los.z = {los.z}')
print(f'los.r = {los.r}')

# centre column
th = np.linspace(0.0, 2*np.pi, 1000)
x_cc = machine_dims[0][0] * np.cos(th)
y_cc = machine_dims[0][0] * np.sin(th)

# IVC
x_ivc = machine_dims[0][1] * np.cos(th)
y_ivc = machine_dims[0][1] * np.sin(th)

plt.figure()
plt.plot(x_cc, y_cc, 'k--')
plt.plot(x_ivc, y_ivc, 'k--')
plt.plot(los.x_start, los.y_start, 'ro', label='start')
plt.plot(los.x_end, los.y_end, 'bo', label='end')
plt.plot(los.x, los.y, 'g', label='los')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()
