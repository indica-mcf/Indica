from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "../../")
from indica.converters.lines_of_sight_jw import LinesOfSightTransform
from xarray import DataArray

# Test line of sight transform
machine_dimensions = ((0.175, 0.8), (-0.6, 0.6))
location = np.array([0.900, -0.100, 0.0])
#direction = np.array([-1.0, -0.5, 0.0])   # under centre column
direction = np.array([-1.0, 0.1, 0.0])    # hitting centre column
#direction = np.array([-1.0, 0.5, 0.0])    # above centre column
direction = direction / np.linalg.norm(direction)

# Line
L = 2.0
x2_ = np.linspace(0.0, 1.0, 100)
xv = location[0] + L*x2_*direction[0]
yv = location[1] + L*x2_*direction[1]

# Centre column
tcc = np.linspace(0.0, 2*np.pi, 1000)
xcc = machine_dimensions[0][0] * np.cos(tcc)
ycc = machine_dimensions[0][0] * np.sin(tcc)

# Outer wall
xow = machine_dimensions[0][1] * np.cos(tcc)
yow = machine_dimensions[0][1] * np.sin(tcc)


# Define Transform object
dl = 0.002
transform = LinesOfSightTransform(location, direction, machine_dimensions=machine_dimensions, name='instrument', dl=dl)
print(transform)

X_start = transform.x_start
Y_start = transform.y_start
X_end = transform.x_end
Y_end = transform.y_end

dl = transform.dl
x2 = transform.x2

print('x2={}'.format(x2))
print('dl={}'.format(dl))

#R_start = transform.R_start
#z_start = transform.z_start
#T_start = transform.T_start
#R_end = transform.R_end
#z_end = transform.z_end
#T_end = transform.T_end
#X_start = R_start * np.cos(T_start)
#Y_start = R_start * np.sin(T_start)
#X_end = R_end * np.cos(T_end)
#Y_end = R_end * np.sin(T_end)
#print('R_end={}'.format(R_end))


plt.figure()
plt.plot(xv, yv, 'r')
plt.plot(xcc, ycc, 'k--')
plt.plot(xow, yow, 'k--')
plt.plot(X_start, Y_start, 'go')
plt.plot(X_end, Y_end, 'mo')
plt.show()


### "distance" method to find "dl"
#x2_arr = np.linspace(0, 1, 100)
#x2 = DataArray(x2_arr, dims=transform.x2_name)
#print(x2)
#print('x2[0]={}'.format(x2[0]))
#dl = transform.distance(transform.x2_name, 0, x2[0:2], 0)[1]
#print(dl)






