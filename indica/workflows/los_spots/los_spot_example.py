import numpy as np
import sys

# Add Indica to python path
path_to_indica = '/home/jonathan.wood/git_home/Indica/'
sys.path.append(path_to_indica)

# Import Indica things
from indica.converters import LineOfSightTransform


# Dummy line-of-sight
origin_x = np.array([1.0], dtype=float)
origin_y = np.array([0.0], dtype=float)
origin_z = np.array([0.0], dtype=float)
direction_x = np.array([-1.0], dtype=float)
direction_y = np.array([0.0], dtype=float)
direction_z = np.array([0.0], dtype=float)
name = 'wowzers'

beamlets = int(5 * 5)
spot_width = 0.01
spot_height = 0.01
spot_shape = 'round'
div_w = 0.0 * 10 * 1e-3  # radians
div_h = 0.0 * 10 * 1e-3  # radians

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
)

print(los.spot_height)
print(los.spot_width)
print(los.spot_shape)
print(los.beamlets)

los.distribute_beamlets()

