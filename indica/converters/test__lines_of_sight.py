import sys
sys.path.append("../..")
from indica.converters import lines_of_sight

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
