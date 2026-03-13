"""Default NBI transform configuration placeholders."""

from copy import deepcopy

import numpy as np

# One default dictionary carrying both transform constructor values and
# placeholder beam geometry values needed by the current NBI/FIDASIM flow.
# Transform geometry units are SI (meters), while FIDASIM beam values currently
# follow fidasim_utils conventions (centimeters and radians).
DEFAULT_NBI_TRANSFORM_CONFIG = {
    "origin_x": np.array([0.0], dtype=float),
    "origin_y": np.array([0.0], dtype=float),
    "origin_z": np.array([0.0], dtype=float),
    "direction_x": np.array([1.0], dtype=float),
    "direction_y": np.array([0.0], dtype=float),
    "direction_z": np.array([0.0], dtype=float),
    # Shared name used by transform and beam configuration.
    "name": "hnbi",
    "machine_dimensions": ((1.83, 3.9), (-1.75, 2.0)),
    "dl": 0.01,
    "passes": 1,
    "beamlets_method": "simple",
    "n_beamlets": 1,
    "spot_width": 0.01,
    "spot_height": 0.01,
    "spot_shape": "round",
    "focal_length": 1.0,
    "plot_beamlets": False,
    "shape": 2,
    "data_source": "TODO: fill source",
    "src": np.array([0.0, 0.0, 0.0], dtype=float),
    "axis": np.array([1.0, 0.0, 0.0], dtype=float),
    "widy": 1.0,
    "widz": 1.0,
    "divy": np.array([0.01, 0.01, 0.01], dtype=float),
    "divz": np.array([0.01, 0.01, 0.01], dtype=float),
    "focy": 100.0,
    "focz": 100.0,
    "naperture": 0,
}


def get_default_nbi_transform_config() -> dict:
    """Return a copy of the merged default NBI transform/beam config."""
    return deepcopy(DEFAULT_NBI_TRANSFORM_CONFIG)
