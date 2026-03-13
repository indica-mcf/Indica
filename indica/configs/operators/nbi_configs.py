"""Default NBI transform configuration placeholders."""

from copy import deepcopy

import numpy as np

DEFAULT_NBI_TRANSFORM_CONFIG = {
    # Source mapping: these defaults are placeholders for NBI database fields.
    # See fields such as LOCATION, DIRECTION, GRID_WIDTH/HEIGHT, GRID_SHAPE, etc.

    # 1) Already supported by LineOfSightTransform
    "origin_x": np.array([0.339], dtype=float),  # DB: LOCATION[..., x] [m]
    "origin_y": np.array([0.375], dtype=float),  # DB: LOCATION[..., y] [m]
    "origin_z": np.array([0.0], dtype=float),  # DB: LOCATION[..., z] [m]
    "direction_x": np.array([-0.779], dtype=float),  # DB: DIRECTION[..., xhat]
    "direction_y": np.array([0.0], dtype=float),  # DB: DIRECTION[..., yhat]
    "direction_z": np.array([0.0], dtype=float),  # DB: DIRECTION[..., zhat]
    "name": "hnbi",  # DB: LABEL / beam identifier
    "machine_dimensions": ((1.83, 3.9), (-1.75, 2.0)),
    "dl": 0.01,
    "passes": 1,
    "beamlets_method": "simple",
    "n_beamlets": 1,
    "spot_width": 0.01,  # DB: GRID_WIDTH [m]
    "spot_height": 0.01,  # DB: GRID_HEIGHT [m]
    "spot_shape": "round",  # DB: GRID_SHAPE (CIRCULAR/RECTANGULAR)
    "focal_length": 1.0,  # Marco: where does this come from?
    "plot_beamlets": False,

    # 3) Not in LineOfSightTransform, but shared NBI beam settings used
    #    by FIDASIM and other NBI-related paths.
    "divy": np.array([0.01, 0.01, 0.01], dtype=float),  # DB: DIVERGENCE_H [rad]
    "divz": np.array([0.01, 0.01, 0.01], dtype=float),  # DB: DIVERGENCE_V [rad]
}

# Beam-schema defaults used across NBI beam models when geometry metadata
# is not explicitly provided on the transform.
NBI_BEAM_SCHEMA_DEFAULTS = {
    "data_source": "TODO: fill source",  # DB: COMMENT / provenance text
    "focy": 0.355,  # DB: FOCUS_H [m]
    "focz": 0.395,  # DB: FOCUS_V [m]
    "naperture": 0,
}


def get_default_nbi_transform_config() -> dict:
    """Return a copy of the default NBI transform config."""
    return deepcopy(DEFAULT_NBI_TRANSFORM_CONFIG)


def get_default_nbi_beam_schema_defaults() -> dict:
    """Return a copy of the default NBI beam schema defaults."""
    return deepcopy(NBI_BEAM_SCHEMA_DEFAULTS)
