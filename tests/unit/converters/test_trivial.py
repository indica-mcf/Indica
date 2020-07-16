"""Test the trivial coordinate transform."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
import numpy as np

from src.converters import TrivialTransform
from ..strategies import arbitrary_coordinates


@composite
def trivial_transforms(draw, min_side=1, min_dims=0):
    x1, x2, t = draw(arbitrary_coordinates(min_side=min_side, min_dims=min_dims))
    bshape = t.shape if isinstance(t, np.ndarray) else ()
    R, z, t = draw(
        arbitrary_coordinates(base_shape=bshape, min_side=min_side, min_dims=min_dims)
    )
    result = TrivialTransform(x1, x2, R, z, t)
    result.set_equilibrium(MagicMock())
    return result


@given(trivial_transforms(), arbitrary_coordinates())
def test_trivial_to_Rz(transform, coords):
    """Test trivial transform returns arguments."""
    R, z, t = transform.convert_to_Rz(*coords)
    assert R is coords[0]
    assert z is coords[1]
    assert t is coords[2]


@given(trivial_transforms(), arbitrary_coordinates())
def test_trivial_from_Rz(transform, coords):
    """Test trivial transform returns arguments."""
    x1, x2, t = transform.convert_from_Rz(*coords)
    assert x1 is coords[0]
    assert x2 is coords[1]
    assert t is coords[2]
