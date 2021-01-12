"""Test the trivial coordinate transform."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
import numpy as np

from indica.converters import TrivialTransform
from indica.utilities import coord_array
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates


@composite
def trivial_transforms(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    result = TrivialTransform()
    result.set_equilibrium(MagicMock())
    return result


@composite
def trivial_transforms_and_axes(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=2
):
    transform = draw(trivial_transforms(domain))
    min_vals, max_vals = zip(*domain)
    x1, x2, t = map(np.ravel, draw(basis_coordinates(min_vals, max_vals, min_side)))
    return (
        transform,
        coord_array(x1, transform.x1_name),
        coord_array(x2, transform.x2_name),
        coord_array(t, "t"),
    )


@given(trivial_transforms(), arbitrary_coordinates())
def test_trivial_to_Rz(transform, coords):
    """Test trivial transform returns arguments."""
    R, z = transform.convert_to_Rz(*coords)
    assert R is coords[0]
    assert z is coords[1]


@given(trivial_transforms(), arbitrary_coordinates())
def test_trivial_from_Rz(transform, coords):
    """Test trivial transform returns arguments."""
    x1, x2 = transform.convert_from_Rz(*coords)
    assert x1 is coords[0]
    assert x2 is coords[1]
