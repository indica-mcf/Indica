"""Test the trivial coordinate transform."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite

from indica.converters import TrivialTransform
from ..strategies import arbitrary_coordinates


@composite
def trivial_transforms(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
    result = TrivialTransform()
    result.set_equilibrium(MagicMock())
    return result


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
