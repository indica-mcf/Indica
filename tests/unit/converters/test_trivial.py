"""Test the trivial coordinate transform."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite

from indica.converters import TrivialTransform
from indica.utilities import coord_array
from ..strategies import arbitrary_coordinates
from ..strategies import basis_coordinates


@composite
def trivial_transforms(draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=1):
    # TODO: update basis_coords to support min_side and min_dims
    x1, x2, t = draw(basis_coordinates(min_side=min_side))
    R, z, t = draw(
        basis_coordinates(
            tuple(dim[0] for dim in domain), tuple(dim[1] for dim in domain), min_side
        )
    )
    result = TrivialTransform(
        coord_array(x1.flat, "R"),
        coord_array(x2.flat, "z"),
        coord_array(R.flat, "R"),
        coord_array(z.flat, "z"),
        coord_array(t.flat, "t"),
    )
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
