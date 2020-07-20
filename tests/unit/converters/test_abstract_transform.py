"""Tests that apply to all subclasses."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
import numpy as np
from pytest import approx
from pytest import raises

from src.converters import CoordinateTransform
from src.converters import EquilibriumException
from src.converters import MagneticCoordinates
from src.converters import TransectCoordinates
from .test_flux_surfaces import flux_coordinates
from .test_lines_of_sight import los_coordinates
from .test_magnetic import magnetic_coordinates
from .test_transect import transect_coordinates
from .test_trivial import trivial_transforms
from ..strategies import arbitrary_coordinates
from ..strategies import domains


@composite
def coordinate_transforms(
    draw, domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), min_side=1, min_dims=0
):
    """Strategy for generating abritrary
    :py:class:`src.converters.CoordinateTransform` objects. They should already
    have had an equilibrium object set.

    Reduces towards simpler coordinate systems.

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        A region in the native coordinate system over which the transform is
        guarnateed to return non-NaN results. Takes form
        ``((x1_start, x1_stop), (x2_start, x2_stop), (t_start, t_stop))``.
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_dims : integer
        The minimum number of dimensions for the default coordinate arrays.
        (Not available for all coordinate systems.)

    """
    return draw(
        sampled_from(
            trivial_transforms(min_side=min_side, min_dims=min_dims),
            transect_coordinates(),
            magnetic_coordinates(domain),
            los_coordinates(domain),
            flux_coordinates(min_side=min_side, min_dims=min_dims),
        )
    )


@composite
def equilibria(draw):
    """Generates equilibrium objects. At present it just returns a mock."""
    return draw(just(MagicMock()))


@given(coordinate_transforms(min_side=2, min_dims=2), integers(1, 1000))
def test_transform_defaults(transform, attempts):
    """Test calling with default arguments always returns same objects"""
    expected_R, expected_z, expected_t1 = transform.convert_to_Rz()
    expected_x1, expected_x2, expected_t2 = transform.convert_from_Rz()
    expected_d1, expected_t3 = transform.distance(1)
    expected_d2, expected_t4 = transform.distance(2)
    for i in range(attempts):
        R, z, t = transform.convert_to_Rz()
        assert R is expected_R
        assert z is expected_z
        assert t is expected_t1
        x1, x2, t = transform.convert_from_Rz()
        assert x1 is expected_x1
        assert x2 is expected_x2
        assert t is expected_t2
        d1, t = transform.distance(1)
        assert d1 is expected_d1
        assert t is expected_t3
        # Transect coordinates always have 1-D default values, so can't
        # get distance in the second spatial dimension
        if not isinstance(transform, (TransectCoordinates, MagneticCoordinates)):
            d2, t = transform.distance(2)
            assert d2 is expected_d2
            assert t is expected_t4


@given(coordinate_transforms(), arbitrary_coordinates())
def test_transform_broadcasting(transform, coords):
    """Test rank/shape of output"""
    expected = np.broadcast(*coords).shape
    R, z, t = transform.convert_to_Rz(*coords)
    assert R.shape == expected
    assert z.shape == expected
    assert t.shape == expected
    x1, x2, t = transform.convert_from_Rz(*coords)
    assert x1.shape == expected
    assert x2.shape == expected
    assert t.shape == expected
    if isinstance(coords[0], np.ndarray) and coords[0].shape[0] > 1:
        d, t = transform.distance(1, *coords)
        assert d.shape == expected
        assert t.shape == expected
    if (
        isinstance(coords[1], np.ndarray)
        and coords[1].ndim > 1
        and coords[1].shape[1] > 1
    ):
        d, t = transform.distance(2, *coords)
        assert d.shape == expected
        assert t.shape == expected


@given(coordinate_transforms(), arbitrary_coordinates())
def test_inverse_transforms(transform, coords):
    """Test convert from/to methods are inverses"""
    x1, x2, t = coords
    x1new, x2new, tnew = transform.convert_from_Rz(*transform.convert_to_Rz(x1, x2, t))
    assert np.all(x1new == approx(x1))
    assert np.all(x2new == approx(x2))
    assert np.all(tnew == approx(t))


@given(coordinate_transforms())
def test_transforms_encoding(transform):
    """Test encode/decode methods are inverses"""
    encoding = transform.encode()
    transform2 = CoordinateTransform.decode(encoding)
    assert transform2.encode() == encoding


@given(coordinate_transforms(), arbitrary_coordinates(min_side=2, min_dims=2))
def test_transform_reverse_direction(transform, coords):
    """Test reversing direction of input to distance method"""
    x1, x2, t = coords
    d = transform.distance(1, x1, x2, t)
    d_reversed = transform.distance(1, x1[::-1, ...], x2[::-1, ...], t)
    assert d_reversed[:-1, ...] == approx(d[:-1, ...])
    d = transform.distance(2, x1, x2, t)
    d_reversed = transform.distance(2, x1[:, ::-1, :], x2[:, ::-1, :], t)
    assert d_reversed[:, :-1, :] == approx(d[:, :-1, :])


@given(coordinate_transforms(), arbitrary_coordinates(min_side=2, min_dims=2))
def test_transform_distance_increasing(transform, coords):
    """Test distances monotonic increasing"""
    d, t = transform.distance(1, *coords)
    assert np.all(d[0, ...] == 0.0)
    assert np.all(d[:-1, ...] < d[1:, ...])
    d, t = transform.distance(2, *coords)
    assert np.all(d[:, 0, ...] == 0.0)
    assert np.all(d[:, :-1, ...] < d[:, 1:, ...])


@given(
    domains().flatmap(lambda d: lists(coordinate_transforms(domain=d), min_size=3)),
    arbitrary_coordinates(),
)
def test_transforms_independent(transforms, coords):
    """Test irrelevance of intermediate transforms"""
    # TODO: Need some way to ensure that
    #       (1) coords are within the range the first transform can convert to R-z
    #       (2) we avoid the singularity in polar coordinates
    expected = transforms[0].convert_to(transforms[-1], *coords)
    for transform1, transform2 in zip(transforms, transforms[1:]):
        coords = transform1.convert_to(transform2, *coords)
    assert np.all(coords[0] == approx(expected[0]))
    assert np.all(coords[1] == approx(expected[1]))
    assert coords[2] is expected[2]


@given(coordinate_transforms(min_side=2, min_dims=2), equilibria())
def test_transform_change_equilibrium(transform, equilibrium):
    """Test setting a new equilibrium is handled properly"""
    expected_R, expected_z, expected_t1 = transform.convert_to_Rz()
    expected_x1, expected_x2, expected_t2 = transform.convert_from_Rz()
    expected_d1, expected_t3 = transform.distance(1)
    expected_d2, expected_t4 = transform.distance(2)
    with raises(EquilibriumException):
        transform.set_equilibrium(equilibrium)
    transform.set_equilibrium(equilibrium, force=True)
    transform.set_equilibrium(equilibrium)
    R, z, t = transform.convert_to_Rz()
    assert R is not expected_R
    assert z is not expected_z
    assert t is not expected_t1
    x1, x2, t = transform.convert_from_Rz()
    assert x1 is not expected_x1
    assert x2 is not expected_x2
    assert t is not expected_t2
    d1, t = transform.distance(1)
    assert d1 is not expected_d1
    assert t is not expected_t3
    # Transect coordinates always have 1-D default values, so can't
    # get distance in the second spatial dimension
    if not isinstance(transform, (TransectCoordinates, MagneticCoordinates)):
        d2, t = transform.distance(2)
        assert d2 is not expected_d2
        assert t is not expected_t4
