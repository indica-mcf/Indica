"""Tests that apply to all subclasses."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import tuples
import numpy as np
from pytest import approx
from pytest import mark

from indica.converters import CoordinateTransform
from .test_flux_surfaces import flux_coordinates
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
    :py:class:`indica.converters.CoordinateTransform` objects. They
    should already have had an equilibrium object set.

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
        one_of(
            [
                trivial_transforms(domain, min_side=min_side),
                transect_coordinates(domain),
                magnetic_coordinates(domain),
                flux_coordinates(domain, min_side=min_side),
                #                los_coordinates(domain),
            ]
        )
    )


@composite
def equilibria(draw):
    """Generates equilibrium objects. At present it just returns a mock."""
    return draw(just(MagicMock()))


@mark.skip(reason="Difficult to test for this in a general way")
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


@given(
    coordinate_transforms(),
    arbitrary_coordinates(min_value=(0.0, 0.0, 0.0), max_value=(1.0, 1.0, 1.0)),
)
def test_inverse_transforms(transform, coords):
    """Test convert from/to methods are inverses"""
    x1, x2, t = coords
    x1new, x2new, tnew = transform.convert_to_Rz(*transform.convert_from_Rz(x1, x2, t))
    assert np.all(np.isclose(x1new, x1, 1e-4, 1e-7))
    assert np.all(np.isclose(x2new, x2, 1e-4, 1e-7))
    assert np.all(np.isclose(tnew, t, 1e-4, 1e-7))


@mark.xfail
@given(coordinate_transforms())
def test_transforms_encoding(transform):
    """Test encode/decode methods are inverses"""
    encoding = transform.encode()
    transform2 = CoordinateTransform.decode(encoding)
    assert transform2.encode() == encoding


@given(
    coordinate_transforms(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), min_side=2, min_dims=3),
)
def test_transform_reverse_direction(transform, coords):
    """Test reversing direction of input to distance method"""
    x1, x2, t = coords
    d = transform.distance(1, x1, x2, t)
    d_reversed = transform.distance(1, x1[:, ::-1, :], x2[:, ::-1, :], t)
    assert d_reversed[:, -1, :] == approx(d[:, -1, :])
    d = transform.distance(2, x1, x2, t)
    d_reversed = transform.distance(2, x1[..., ::-1], x2[..., ::-1], t)
    assert d_reversed[..., -1] == approx(d[..., -1])


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
    domains().flatmap(
        lambda d: tuples(
            just(d), lists(coordinate_transforms(domain=d), min_size=3, max_size=10)
        )
    ),
    arbitrary_coordinates(min_value=(0.0, 0.0, 0.0), max_value=(1.0, 1.0, 1.0)),
)
def test_transforms_independent(domain_transforms, normalised_coords):
    """Test irrelevance of intermediate transforms"""
    domain, transforms = domain_transforms
    Rz_coords = (
        co * (dom[1] - dom[0]) + dom[0] for co, dom in zip(normalised_coords, domain)
    )
    coords = transforms[0].convert_from_Rz(*Rz_coords)
    expected = transforms[0].convert_to(transforms[-1], *coords)
    for transform1, transform2 in zip(transforms, transforms[1:]):
        coords = transform1.convert_to(transform2, *coords)
    assert np.all(coords[0] == approx(expected[0], abs=1e-5, rel=1e-5))
    assert np.all(coords[1] == approx(expected[1], abs=1e-5, rel=1e-5))
    assert coords[2] is expected[2]
