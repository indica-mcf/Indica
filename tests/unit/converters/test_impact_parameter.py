"""Test impact parameter coordinate systems."""

from hypothesis import given
from hypothesis.strategies import composite
import numpy as np
from xarray.testing import assert_allclose

from indica.converters import ImpactParameterCoordinates
from .test_flux_surfaces import flux_coordinates
from .test_lines_of_sight import los_coordinates
from .test_lines_of_sight import parallel_los_coordinates
from ..strategies import arbitrary_coordinates


@composite
def impact_parameter_coordinates(
    draw,
    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    min_side=2,
    min_los=2,
    max_los=10,
    min_num=2,
    max_num=10,
    default_Rz=True,
):
    """Generates :py:class:`indica.converters.ImpactParameterCoordinates`
    objects with lines of sight radiation from a point. At present
    this point is on the edge of hte Tokamak, for reasons of
    convenience.

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        A region in the native coordinate system over which the transform is
        guarnateed to return non-NaN results. Takes form
        ``((x1_start, x1_stop), (x2_start, x2_stop), (t_start, t_stop))``.
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    min_num: int
        The minimum number of intervals in which to divide the lines of sight
    max_num: int
        The maximum number of intervals in which to divide the lines of sight

    Returns
    -------
    : ImpactParameterCoordinates
        The coordinate transform object.

    """
    los_transform = los_coordinates(
        domain[0:2], min_los, max_los, min_num, max_num, default_Rz
    )
    flux_transform = flux_coordinates(domain, min_side)
    los_transform.set_equilibrium(flux_transform.equilibrium, force=True)
    return ImpactParameterCoordinates(los_transform, flux_transform)


@composite
def parallel_impact_parameter_coordinates(
    draw,
    domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
    min_side=2,
    min_los=2,
    max_los=10,
    min_num=2,
    max_num=10,
):
    """Generates :py:class:`indica.converters.ImpactParameterCoordinates`
    objects with lines of sight radiation from a point. At present
    this point is on the edge of hte Tokamak, for reasons of
    convenience.

    Parameters
    ----------
    domain : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        A region in the native coordinate system over which the transform is
        guarnateed to return non-NaN results. Takes form
        ``((x1_start, x1_stop), (x2_start, x2_stop), (t_start, t_stop))``.
    min_side : integer
        The minimum number of elements in an unaligned dimension for the
        default coordinate arrays. (Not available for all coordinate systems.)
    min_los: int
        The minimum number of lines of sight
    max_los: int
        The maximum number of lines of sight
    min_num: int
        The minimum number of intervals in which to divide the lines of sight
    max_num: int
        The maximum number of intervals in which to divide the lines of sight

    Returns
    -------
    transform: TransectCoordinates
        The coordinate transform object.
    vertical: bool
        Whether the lines of sight are vertical (True) or horizontal (False)
    R_vals: ndarray
        The major radius positions of datapoints on the line of sight grid.
    z_vals: ndarray
        The vertical positions of datapoints on the line of sight grid.

    """
    los_transform, vertical, R_vals, z_vals = draw(
        parallel_los_coordinates(domain[0:2], min_los, max_los, min_num, max_num)
    )
    flux_transform = draw(flux_coordinates(domain, min_side))
    los_transform.set_equilibrium(flux_transform.equilibrium, force=True)
    return (
        ImpactParameterCoordinates(los_transform, flux_transform),
        vertical,
        R_vals,
        z_vals,
    )


@given(
    parallel_impact_parameter_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), xarray=True),
)
def test_parallel_from_Rz(transform_data, coords):
    transform, vertical, _, _ = transform_data
    R, z, t = coords
    rhomin, position, tnew = transform.convert_from_Rz(*coords)
    los, position2, _ = transform.lines_of_sight.convert_from_Rz(*coords)
    assert np.all(position == position2)
    if vertical:
        rho_expected = transform.equilibrium.flux_coords(
            R, transform.equilibrium.zmag, t, transform.flux_surfaces.flux_kind
        )
    else:
        rho_expected = transform.equilibrium.flux_coords(
            transform.equilibrium.Rmag, z, t, transform.flux_surfaces.flux_kind
        )
    assert_allclose(rho_expected, rhomin)
    assert t is tnew


@given(
    parallel_impact_parameter_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), xarray=True),
)
def test_parallel_to_Rz(transform_data, coords):
    transform, vertical, Rvals, zvals = transform_data
    R, z, t = coords
    if vertical:
        rho = transform.equilibrium.flux_coords(
            R, transform.equilibrium.zmag, t, transform.flux_surfaces.flux_kind
        )
        positions = (z - zvals[0]) / (zvals[-1] - zvals[0])
    else:
        rho = transform.equilibrium.flux_coords(
            transform.equilibrium.Rmag, z, t, transform.flux_surfaces.flux_kind
        )
        positions = (R - Rvals[0]) / (Rvals[-1] - Rvals[0])
    Rnew, znew, tnew = transform.convert_to_Rz(rho, positions, t)
    assert_allclose(R, Rnew)
    assert_allclose(z, znew)
    assert t is tnew
