"""Test impact parameter coordinate systems."""

from unittest.mock import Mock

from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import floats
import numpy as np
from numpy.testing import assert_allclose
from pytest import mark
from xarray import broadcast
from xarray import DataArray
from xarray import where

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
    min_los=5,
    max_los=15,
    min_num=5,
    max_num=15,
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
    min_los=5,
    max_los=15,
    min_num=5,
    max_num=15,
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

    los_coord = draw(floats(0.0, len(R_vals) - 1 if vertical else len(z_vals) - 1))
    position_coord = draw(floats(0.0, 1.0))

    def mock_Rz_to_los(R, z, t):
        """Dummy routine to pretend to convert R, z coords to LOS coords."""
        trial_data = R + z + t
        ones = np.ones_like(trial_data.data)
        los = DataArray(
            los_coord * ones, dims=trial_data.dims, coords=trial_data.coords
        )
        position = DataArray(
            position_coord * ones, dims=trial_data.dims, coords=trial_data.coords
        )
        return los, position

    los_transform.convert_from_Rz = Mock(side_effect=mock_Rz_to_los)
    flux_transform = draw(flux_coordinates(domain, min_side))
    los_transform.set_equilibrium(flux_transform.equilibrium, force=True)
    n = len(z_vals) if vertical else len(R_vals)
    return (
        ImpactParameterCoordinates(los_transform, flux_transform, n - 1),
        vertical,
        R_vals,
        z_vals,
    )


def not_near_magnetic_axis(x, xcoords, xlos, xmag):
    """Checks whether any R-z coordinates are near the magnetic axis. The
    "x" in argument names should be taken to refer to either R or z,
    depending on whether the lines of sight are vertical or horizontal.

    Parameters
    ----------
    x
        Either "R" or "z"
    xcoords
        The coordinates to check.
    xvals
        The positions of the lines of sight.
    xmag
        The position of the magneitc axis on the x coordinate.

    Returns
    -------
    :
        Element-wise True if a coordinate is not near the magnetix axis, False
        if it is.

    """
    if xlos[1] - xlos[0] > 0:
        indices = (xlos >= xmag).argmax(x)
    else:
        indices = (xlos < xmag).argmax(x)
    los_below_axis = xlos.isel({x: indices - 1})
    los_above_axis = xlos.isel({x: indices})
    return (xcoords - los_below_axis) * (xcoords - los_above_axis) >= 0


pytestmark = mark.filterwarnings(
    "ignore:(invalid value|divide by zero) encountered in true_divide"
)


@given(
    parallel_impact_parameter_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), xarray=True),
)
@settings(report_multiple_bugs=False, deadline=500)
def test_parallel_from_Rz(transform_data, coords):
    transform, vertical, Rvals, zvals = transform_data
    R, z, t = coords
    rhomin, position = transform.convert_from_Rz(*coords)
    los, position2 = transform.lines_of_sight.convert_from_Rz(*coords)
    assert np.all(position == position2)
    Rmag = transform.equilibrium.rmag.interp(t=t, method="nearest")
    zmag = transform.equilibrium.zmag.interp(t=t, method="nearest")
    # Catch any cases where round-off-error means LOS > number of
    # lines available
    # los = where(los > nlos - 1, nlos - 1, los)

    # lines_of_sight.convert_from_Rz is mocked, so need to figure out
    # what actual R, z coordinates would be for the results it
    # returns.
    new_R, new_z = transform.lines_of_sight.convert_to_Rz(los, position2, t)
    tnearest = transform.rho_min.coords["t"].sel(t=t, method="nearest")
    if vertical:
        # This transform isn't reliable near the magnetic axis, so
        # don't run the test there
        assume(np.all(not_near_magnetic_axis("R", new_R, Rvals, Rmag)))
        zval_axis = zvals[np.abs(zvals - zmag).argmin("z")]
        rho_expected, _, _ = transform.equilibrium.flux_coords(
            new_R, zval_axis, tnearest, transform.flux_surfaces.flux_kind
        )
        rho_expected = where(new_R < Rmag, -rho_expected, rho_expected)
    else:
        # This transform isn't reliable near the magnetic axis, so
        # don't run the test there
        assume(np.all(not_near_magnetic_axis("z", new_z, zvals, zmag)))
        Rval_axis = Rvals[np.abs(Rvals - Rmag).argmin("R")]
        rho_expected, _, _ = transform.equilibrium.flux_coords(
            Rval_axis, new_z, tnearest, transform.flux_surfaces.flux_kind
        )
        rho_expected = where(new_z < zmag, -rho_expected, rho_expected)
    assert_allclose(
        rho_expected, rhomin.transpose(*rho_expected.dims), rtol=1e-4, atol=1e-1
    )


@given(
    parallel_impact_parameter_coordinates(),
    arbitrary_coordinates((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), xarray=True),
)
@settings(deadline=500)
def test_parallel_to_Rz(transform_data, coords):
    transform, vertical, Rvals, zvals = transform_data
    R, z, t = coords
    Rmag = transform.equilibrium.rmag.interp(t=t, method="nearest")
    zmag = transform.equilibrium.zmag.interp(t=t, method="nearest")
    if vertical:
        # This transform isn't reliable near the magnetic axis, so
        # don't run the test there
        assume(np.all(not_near_magnetic_axis("R", R, Rvals, Rmag)))
        # Find z-value along lines of sight closest to the magnetic axis
        zval_axis = zvals[np.abs(zvals - zmag).argmin("z")]
        rho, _, _ = transform.equilibrium.flux_coords(
            R, zval_axis, t, transform.flux_surfaces.flux_kind
        )
        rho = where(R < Rmag, -rho, rho)
        positions = (z - zvals[0]) / (zvals[-1] - zvals[0])
    else:
        # This transform isn't reliable near the magnetic axis, so
        # don't run the test there
        assume(np.all(not_near_magnetic_axis("z", z, zvals, zmag)))
        # Find R-value along lines of sight closest to the magnetic axis
        Rval_axis = Rvals[np.abs(Rvals - Rmag).argmin("R")]
        rho, _, _ = transform.equilibrium.flux_coords(
            Rval_axis, z, t, transform.flux_surfaces.flux_kind
        )
        rho = where(z < zmag, -rho, rho)
        positions = (R - Rvals[0]) / (Rvals[-1] - Rvals[0])
    # Need to handle negative values based on position relative to mag-axis
    reduction_dims = [
        dim for dim in transform.rho_min.dims if dim != "t" and dim not in t.dims
    ]
    impact_min = transform.rho_min.interp(t=t, method="nearest").min(reduction_dims)
    impact_max = transform.rho_min.interp(t=t, method="nearest").max(reduction_dims)
    rho = where(rho < impact_min, impact_min, rho)
    rho = where(rho > impact_max, impact_max, rho)
    Rnew, znew = transform.convert_to_Rz(rho, positions, t)
    R, z, _ = broadcast(R, z, t)
    assert_allclose(R, Rnew.transpose(*R.dims), atol=7e-2)
    assert_allclose(z, znew.transpose(*z.dims), atol=7e-2)
