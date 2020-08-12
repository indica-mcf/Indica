"""Test calculations using equilibrium data."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import sampled_from
import numpy as np
from pytest import approx
from scipy.integrate import quad

from indica.equilibrium import Equilibrium
from .data_strategies import equilibrium_data
from .fake_equilibrium import FLUX_TYPES
from .strategies import arbitrary_coordinates


@composite
def equilibria(draw):
    """Generate :py:class:`indica.equilibrium.Equilibrium` objects for
    testing purposes.

    """
    # TODO: Create version of this routine that sets the offset
    return Equilibrium(draw(equilibrium_data()), sess=MagicMock())


@composite
def flux_types(draw):
    return draw(sampled_from(FLUX_TYPES))


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_lfs_rad_consistent(equilib_dat, coords, ftype):
    """Check R_lfs actually on low-flux surfaces and have appropriate flux
    surface values.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    rho, theta, time = coords
    R_lfs, t = equilib.R_lfs(rho, time, ftype)
    z_mag = equilib_dat["zmag"].interp(t=time)
    R_mag = equilib_dat["rmag"].interp(t=time)  # TODO: add R-shift
    assert np.all(rho == approx(equilib.flux_coords(R_lfs, z_mag, time)[0]))
    assert np.all(R_lfs >= R_mag)


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_hfs_rad_consistent(equilib_dat, coords, ftype):
    """Check R_hfs actually on high-flux surfaces and have appropriate
    flux surface values.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    rho, theta, time = coords
    R_hfs, t = equilib.R_hfs(rho, time, ftype)
    z_mag = equilib_dat["zmag"].interp(t=time)
    R_mag = equilib_dat["rmag"].interp(t=time)  # TODO: add R-shift
    assert np.all(rho == approx(equilib.flux_coords(R_hfs, z_mag, time)[0]))
    assert np.all(R_hfs <= R_mag)


@given(
    equilibrium_data(), flux_types(),
)
def test_lfs_rad_expected(equilib_dat, ftype):
    """Check R_lfs matches input data used to construct equilibrium object.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    expected = equilib_dat["rmjo"]
    time = np.expand_dims(expected.dims["t"], (0, 1))
    rho = equilib.convert_flux_coords(
        expected.dims["rho"], 0.0, time, "poloidal", ftype
    )
    R_lfs, t = equilib.R_lfs(rho, time, ftype)
    assert np.all(R_lfs == approx(expected.values))


@given(
    equilibrium_data(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_hfs_rad_expected(equilib_dat, coords, ftype):
    """Check R_hfs matches input data used to construct equilibrium object.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    expected = equilib_dat["rmji"]
    time = np.expand_dims(expected.dims["t"], (0, 1))
    rho = equilib.convert_flux_coords(
        expected.dims["rho"], 0.0, time, "poloidal", ftype
    )
    R_rfs, t = equilib.R_rfs(rho, time, ftype)
    assert np.all(R_rfs == approx(expected.values))


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_flux_spatial_conversion(equilib, coords, ftype):
    """Check flux and spatial coordinate conversions can invert each other.

    """
    rho, theta, time = coords
    R, z, t = equilib.spatial_coords(rho, theta, time, ftype)
    rho2, theta2, t = equilib.flux_coords(R, z, t, ftype)
    assert np.all(rho2 == approx(rho))
    assert np.all(theta2 == approx(theta))
    assert t is time


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
    flux_types(),
)
def test_flux_flux_conversion(equilib, coords, ftype1, ftype2):
    """Check conversions back and forth between different types of flux
    coordinates can invert properly.

    """
    rho, theta, time = coords
    rho_tmp, theta_tmp, t = equilib.convert_flux_coords(
        rho, theta, time, ftype1, ftype2
    )
    rho2, theta2, t = equilib.convert_flux_coords(rho_tmp, theta_tmp, t, ftype2, ftype1)
    assert np.all(rho2 == approx(rho))
    assert np.all(theta2 == approx(theta))
    assert t is time


@given(
    equilibrium_data(min_spatial=10, max_spatial=20),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_minor_radius(equilib_dat, coords, ftype):
    """Check minor radius of location matches data used to construct
    equilibrium object.

    """
    rho, theta, time = coords
    equilib = Equilibrium(equilib_dat, MagicMock())
    psi = equilib_dat["psi"]
    Rmag = equilib_dat["rmag"]
    zmag = equilib_dat["zmag"]
    faxs = equilib_dat["faxs"]
    fbnd = equilib_dat["fbnd"]
    minor_radius, _ = equilib.minor_radius(*coords, ftype)
    rho_grid = rho * np.ones_like(theta) * np.ones_like(time)
    theta_grid = np.ones_like(rho) * theta * np.ones_like(time)
    time_grid = np.ones_like(rho) * np.ones_like(theta) * time
    for min_r, rh, th, t in zip(minor_radius, rho_grid, theta_grid, time_grid):
        R = Rmag.interp(t=t) + min_r * np.cos(th)
        z = zmag.interp(t=t) + min_r * np.sin(th)
        psi_unnorm = psi.interp(R=R, z=z, t=t)
        rho_actual = np.sqrt(
            (psi_unnorm - faxs.interp(t=t)) / (fbnd.interp(t) - faxs.interp(t=t))
        )
        assert rho_actual == approx(rh, abs=1e-4, rel=1e-4)


# Check Btot and minor rad using some sort of dummy data (e.g., some factor
# times distance from magnetic axis?)


@given(
    equilibrium_data(min_spatial_points=10, max_spatial_points=20),
    arbitrary_coordinates((0.15, 0.0, 75.0), (1.0, 2 * np.pi, 80.0)),
    flux_types(),
)
def test_volume_derivative(equilib_dat, coords, ftype):
    """Check derivatives of volumes enclosed within flux surfaces match
    Jacobian in input data. Need to use a reasonably large number of
    spatial points to ensure integration is done accurately.

    """
    equilib = Equilibrium(equilib_dat, sess=MagicMock())
    vjac = equilib_dat["vjac"]
    rho, _, time = coords
    dpsi = 0.01
    psin, _, _ = equilib.convert_flux_coords(rho, 0.0, time, ftype, "poloidal") ** 2
    psin1 = psin + dpsi
    psin2 = psin - dpsi
    rho1, _, _ = equilib.convert_flux_coords(
        np.sqrt(psin1), 0.0, time, "poloidal", ftype
    )
    rho2, _, _ = equilib.convert_flux_coords(
        np.sqrt(psin2), 0.0, time, "poloidal", ftype
    )
    vol1, _ = equilib.enclosed_volume(rho1, time, ftype)
    vol2, _ = equilib.enclosed_volume(rho2, time, ftype)
    dvol = (vol2 - vol1) / (psin2 - psin1)
    rho_grid = rho * np.ones_like(time)
    time_grid = np.ones_like(rho) * time
    for dv, rh, t in zip(dvol, rho_grid, time_grid):
        assert dv == approx(vjac.interp(rho=rh, t=t), rel=1e-4, abs=1e-4)


@given(
    equilibrium_data(min_spatial_points=10, max_spatial_points=20).map(
        lambda x: Equilibrium(x, sess=MagicMock())
    ),
    floats(0.0, 1.0, exclude_min=True),
    floats(75.0, 80.0),
    flux_types(),
)
def test_volume_enclosed(equilib, rho, time, ftype):
    """Check enclosed volume is actually correct for that flux surface by
    integrating area within it. It is assumed that the data from which
    the object is constructed is valid.

    """
    Rmag, R_err = quad(
        lambda th: equilib.spatial_coords(rho, th, time, ftype)[0], 0.0, 2 * np.pi
    )
    Rmag /= 2 * np.pi
    area, a_err = quad(
        lambda th: equilib.minor_radius(rho, th, time, ftype), 0.0, 2 * np.pi
    )
    expected = 2 * np.pi * Rmag * area
    actual, _ = equilib.enclosed_volume(rho, time)
    tol = 2 * np.pi * np.sqrt(Rmag ** 2 * a_err ** 2 + area ** 2 * R_err ** 2) * 1e4
    assert actual == approx(expected, rel=tol, abs=tol)


# Test use of offset picker

# Check provenance is correct
