"""Test calculations using equilibrium data."""

from unittest.mock import MagicMock

from hypothesis import given
from hypothesis import settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import tuples
import numpy as np
from pytest import approx
from scipy.integrate import quad
from xarray import broadcast
from xarray import DataArray
from xarray import where

from indica.converters import TransectCoordinates
from indica.equilibrium import Equilibrium
from .data_strategies import data_arrays_from_coords
from .data_strategies import equilibrium_data
from .fake_equilibrium import flux_types
from .strategies import arbitrary_coordinates


@composite
def offset_pickers(draw, min_value=0.0, max_value=0.04):
    """Generate a mock offset-picker object which will return some fixed value."""
    return MagicMock(return_value=(draw(floats(min_value, max_value)), True))


@composite
def electron_temperatures(
    draw,
    rho,
    zmag,
    machine_dimensions=((1.83, 3.9), (-1.75, 2.0)),
    min_spatial_points=3,
    max_spatial_points=10,
    min_time_points=3,
    max_time_points=10,
    start_time=75.0,
    end_time=80.0,
    offset=floats(0.0, 0.04),
):
    """Generate electron temperature data suitable for calibrating the
    equilibrium."""
    zmin = float(zmag.min())
    zmax = float(zmag.max())
    nspace = draw(integers(min_spatial_points, max_spatial_points))
    R_vals = np.linspace(*machine_dimensions[0], nspace)
    zstart = draw(floats(machine_dimensions[1][0], zmin))
    zend = draw(floats(zmax, machine_dimensions[1][1]))
    if draw(booleans()):
        zstart, zend = zend, zstart
    z_vals = np.linspace(zstart, zend, nspace)
    transform = TransectCoordinates(R_vals, z_vals)
    ntime = draw(integers(min_time_points, max_time_points))
    times = np.linspace(start_time, end_time, ntime)
    times_scaled = np.linspace(0.0, 1.0, ntime)
    R_array = DataArray(R_vals, dims="index")
    z_array = DataArray(z_vals, dims="index")
    rhos = rho.interp(R=R_array, z=z_array)
    indices = rhos.indica.invert_root(
        1.0, "index", nspace - 1, method="cubic"
    ).assign_coords(t=(rhos.t - start_time) / (end_time - start_time))
    m = DataArray(
        draw(arrays(float, ntime, elements=floats(-1e4, -10.0), fill=just(-1e3))),
        coords=[("t", times_scaled)],
    )
    b = 1e2 - m * indices.interp(t=times_scaled, method="nearest") / nspace
    return draw(
        data_arrays_from_coords(
            ("temperature", "electrons"),
            transform,
            just(
                lambda x1, x2, t: np.reshape(m.interp(t=np.ravel(t)).data, t.shape) * x1
                + np.reshape(b.interp(t=np.ravel(t)).data, t.shape)
            ),
            [None, None, np.expand_dims(times, 1)],
            rel_sigma=0.001,
        )
    )


@composite
def equilibrium_data_and_Te(
    draw,
    machine_dims=((1.83, 3.9), (-1.75, 2.0)),
    min_spatial_points=5,
    max_spatial_points=15,
    min_time_points=2,
    max_time_points=8,
    start_time=75.0,
    end_time=80.0,
    Btot_factor=None,
):
    data = draw(
        equilibrium_data(
            machine_dims,
            min_spatial_points,
            max_spatial_points,
            min_time_points,
            max_time_points,
            start_time,
            end_time,
            Btot_factor,
        )
    )
    if draw(booleans()):
        rho = np.sqrt((data["psi"] - data["faxs"]) / (data["fbnd"] - data["faxs"]))
        Te = draw(
            electron_temperatures(
                rho,
                data["zmag"],
                min_spatial_points=min_spatial_points,
                max_spatial_points=max_spatial_points,
            )
        )
    else:
        Te = None
    return data, Te


@composite
def equilibria(
    draw,
    min_offset=0.0,
    max_offset=0.04,
    min_spatial_points=5,
    max_spatial_points=15,
    min_time_points=2,
    max_time_points=8,
    Btot_factor=None,
):
    """Generate :py:class:`indica.equilibrium.Equilibrium` objects for
    testing purposes.

    """
    data, Te = draw(
        equilibrium_data_and_Te(
            ((1.83, 3.9), (-1.75, 2.0)),
            min_spatial_points,
            max_spatial_points,
            min_time_points,
            max_time_points,
            Btot_factor=Btot_factor,
        )
    )
    return Equilibrium(
        data,
        Te,
        MagicMock(),
        draw(offset_pickers(min_offset, max_offset)),
    )


@settings(report_multiple_bugs=False)
@given(
    equilibria(),
    arbitrary_coordinates(
        (0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), max_side=5, xarray=True
    ),
    flux_types(),
)
def test_lfs_rad_consistent(equilib, coords, ftype):
    """Check R_lfs actually on low-flux surfaces and have appropriate flux
    surface values.

    """
    rho, theta, time = coords
    R_lfs, t = equilib.R_lfs(rho, time, ftype)
    R_mag, z_mag, _ = equilib.spatial_coords(0.0, 0.0, time)
    assert np.all(
        np.isclose(
            rho, equilib.flux_coords(R_lfs, z_mag, time)[0], rtol=1e-6, atol=1e-1
        )
    )
    assert np.all(np.abs(R_lfs - R_mag) <= 1e-1)


@given(
    equilibria(),
    arbitrary_coordinates(
        (0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), max_side=5, xarray=True
    ),
    flux_types(),
)
def test_hfs_rad_consistent(equilib, coords, ftype):
    """Check R_hfs actually on high-flux surfaces and have appropriate
    flux surface values.

    """
    rho, theta, time = coords
    R_hfs, t = equilib.R_hfs(rho, time, ftype)
    R_mag, z_mag, _ = equilib.spatial_coords(0.0, 0.0, time)
    assert np.all(rho == approx(equilib.flux_coords(R_hfs, z_mag, time)[0]))
    assert np.all(R_hfs <= R_mag)


@settings(deadline=500)
@given(
    equilibrium_data_and_Te(max_spatial_points=10, max_time_points=5),
    flux_types(),
    offset_pickers(),
    booleans(),
)
def test_lfs_rad_expected(equilib_Te, ftype, offset, use_explicit_t):
    """Check R_lfs matches input data used to construct equilibrium object."""
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
    expected = equilib_dat["rmjo"] + (offset.return_value[0] if Te is not None else 0.0)
    t = expected.coords["t"] if use_explicit_t else None
    rho, _ = equilib.convert_flux_coords(
        expected.coords["rho_poloidal"], t, "poloidal", ftype
    )
    R_lfs, t = equilib.R_lfs(rho, t, ftype)
    assert R_lfs.values == approx(expected.values)


@settings(deadline=500)
@given(
    equilibrium_data_and_Te(),
    flux_types(),
    offset_pickers(),
    booleans(),
)
def test_hfs_rad_expected(equilib_Te, ftype, offset, use_explicit_t):
    """Check R_hfs matches input data used to construct equilibrium object."""
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
    expected = equilib_dat["rmji"] + (offset.return_value[0] if Te is not None else 0.0)
    t = expected.coords["t"] if use_explicit_t else None
    rho, _ = equilib.convert_flux_coords(
        expected.coords["rho_poloidal"], t, "poloidal", ftype
    )
    R_hfs, t = equilib.R_hfs(rho, t, ftype)
    assert R_hfs == approx(expected.values)


@settings(report_multiple_bugs=False, deadline=None)
@given(
    equilibria(),
    arbitrary_coordinates(
        (0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), max_side=5, xarray=True
    ),
    flux_types(),
)
def test_flux_spatial_conversion(equilib, coords, ftype):
    """Check flux and spatial coordinate conversions can invert each other."""
    rho, theta, time = coords
    R, z, t = equilib.spatial_coords(rho, theta, time, ftype)
    rho2, theta2, t = equilib.flux_coords(R, z, t, ftype)
    assert np.allclose(rho2, rho, atol=1e-1, rtol=1e-6)
    assert np.all(
        np.where(
            np.isclose(rho, 0.0, rtol=1e-1, atol=1e-1),
            True,
            np.isclose(theta2, theta, atol=1e-1, rtol=1e-6),
        )
    )
    assert t is time


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), xarray=True),
    flux_types(),
    flux_types(),
)
def test_flux_flux_conversion(equilib, coords, ftype1, ftype2):
    """Check conversions back and forth between different types of flux
    coordinates can invert properly.

    """
    rho, theta, time = coords
    rho_tmp, t = equilib.convert_flux_coords(rho, time, ftype1, ftype2)
    rho2, t = equilib.convert_flux_coords(rho_tmp, t, ftype2, ftype1)
    assert np.allclose(rho2, rho, rtol=1e-6, atol=1e-12)
    assert t is time


@settings(report_multiple_bugs=False, deadline=None)
@given(
    equilibrium_data_and_Te(min_spatial_points=10, max_spatial_points=20),
    arbitrary_coordinates(
        (0.0, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), max_side=5, xarray=True
    ),
    flux_types(),
    offset_pickers(),
)
def test_minor_radius(equilib_Te, coords, ftype, offset):
    """Check minor radius of location matches data used to construct
    equilibrium object.

    """
    equilib_dat, Te = equilib_Te
    rho, theta, time = coords
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
    psi = equilib_dat["psi"]
    Rmag = equilib_dat["rmag"] + offset.return_value[0]
    zmag = equilib_dat["zmag"]
    faxs = equilib_dat["faxs"]
    fbnd = equilib_dat["fbnd"]
    minor_radius, _ = equilib.minor_radius(*coords, ftype)
    rho_grid, theta_grid, time_grid = np.broadcast_arrays(rho, theta, time)
    R = Rmag.interp(t=time, method="nearest") + minor_radius * np.cos(theta)
    z = zmag.interp(t=time, method="nearest") + minor_radius * np.sin(theta)
    delta_psi_unnorm = (
        (psi - faxs)
        .interp(t=time, method="nearest")
        .indica.interp2d(
            R=R,
            z=z,
            zero_coords={
                "R": Rmag.interp(t=time, method="nearest"),
                "z": zmag.interp(t=time, method="nearest"),
            },
            method="cubic",
        )
    )
    delta_psi_unnorm = where(
        np.logical_and(delta_psi_unnorm < 0.0, delta_psi_unnorm > -1e-2),
        0.0,
        delta_psi_unnorm,
    )
    rho_actual = np.sqrt(
        delta_psi_unnorm
        / (
            fbnd.interp(t=time, method="nearest")
            - faxs.interp(t=time, method="nearest")
        )
    )
    rho_actual, rho = broadcast(rho_actual, rho)
    np.testing.assert_allclose(rho_actual, rho, atol=1e-1, rtol=1e-1)


# Check Btot and minor rad using some sort of dummy data (e.g., some factor
# times distance from magnetic axis?)


@given(
    equilibrium_data_and_Te(min_spatial_points=10, max_spatial_points=20),
    arbitrary_coordinates((0.15, 0.0, 75.0), (1.0, 2 * np.pi, 80.0), xarray=True),
    flux_types(),
    offset_pickers(),
)
def test_volume_derivative(equilib_Te, coords, ftype, offset):
    """Check derivatives of volumes enclosed within flux surfaces match
    Jacobian in input data. Need to use a reasonably large number of
    spatial points to ensure integration is done accurately.

    """
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
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
    equilibrium_data_and_Te(min_spatial_points=10, max_spatial_points=20),
    floats(0.0, 1.0, exclude_min=True),
    floats(75.0, 80.0),
    flux_types(),
    offset_pickers(),
)
def test_volume_enclosed(equilib_Te, rho, time, ftype, offset):
    """Check enclosed volume is actually correct for that flux surface by
    integrating area within it. It is assumed that the data from which
    the object is constructed is valid.

    """
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
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


@given(
    equilibria(),
    arbitrary_coordinates((0.0, 75.0), (1.0, 80.0), max_side=5, xarray=True),
    flux_types(),
)
def test_invert_volume_enclosed(equilibrium, coords, ftype):
    """Tests that can correctly invert the calculation of volume enclosed by
    a flux surface."""
    rho, t = coords
    vol, t = equilibrium.enclosed_volume(rho, t, ftype)
    rho2, t2 = equilibrium.invert_enclosed_volume(vol, t, ftype)
    assert rho2 == approx(rho)
    assert t2 is t


@given(
    floats(1.0, 100.0).flatmap(
        lambda x: tuples(
            just(x),
            equilibria(min_spatial_points=10, max_spatial_points=20, Btot_factor=x),
        )
    ),
    arbitrary_coordinates(
        (1.83, -1.75, 75.0), (3.9, 2.0, 80.0), max_side=5, xarray=True
    ),
)
def test_magnetic_field_strength(strength_equilibrium, coords):
    """Tests that the total magnetic field strenght is calculated correctly. It
    hould be a constant."""
    strength, equilib = strength_equilibrium
    R, z, t = coords
    Btot, t2 = equilib.Btot(R, z, t)
    assert t2 is t
    assert np.all(Btot == approx(strength, rel=1e-4, abs=1e-4))


@given(
    equilibrium_data_and_Te(),
    flux_types(),
    floats(0.0, 2 * np.pi),
    offset_pickers(),
)
def test_offsets(equilib_Te, ftype, theta, offset):
    """Tests that offsets to the equilibrium data are handled correctly."""
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, MagicMock(), offset)
    times = equilib_dat["rmag"].coords["t"]
    R, z, _ = equilib.spatial_coords(0.0, theta, times, ftype)
    assert np.all(R - offset.return_value[0] == approx(equilib_dat["rmag"]))


@given(equilibrium_data_and_Te(), offset_pickers())
def test_provenance(equilib_Te, offset):
    """Test that the appropriate provenance is created for the equilibrium object."""
    session = MagicMock()
    equilib_dat, Te = equilib_Te
    equilib = Equilibrium(equilib_dat, Te, session, offset)
    session.prov.entity.assert_called()
    # TODO: Consider checking that creation time and offset value are correct.
    for data in equilib_dat.values():
        equilib.provenance.hadMember.assert_any_call(data.attrs["partial_provenance"])
