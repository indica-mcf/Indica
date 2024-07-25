"""Tests that lines of sight can be correctly read from the SURF data file."""

from itertools import cycle
from pathlib import Path
import re

from hypothesis import assume
from hypothesis import example
from hypothesis import given
from hypothesis.strategies import booleans
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
import pytest

import indica.readers.surf_los as surf_los

filepath = Path(surf_los.__file__).parent.parent / "data/surf_los.dat"
PIXEL = 0.00099


def change_case(string, is_uppercase):
    """Change letters in the string to be upper case where
    ``is_uppercase`` is true, false elsehwere."""
    return "".join(
        [
            letter.upper() if upper else letter.lower()
            for letter, upper in zip(string, cycle(is_uppercase))
        ]
    )


def assert_read_sxr_los(
    pulse, upper_case, camera, r_start_expected, z_start_expected, theta_chip, gamma
):
    """Test reading of lines of sight for SXR cameras"""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("sxr/" + camera, upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 35
    )
    theta_chip = np.radians(theta_chip - 360)
    f = 0.03531
    assert np.all(rstart == pytest.approx(r_start_expected))
    assert np.all(zstart == pytest.approx(z_start_expected))
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))
    theta = np.arctan2(zend[0] - zstart[0], rend[0] - rstart[0])
    assert theta_chip - theta == pytest.approx(np.arctan2(gamma * 17 * PIXEL, f))
    theta = np.arctan2(zend[10] - zstart[10], rend[10] - rstart[10])
    assert theta_chip - theta == pytest.approx(np.arctan2(gamma * 7 * PIXEL, f))


@given(integers(28792, 63899), lists(booleans(), min_size=1))
def test_read_sxr_t_los_1(pulse, upper_case):
    """Test reading of lines of sight for SXR camera T
    TODO: check why this test sometimes fails in CI"""
    assert_read_sxr_los(pulse, upper_case, "t", 2.848, 2.172, 265, -1)


@given(integers(63900, 87999), lists(booleans(), min_size=1))
def test_read_sxr_t_los_2(pulse, upper_case):
    """Test reading of lines of sight for SXR camera T"""
    assert_read_sxr_los(pulse, upper_case, "t", 2.848, 2.172, 275, 1)


@given(integers(88000, 92504), lists(booleans(), min_size=1))
def test_read_sxr_t_los_3(pulse, upper_case):
    """Test reading of lines of sight for SXR camera T"""
    assert_read_sxr_los(pulse, upper_case, "t", 2.848, 2.182, 275, 1)


@given(integers(92505, 10000000), lists(booleans(), min_size=1))
def test_read_sxr_t_los_4(pulse, upper_case):
    """Test reading of lines of sight for SXR camera T"""
    assert_read_sxr_los(pulse, upper_case, "t", 2.848, 2.172, 275, 1)


@given(integers(28792, 10000000), lists(booleans(), min_size=1))
def test_read_sxr_v_los(pulse, upper_case):
    """Test reading of lines of sight for SXR camera V"""
    assert_read_sxr_los(pulse, upper_case, "v", 2.848, 2.172, 265, -1)


@given(integers(35779, 10000000), lists(booleans(), min_size=1))
def test_read_sxr_h_los(pulse, upper_case):
    """Test reading of lines of sight for SXR camera h"""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("sxr/h", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 17
    )
    assert np.all(rstart == pytest.approx(5.917))
    assert np.all(zstart[:8] == pytest.approx(0.414))
    assert np.all(zstart[8:] == pytest.approx(-0.408))
    assert np.all(Tstart == pytest.approx(0.0))
    assert rend[0] == pytest.approx(0.429873)
    assert zend[0] == pytest.approx(-2.66607)
    assert Tend[0] == pytest.approx(1.53583)
    assert rend[10] == pytest.approx(0.429705)
    assert zend[10] == pytest.approx(1.02936)
    assert Tend[10] == pytest.approx(1.53555)


@given(integers(28792, 80288), lists(booleans(), min_size=1))
def test_read_kk3_los_1(pulse, upper_case):
    """Test reading the lines of sight for KK3."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("kk3", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 1
    )
    assert rstart[0] == pytest.approx(5.37)
    assert rend[0] == pytest.approx(1.563)
    assert zstart[0] == pytest.approx(0.133)
    assert zend[0] == pytest.approx(0.133)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(80289, 10000000), lists(booleans(), min_size=1))
def test_read_kk3_los_2(pulse, upper_case):
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("kk3", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 1
    )
    assert rstart[0] == pytest.approx(5.37)
    assert rend[0] == pytest.approx(1.563)
    assert zstart[0] == pytest.approx(0.248)
    assert zend[0] == pytest.approx(0.248)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(82264, 10000000), lists(booleans(), min_size=1))
def test_read_bolo_h_los_1(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5h", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 24
    )
    assert rstart[0] == pytest.approx(5.0301)
    assert rend[0] == pytest.approx(2.3929)
    assert zstart[0] == pytest.approx(0.4052)
    assert zend[0] == pytest.approx(-1.7272)
    assert rstart[10] == pytest.approx(5.0513)
    assert rend[10] == pytest.approx(1.9325)
    assert zstart[10] == pytest.approx(0.3538)
    assert zend[10] == pytest.approx(-0.5836)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(73758, 82263), lists(booleans(), min_size=1))
def test_read_bolo_h_los_2(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5h", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 24
    )
    assert rstart[0] == pytest.approx(5.0301)
    assert rend[0] == pytest.approx(2.3929)
    assert zstart[0] == pytest.approx(0.4052)
    assert zend[0] == pytest.approx(-1.7272)
    assert rstart[10] == pytest.approx(5.0513)
    assert rend[10] == pytest.approx(1.9325)
    assert zstart[10] == pytest.approx(0.3538)
    assert zend[10] == pytest.approx(-0.5836)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(63718, 73757), lists(booleans(), min_size=1))
def test_read_bolo_h_los_3(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5h", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 24
    )
    assert rstart[0] == pytest.approx(5.0301)
    assert rend[0] == pytest.approx(2.3929)
    assert zstart[0] == pytest.approx(0.4052)
    assert zend[0] == pytest.approx(-1.7272)
    assert rstart[10] == pytest.approx(5.0513)
    assert rend[10] == pytest.approx(1.9325)
    assert zstart[10] == pytest.approx(0.3538)
    assert zend[10] == pytest.approx(-0.5836)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(82264, 10000000), lists(booleans(), min_size=1))
def test_read_bolo_v_los_1(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5v", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 32
    )
    assert rstart[0] == pytest.approx(3.2311)
    assert rend[0] == pytest.approx(3.8678)
    assert zstart[0] == pytest.approx(2.3932)
    assert zend[0] == pytest.approx(0.6155)
    assert rstart[10] == pytest.approx(3.1708)
    assert rend[10] == pytest.approx(2.734)
    assert zstart[10] == pytest.approx(2.3894)
    assert zend[10] == pytest.approx(-1.6615)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(73758, 82263), lists(booleans(), min_size=1))
def test_read_bolo_v_los_2(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5v", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 32
    )
    assert rstart[0] == pytest.approx(3.2311)
    assert rend[0] == pytest.approx(3.8678)
    assert zstart[0] == pytest.approx(2.3932)
    assert zend[0] == pytest.approx(0.6155)
    assert rstart[10] == pytest.approx(3.1708)
    assert rend[10] == pytest.approx(2.734)
    assert zstart[10] == pytest.approx(2.3894)
    assert zend[10] == pytest.approx(-1.6615)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


@given(integers(63718, 73757), lists(booleans(), min_size=1))
def test_read_bolo_v_los_3(pulse, upper_case):
    """Test reading lines of sight for bolometric camera H."""
    rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
        filepath, pulse, change_case("bolo/kb5v", upper_case)
    )
    assert (
        len(rstart)
        == len(rend)
        == len(zstart)
        == len(zend)
        == len(Tstart)
        == len(Tend)
        == 32
    )
    assert rstart[0] == pytest.approx(3.2311)
    assert rend[0] == pytest.approx(3.8678)
    assert zstart[0] == pytest.approx(2.3932)
    assert zend[0] == pytest.approx(0.6155)
    assert rstart[10] == pytest.approx(3.1708)
    assert rend[10] == pytest.approx(2.734)
    assert zstart[10] == pytest.approx(2.3894)
    assert zend[10] == pytest.approx(-1.6615)
    assert np.all(Tstart == pytest.approx(0.0))
    assert np.all(Tend == pytest.approx(0.0))


AVAILABLE_INSTRUMENTS = ["sxr/h", "sxr/t", "sxr/v", "kk3", "bolo/kb5h", "bolo/kb5v"]
INVALID_PULSES = {
    "sxr/h": one_of(integers(max_value=28792), integers(min_value=10000001)),
    "sxr/t": one_of(integers(max_value=28792), integers(min_value=10000001)),
    "sxr/v": one_of(integers(max_value=28792), integers(min_value=10000001)),
    "kk3": one_of(integers(max_value=28792), integers(min_value=10000001)),
    "bolo/kb5h": one_of(
        integers(3, 63718), integers(max_value=0), integers(min_value=100000001)
    ),
    "bolo/kb5v": one_of(
        integers(3, 63718), integers(max_value=0), integers(min_value=100000001)
    ),
}


@given(text(), integers(0, 10000000))
@example("*", 1)
def test_invalid_instrument(instrument, pulse):
    """Test an exception is raised when try to read data for an instrument
    that doesn't exist."""
    assume(instrument.lower() not in AVAILABLE_INSTRUMENTS)
    with open(filepath, "r", encoding="latin-1") as f:
        data = f.read()
    assume(not re.search(rf"^\*.*{re.escape(instrument)}.*$", data, re.MULTILINE))
    with pytest.raises(surf_los.SURFException):
        surf_los.read_surf_los(filepath, pulse, instrument)


@given(
    sampled_from(AVAILABLE_INSTRUMENTS).flatmap(
        lambda x: tuples(just(x), INVALID_PULSES[x])
    )
)
def test_invalid_pulse(instrument_pulse):
    """Test an exception is raised when try to read data for a pulse which is
    not available."""
    instrument, pulse = instrument_pulse
    with pytest.raises(surf_los.SURFException):
        surf_los.read_surf_los(filepath, pulse, instrument)
