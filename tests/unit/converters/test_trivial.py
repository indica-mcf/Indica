"""Test the trivial coordinate transform."""

import numpy as np

from indica.converters import TrivialTransform

TRIVIAL = TrivialTransform()


def test_convert_to_Rz_single():
    """Test trivial transform returns arguments."""
    x1, x2, t = 0, 1, 2
    _x1, _x2 = TRIVIAL.convert_to_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2


def test_convert_to_Rz_array():
    """Test trivial transform returns arguments."""
    x1, x2, t = np.array([0, 1]), np.array([2, 3]), 2
    _x1, _x2 = TRIVIAL.convert_to_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2


def test_convert_to_Rz_multi_time():
    """Test trivial transform returns arguments."""
    x1, x2, t = np.array([0, 1]), np.array([2, 3]), np.array([2, 3, 4, 5])
    _x1, _x2 = TRIVIAL.convert_to_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2


def test_convert_from_Rz_single():
    """Test trivial transform returns arguments."""
    x1, x2, t = 0, 1, 2
    _x1, _x2 = TRIVIAL.convert_from_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2


def test_convert_from_Rz_array():
    """Test trivial transform returns arguments."""
    x1, x2, t = np.array([0, 1]), np.array([2, 3]), 2
    _x1, _x2 = TRIVIAL.convert_from_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2


def test_convert_from_Rz_multi_time():
    """Test trivial transform returns arguments."""
    x1, x2, t = np.array([0, 1]), np.array([2, 3]), np.array([2, 3, 4, 5])
    _x1, _x2 = TRIVIAL.convert_from_Rz(x1, x2, t)
    assert x1 is _x1
    assert x2 is _x2
