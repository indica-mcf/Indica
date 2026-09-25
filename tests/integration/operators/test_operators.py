import pytest

from indica.examples.example_operators import fit_ts
from indica.examples.example_operators import tomo_asymmetry
from indica.examples.example_operators import tomo_sym_1D

try:
    from indica.examples.example_operators import aurora_run
except ImportError:
    pass

# TODO: expand testing to test output of operators, not just that they run ;-)!!


def test_tomo_1D_with_asymmetry():
    _ = tomo_sym_1D(asymmetric_profile=True, plot=False)


def test_tomo_1D_with_symmetry():

    _ = tomo_sym_1D(asymmetric_profile=False, plot=False)


def test_tomo_asymmetry_with_asymmetry():

    _ = tomo_asymmetry(asymmetric_profile=True, plot=False)


def test_tomo_asymmetry_with_symmetry():

    _ = tomo_asymmetry(asymmetric_profile=False, plot=False)


def test_fit_ts():

    _ = fit_ts(plot=False)


def test_aurora():
    pytest.importorskip("aurora", reason="Issues with Aurora installation")
    _ = aurora_run()
