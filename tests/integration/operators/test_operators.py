from indica.examples.example_operators import example_fit_ts
from indica.examples.example_operators import example_tomo_1D
from indica.examples.example_operators import example_tomo_asymmetry

# TODO: expand testing to test output of operators, not just that they run ;-)!!


def test_tomo_1D_with_asymmetry():
    _ = example_tomo_1D(asymmetric_profile=True, plot=False)


def test_tomo_1D_with_symmetry():

    _ = example_tomo_1D(asymmetric_profile=False, plot=False)


def test_tomo_asymmetry_with_asymmetry():

    _ = example_tomo_asymmetry(asymmetric_profile=True, plot=False)


def test_tomo_asymmetry_with_symmetry():

    _ = example_tomo_asymmetry(asymmetric_profile=False, plot=False)


def test_fit_ts():

    _ = example_fit_ts(plot=False)
