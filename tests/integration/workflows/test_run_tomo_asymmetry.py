from indica.examples.example_operators import example_tomo_asymmetry

# TODO: expand testing!!


def test_tomo_asymmetry_with_asymmetry():

    _ = example_tomo_asymmetry(asymmetric_profile=True, debug=False, plot=False)


def test_tomo_asymmetry_with_symmetry():

    _ = example_tomo_asymmetry(asymmetric_profile=False, debug=False, plot=False)
