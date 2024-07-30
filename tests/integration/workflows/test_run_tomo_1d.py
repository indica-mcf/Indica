from indica.examples.example_operators import example_tomo_1D

# TODO: expand testing!!

def test_tomo_1D_with_asymmetry():
    _ = example_tomo_1D(asymmetric_profile=True, plot=False)


def test_tomo_1D_with_symmetry():

    _ = example_tomo_1D(asymmetric_profile=False, plot=False)
