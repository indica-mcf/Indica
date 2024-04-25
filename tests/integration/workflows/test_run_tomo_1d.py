from indica.workflows.run_tomo_1d import example_tomo


def test_run_tomo_1d():

    _ = example_tomo(pulse=0, plot=False)
