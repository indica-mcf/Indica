from indica.workflows.zeff_workflows import calculate_zeff


def test_calculate_zeff_phantom():

    _ = calculate_zeff(pulse=0, plot=False)
