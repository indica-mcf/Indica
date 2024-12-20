from indica.workflows.zeff_workflows import example_zeff_bremstrahlung

# TODO: expand testing to test output, not just that it runs ;-)!!


def test_zeff_brems_example():
    _ = example_zeff_bremstrahlung(plot=False)