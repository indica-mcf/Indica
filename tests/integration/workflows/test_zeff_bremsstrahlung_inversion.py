from indica.workflows.zeff_workflows import zeff_bremstrahlung

# TODO: expand testing to test output, not just that it runs ;-)!!


def test_zeff_brems_example():
    _ = zeff_bremstrahlung(plot=False)
