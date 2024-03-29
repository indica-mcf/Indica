"""Test attributes available for all available data quantities."""


def test_datatypes():
    """Test import of AVAILABLE_QUANTITIES
    If fails --> check"""
    from indica.datatypes import DATATYPES
    from indica.readers.available_quantities import AVAILABLE_QUANTITIES

    datatypes = DATATYPES
    available_quantities = AVAILABLE_QUANTITIES

    assert datatypes == DATATYPES
    assert available_quantities == AVAILABLE_QUANTITIES
