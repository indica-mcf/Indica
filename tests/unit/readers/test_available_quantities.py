"""Test attributes available for all available data quantities."""
from indica.datatypes import DATATYPES
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


def test_datatypes():
    """Test all quantities in AVAILABLE_QUANTITIES have a corresponding DATATYPE"""

    for instrument, quantities in AVAILABLE_QUANTITIES.items():
        for quantity, datatype in quantities.items():
            assert datatype in DATATYPES
