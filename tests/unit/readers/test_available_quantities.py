"""Test attributes available for all available data quantities."""
from indica.datatypes import DATATYPES
from indica.datatypes import UNITS
from indica.readers.available_quantities import INSTRUMENT_DATASTRUCTURE


def test_datatypes():
    """Test all quantities in INSTRUMENT_DATASTRUCTURE have a corresponding DATATYPE"""

    for instrument, quantities in INSTRUMENT_DATASTRUCTURE.items():
        for quantity, datatype in quantities.items():
            long_name, units_key = DATATYPES[datatype]

            assert datatype in DATATYPES
            assert units_key in UNITS
