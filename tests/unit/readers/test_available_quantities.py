"""Test attributes available for all available data quantities."""
from indica.datatypes import DATATYPES
from indica.datatypes import UNITS
from indica.readers.available_quantities import AVAILABLE_QUANTITIES


def test_datatypes():
    """Test all quantities in AVAILABLE_QUANTITIES have a corresponding DATATYPE"""

    for instrument, quantities in AVAILABLE_QUANTITIES.items():
        for quantity, datatype_dims in quantities.items():
            datatype, dims = datatype_dims
            long_name, units_key = DATATYPES[datatype]

            assert datatype in DATATYPES
            assert units_key in UNITS
