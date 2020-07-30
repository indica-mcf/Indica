"""Test methods present on the base class DataReader."""

from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
import numpy as np

from src.datatypes import ELEMENTS
from .mock_reader import MockReader
from ..converters.test_lines_of_sight import los_coordinates
from ..converters.test_magnetic import magnetic_coordinates
from ..converters.test_transect import transect_coordinates
from ..data_strategies import array_dictionaries
from ..data_strategies import data_arrays
from ..data_strategies import equilibrium_data
from ..strategies import float_series


@composite
def dicts_with(draw, *options, min_size=0, max_size=None):
    """Strategy to produce a dictionary containig some combination of the
    key-value pairs passed as arguments in the form of tuples.

    """
    return dict(
        draw(
            lists(
                sampled_from(options), min_size=min_size, max_size=max_size, unique=True
            )
        )
    )


@composite
def expected_data(draw, coordinate_transform, *options, unique_transforms=False):
    """Strategy to produce a dictionary of DataArrays of the type that
    could be returned by a read operation.

    Parameters
    ----------
    coordinate_transform : CoordinateTransform strategy
        Strategy to generate the coordinate transform object describing these
        data.
    options : Tuple[str, Tuple[str, str]]
        Tuples describing possibly quantities to be produced. Follows format
        ``(quantity_name, (general_datatype, specific_datatype))``.
    unique_transforms : bool
        Whether the data arrays in the result should all use the same transform
        object or unique ones of the same class.
    """
    time = (float_series(0.0, 1e3),)
    if not unique_transforms:
        return draw(
            array_dictionaries(
                draw(coordinate_transform),
                dict(options),
                override_coords=[None, None, time],
            )
        )
    else:
        items = draw(dicts_with(options))
        result = {}
        for key, datatype in items:
            result[key] = data_arrays(datatype, coordinate_transform)
        return result


def find_dropped_channels(array, dimension):
    """Returns a list of the channel index numbers for channels which have
    been dropped from ``dimension`` of ``data``.

    """
    if "dropped" not in array.attrs:
        return []
    return [
        np.nonzero(array.coords[dimension] == v)[0][0]
        for v in array.attrs["dropped"].coords[dimension]
    ]


def assert_data_arrays_equal(actual, expected):
    """Performs various assertions to confirm that the two DataArray objects
    are equivalent."""
    assert actual.name == expected.name
    assert actual.attrs["datatype"] == expected.attrs["datatype"]
    assert actual.equals(expected)
    if "error" in expected.attrs:
        assert actual.attrs["error"].equals(expected.attrs["error"])
    if "dropped" in expected.attrs:
        assert actual.attrs["dropped"].equals(expected.attrs["dropped"])
        if "error" in expected.attrs:
            assert (
                actual.attrs["dropped"]
                .attrs["error"]
                .equals(expected.attrs["dropped"].attrs["error"])
            )
    assert actual.attrs["transform"] == expected.attrs["transform"]


def finish_fake_array(array, instrument, quantity, coord_name1=None, coord_name2=None):
    """Modify the provided data array (in place) to use appropriate names."""
    array.name = instrument + "_" + quantity
    dims = ["t"]
    if len(array.dims) > 1:
        dims.append(coord_name1 if coord_name1 else instrument + "_coord")
    if len(array.dims) > 2:
        dims.append(coord_name2 if coord_name2 else instrument + "_coord2")
    if len(array.dims) > 3:
        dims.extend(array.dims[3:])
    dims = tuple(dims)
    array.dims = dims
    if "error" in array.attrs:
        array.attrs["error"].dims = dims
    if "dropped" in array.attrs:
        array.attrs["dropped"].dims = dims
        if "error" in array.attrs:
            array.attrs["dropped"].attrs["error"].dims = dims


@given(
    expected_data(
        transect_coordinates(),
        ("ne", ("number_density", "electrons")),
        ("te", ("temperature", "electrons")),
    ),
    text(),
    text(),
    integers(),
)
def test_thomson_scattering(data, uid, instrument, revision):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_thomson_scattering(uid, instrument, revision, quantities)
    reader._get_thomson_scattering.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "thomson_scattering",
            uid,
            instrument,
            revision,
            q,
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    sampled_from(ELEMENTS).flatmap(
        lambda x: expected_data(
            transect_coordinates(),
            ("angf", ("angular_freq", x)),
            ("conc", ("concentration", x)),
            ("ti", ("temperature", x)),
        )
    ),
    text(),
    text(),
    integers(),
)
def test_charge_exchange(data, uid, instrument, revision):
    """Test the get_charge_exchange method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_charge_exchange(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_charge_exchange(uid, instrument, revision, quantities)
    reader._get_charge_exchange.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "charge_exchange",
            uid,
            instrument,
            revision,
            q,
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(magnetic_coordinates(), ("te", ("temperature", "electrons")),),
    text(),
    text(),
    integers(),
)
def test_cyclotron_emissions(data, uid, instrument, revision):
    """Test the get_cyclotron_emissions method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_cyclotron_emissions(uid, instrument, revision, quantities)
    reader._get_cyclotron_emissions.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "cyclotron_emissions",
            uid,
            instrument,
            revision,
            q,
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("H", ("luminous_flux", "sxr")),
        ("T", ("luminous_flux", "sxr")),
        ("V", ("luminous_flux", "sxr")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
)
def test_radiation(data, uid, instrument, revision):
    """Test the get_radiation method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_radiation(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_radiation(uid, instrument, revision, quantities)
    reader._get_radiation.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "radiation",
            uid,
            instrument,
            revision,
            q,
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("H", ("luminous_flux", "bolometric")),
        ("V", ("luminous_flux", "bolometric")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
)
def test_bolometry(data, uid, instrument, revision):
    """Test the get_bolometry method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_bolometry(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_bolometry(uid, instrument, revision, quantities)
    reader._get_bolometry.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "bolometry",
            uid,
            instrument,
            revision,
            q,
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("H", ("effective_charge", "plasma")),
        ("V", ("effective_charge", "plasma")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
)
def test_bremsstrahlung_spectroscopy(data, uid, instrument, revision):
    """Test the get_bremsstrahlung_spectroscopy method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader()
    reader.set_bremsstrahlung_spectroscopy(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_bremsstrahlung_spectroscopy(
        uid, instrument, revision, quantities
    )
    reader._get_bremsstrahlung_spectroscopy.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)
        reader.create_provenance.assert_any_call(
            "bolometry", uid, instrument, revision, q, []
        )


@given(
    equilibrium_data(),
    lists(
        sampled_from(
            [
                "f",
                "faxs",
                "fbnd",
                "ftor",
                "rmji",
                "rmjo",
                "psi",
                "vjac",
                "rmag",
                "zmag",
                "rsep",
                "zsep",
            ]
        ),
        unique=True,
    ),
    text(),
    text(),
    integers(),
)
def test_equilibrium(data, quantities, uid, calculation, revision):
    """Test the get_equilibrium method correctly combines and processes raw
    data.

    """
    reader = MockReader()
    reader.set_equilibrium(data["ftor"], data)
    results = reader.get_equilibrium(uid, calculation, revision, set(quantities))
    for actual, expected in [(results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected)


@patch.object(MockReader, "close")
def test_context_manager(mock_close):
    """Check works properly in context manager."""
    with MockReader() as reader:
        print(reader.requires_authentication)
    mock_close.assert_called()


@given(text(), text())
def test_default_authentication(username, password):
    """Check default authenticate always returns True"""
    reader = MockReader()
    assert reader.authenticate(username, password)


# Check appropriate PROV data created for reader object

# Check selecting channels properly handles caching
