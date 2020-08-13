"""Test methods present on the base class DataReader."""

from copy import copy
import datetime
import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
import numpy as np
import prov.model as prov
from pytest import fixture
from xarray import DataArray

import indica.converters.time
from indica.datatypes import ELEMENTS
from .mock_reader import ConcreteReader
from .mock_reader import MockReader
from ..converters.test_lines_of_sight import los_coordinates
from ..converters.test_magnetic import magnetic_coordinates
from ..converters.test_transect import transect_coordinates
from ..data_strategies import array_dictionaries
from ..data_strategies import data_arrays
from ..data_strategies import equilibrium_data
from ..strategies import float_series


times = lists(floats(0.0, 1000.0), min_size=2, max_size=2).map(sorted)
max_freqs = floats(0.1, 1000.0)


def fake_bin_in_time(tstart: float, tend: float, interval: float, data: DataArray):
    """Fake implementation of indica.converters.time.bin_in_time. Rather than
    averaging values it downsamples, taking the value from the nearest
    available point."""
    npoints = round((tend - tstart) / interval) + 1
    tlabels = np.linspace(tstart, tend, npoints)
    return data.sel(t=tlabels, method="nearest")


@fixture
def patch_bin_in_time(monkeypatch):
    """Patch indica.converters.time.bin_in_time to user fake_bin_in_time."""
    monkeypatch.setattr(indica.converters.time, "bin_in_time", fake_bin_in_time)


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


def assert_data_arrays_equal(actual, expected, tstart, tend, max_freq):
    """Performs various assertions to confirm that the two DataArray objects
    are equivalent."""
    assert actual.name == expected.name
    assert actual.attrs["datatype"] == expected.attrs["datatype"]
    times = actual.coords["t"]
    assert np.all(times >= tstart - 0.5)
    assert np.all(times <= tend + 0.5)
    assert np.all(np.unique(times) == times)
    assert np.all(np.isin(times, expected.coords["t"]))
    assert len(times) / (times[-1] - times[0]) <= max_freq
    tslice = np.argwhere(np.isin(expected.coords["t"], times))
    assert actual.equals(expected.isel(t=tslice))
    if "error" in expected.attrs:
        assert actual.attrs["error"].equals(expected.attrs["error"].isel(t=tslice))
    if "dropped" in expected.attrs:
        assert actual.attrs["dropped"].equals(expected.attrs["dropped"].isel(t=tslice))
        if "error" in expected.attrs:
            assert (
                actual.attrs["dropped"]
                .attrs["error"]
                .equals(expected.attrs["dropped"].attrs["error"].isel(t=tslice))
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
    times,
    max_freqs,
)
def test_thomson_scattering(
    data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time
):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_thomson_scattering(uid, instrument, revision, quantities)
    reader._get_thomson_scattering.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "thomson_scattering",
            uid,
            instrument,
            revision,
            q,
            [],
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
    times,
    max_freqs,
)
def test_charge_exchange(
    data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time
):
    """Test the get_charge_exchange method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_charge_exchange(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_charge_exchange(uid, instrument, revision, quantities)
    reader._get_charge_exchange.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "charge_exchange",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(magnetic_coordinates(), ("te", ("temperature", "electrons")),),
    text(),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_cyclotron_emissions(
    data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time
):
    """Test the get_cyclotron_emissions method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_cyclotron_emissions(uid, instrument, revision, quantities)
    reader._get_cyclotron_emissions.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "cyclotron_emissions",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("h", ("luminous_flux", "sxr")),
        ("t", ("luminous_flux", "sxr")),
        ("v", ("luminous_flux", "sxr")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_sxr(data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time):
    """Test the get_radiation method correctly combines and processes
    raw SXR data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_radiation(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_radiation(uid, instrument, revision, quantities)
    reader._get_radiation.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "radiation",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("kb5h", ("luminous_flux", "bolometric")),
        ("kb5v", ("luminous_flux", "bolometric")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_bolometry(
    data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time
):
    """Test the get_radiation method correctly combines and processes
    raw bolometry data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_radiation(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_radiation(uid, instrument, revision, quantities)
    reader._get_bolometry.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "radiation",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    expected_data(
        los_coordinates(),
        ("h", ("effective_charge", "plasma")),
        ("v", ("effective_charge", "plasma")),
        unique_transforms=True,
    ),
    text(),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_bremsstrahlung_spectroscopy(
    data, uid, instrument, revision, time_range, max_freq, patch_bin_in_time
):
    """Test the get_bremsstrahlung_spectroscopy method correctly combines and processes
    raw data."""
    [finish_fake_array(v, instrument, k) for k, v in data.items]
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_bremsstrahlung_spectroscopy(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_bremsstrahlung_spectroscopy(
        uid, instrument, revision, quantities
    )
    reader._get_bremsstrahlung_spectroscopy.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "bremsstrahlung_spectroscopy",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
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
    times,
    max_freqs,
)
def test_equilibrium(
    data,
    quantities,
    uid,
    calculation,
    revision,
    time_range,
    max_freq,
    patch_bin_in_time,
):
    """Test the get_equilibrium method correctly combines and processes raw
    data.

    """
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_equilibrium(data["ftor"], data)
    results = reader.get_equilibrium(uid, calculation, revision, set(quantities))
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        reader.create_provenance.assert_any_call(
            "equilibrium",
            uid,
            calculation,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


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


def get_only_record(doc, record_type):
    """Get the only record from the Prov document with the specified type."""
    records = doc.get_records(record_type)
    assert len(records) == 1
    return records[0]


@given(
    floats(), floats(), floats(), dictionaries(text(), text()),
)
def test_prov_for_reader(tstart, tend, max_freq, extra_reader_attrs):
    """Check appropriate PROV data created for reader object"""
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    t1 = datetime.datetime.now()
    reader = ConcreteReader(tstart, tend, max_freq, session, **extra_reader_attrs)
    t2 = datetime.datetime.now()
    assert hasattr(reader, "agent")
    assert isinstance(reader.agent, prov.Agent)
    assert hasattr(reader, "entity")
    assert isinstance(reader.entity, prov.Entity)
    assert reader.agent.identifier == reader.entity.identifier == reader.prov_id
    deleg = get_only_record(doc, prov.ProvDelegation)
    assert reader.agent.idnetifier == deleg.get_attribute("prov:delegate")[1]
    assert session.agent.identifier == deleg.get_attribute("prov:responsible")[1]
    gen = get_only_record(doc, prov.ProvGeneration)
    assert reader.entity.identifier == gen.get_attribute("prov:entity")[1]
    assert session.session.identifier == gen.get_attribute("prov:activity")[1]
    assert t1 < gen.get_attribute("prov:time")[1] < t2
    attribution = get_only_record(doc, "prov.ProvAttribution")
    assert reader.entity.identifier == attribution.get_attribute("prov:entity")[1]
    assert session.agent.identifier == attribution.get_attribute("prov:agent")[1]


numbers = one_of(integers(), floats(allow_nan=False, allow_infinity=False))


@given(
    text(),
    text(),
    text(),
    integers(),
    text(),
    lists(text(), max_size=10, unique=True),
    lists(numbers, max_size=20, unique=True),
)
def test_prov_for_data(
    diagnostic, uid, instrument, revision, quantity, data_objects, ignored
):
    """Check appropriate PROV data created for data object"""
    doc = prov.ProvDocument()
    doc.set_default_namespace("https://ccfe.ukaea.uk/")
    session = MagicMock(
        prov=doc,
        agent=doc.agent("session_agent"),
        session=doc.activity("session_activity"),
    )
    reader = ConcreteReader(0.0, 1.0, 100.0, session)
    reader.DIAGNOSTIC_QUANTITIES = MagicMock()
    reader._start_time = datetime.datetime.now()
    entity = reader.create_provenance(
        diagnostic, uid, instrument, revision, quantity, data_objects, ignored
    )
    t2 = datetime.datetime.now()
    assert entity.get_attribute(prov.PROV_TYPE)[1] == "DataArray"
    assert isinstance(entity.get_attribute(prov.PROV_VALUE)[1], MagicMock)
    assert entity.get_attribute("ignored_channels") == str(ignored)
    generated = get_only_record(doc, prov.ProvGeneration)
    assert entity.identifier == generated.get_attribute("prov:entity")[1]
    activity_id = generated.get_attribute("prov:activity")[1]
    end_time = generated.get_attribute("prov:time")[1]
    assert reader._start_time < end_time < t2
    informed = get_only_record(doc, prov.ProvCommunication)
    assert activity_id == informed.get_attribute("prov:informed")[1]
    assert session.session.identifier == informed.get_attribute("prov:informant")[1]
    expected_agents = [session.agent.identifer, reader.agent.identifier]
    for a in doc.get_records(prov.ProvAssociation):
        assert activity_id == a.get_attribute("prov:activity")[1]
        agent_id = a.get_attribute("prov:agent")[1]
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    expected_agents = [session.agent.identifer, reader.agent.identifer]
    for a in doc.get_records(prov.ProvAttribution):
        assert entity.identifier == a.get_attribute("prov:entity")[1]
        agent_id = a.get_attribute("prov:agent")[1]
        assert agent_id in expected_agents
        expected_agents.remove(agent_id)
    assert len(expected_agents) == 0
    data = copy(data_objects)
    for d in doc.get_records(prov.ProvDerivation):
        assert entity.identifer == d.get_attribute("prov:generatedEntity")[1]
        used_id = d.get_attribute("prov:usedEntity")[1]
        assert used_id in data
        data.remove(used_id)
    assert len(data) == 0
    data = data_objects
    for u in doc.get_records(prov.ProvUsage):
        assert activity_id == u.get_attribute("prov:activity")
        entity_id = u.get_attribute("prov:entity")
        assert entity_id in data
        data.remove(entity_id)
    assert len(data) == 0


@fixture
def cachedir():
    """Set up a fake cache directory for testing getting of channels to
    drop.

    """
    import indica.readers.abstractreader as areader

    old_cache = areader.CACHE_DIR
    userdir = os.path.expanduser("~")
    with TemporaryDirectory(dir=userdir) as new_cache:
        areader.CACHE_DIR = os.path.relpath(new_cache, userdir)
        yield areader.CACHE_DIR
    areader.CACHE_DIR = old_cache


@given(
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    text(),
    lists(numbers, unique=True),
    lists(numbers, unique=True),
    lists(numbers, unique=True),
)
def test_select_channels(key, dim, bad_channels, expected1, expected2, cachedir):
    """Check selecting channels properly handles caching."""
    data = MagicMock()
    selector = MagicMock()
    selector.return_value = expected1
    reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector)
    cachefile = os.path.join("~", cachedir, reader.__class__.__name__, key)
    if os.path.isfile(cachefile):
        os.remove(cachefile)
    # Test when no cache file is present
    channels = reader._select_channels(key, data, dim, bad_channels)
    assert np.all(channels == expected1)
    selector.assert_called_with(data, dim, bad_channels, [])
    assert os.path.isfile(cachefile)
    # Check when cache file present but select different channels
    creation_time = os.path.getctime(cachefile)
    selector.return_value = expected2
    channels = reader._select_channels(key, data, dim, bad_channels)
    assert np.all(channels == expected2)
    selector.assert_called_with(data, dim, bad_channels, expected1)
    mod_time1 = os.path.getmtime(cachefile)
    assert creation_time < mod_time1
    # Check when cache file present and reuse those channels
    selector.side_effect = lambda data, dim, bad, cached: cached
    channels = reader._select_channels(key, data, dim, bad_channels)
    assert np.all(channels == expected2)
    selector.assert_called_with(data, dim, bad_channels, expected2)
    mod_time2 = os.path.getmtime(cachefile)
    assert mod_time1 < mod_time2
