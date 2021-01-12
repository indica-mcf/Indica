"""Test methods present on the base class DataReader."""

from contextlib import contextmanager
from copy import copy
import datetime
import os
from tempfile import TemporaryDirectory
from time import sleep
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import one_of
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
import prov.model as prov
from pytest import approx
from pytest import mark
from xarray import DataArray

from indica.converters import TransectCoordinates
from indica.datatypes import ELEMENTS
from indica.utilities import coord_array
from .mock_reader import ConcreteReader
from .mock_reader import MockReader
from ..converters.test_lines_of_sight import los_coordinates_and_axes
from ..converters.test_magnetic import magnetic_coordinates_and_axes
from ..converters.test_transect import transect_coordinates_and_axes
from ..data_strategies import array_dictionaries
from ..data_strategies import data_arrays
from ..data_strategies import equilibrium_data
from ..strategies import machine_dimensions


tstarts = floats(0.0, 1000.0)
tends = floats(0.0, 1000.0).map(lambda x: 1000.0 - x)
times = tuples(tstarts, tends).map(sorted)
max_freqs = floats(1e-3, 10.0).map(lambda x: 1 / x)


@composite
def dicts_with(draw, *options, min_size=1, max_size=None):
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
def expected_data(
    draw,
    coordinate_transform_and_axes,
    *options,
    unique_transforms=False,
):
    """Strategy to produce a dictionary of DataArrays of the type that
    could be returned by a read operation.

    Parameters
    ----------
    coordinate_transform_and_axes : strategy
        Strategy to generate the coordinate transform object describing these
        data and the axes for the x1, x2, and t dimensions.
    options : Tuple[str, Tuple[str, str]]
        Tuples describing possibly quantities to be produced. Follows format
        ``(quantity_name, (general_datatype, specific_datatype))``.
    unique_transforms : bool
        Whether the data arrays in the result should all use the same transform
        object or unique ones of the same class.

    """

    def rename_coord(coord, ppa):
        if coord.ndim == 0:
            return coord
        old = coord.dims[0]
        parts = old.split("_")
        parts.insert(-1, ppa)
        return coord.rename({old: "_".join(parts)})

    start = draw(floats(0.0, 9.9999e2))
    stop = 1e3 - draw(floats(0.0, 9.9999e2 - start, exclude_min=True))
    n = draw(integers(2, 20))
    time = coord_array(np.linspace(start, stop, n), "t")
    if not unique_transforms:
        transform, x1, x2, t = draw(coordinate_transform_and_axes)
        result = draw(
            array_dictionaries(
                transform,
                (x1, x2, time),
                dict(options),
            )
        )
    else:
        items = draw(dicts_with(*options))
        result = {}
        for key, datatype in items.items():
            result[key] = draw(
                data_arrays(
                    datatype,
                    coordinate_transform_and_axes.map(
                        lambda x: (
                            x[0],
                            rename_coord(x[1], key),
                            rename_coord(x[2], key),
                            time,
                        )
                    ),
                )
            )
    for array in result.values():
        transform = array.attrs["transform"]
        if hasattr(transform, "equilibrium"):
            del transform.equilibrium
        if isinstance(transform, TransectCoordinates):
            x1name = transform.x1_name
            to_fix = [array]
            if "error" in array.attrs:
                to_fix.append(array.attrs["error"])
            if "dropped" in array.attrs:
                to_fix.append(array.attrs["dropped"])
                if "error" in array.attrs:
                    to_fix.append(array.attrs["dropped"].attrs["error"])
            for a in to_fix:
                a.coords["R"] = DataArray(
                    transform.R_vals.y, coords=[(x1name, a.coords[x1name].values)]
                )
                a.coords["z"] = DataArray(
                    transform.z_vals.y, coords=[(x1name, a.coords[x1name].values)]
                )
    return result


def find_dropped_channels(array, dimension):
    """Returns a list of the channel index numbers for channels which have
    been dropped from ``dimension`` of ``data``.

    """
    if "dropped" not in array.attrs:
        return []
    return array.attrs["dropped"].coords[dimension].values


def assert_values_equal(actual, expected, tstart, tend):
    """Checks actual array is equal to the expected one within the
    requested time range.

    """
    tslice = slice(tstart, tend)
    assert actual.equals(expected.sel(t=tslice))
    if "error" in expected.attrs:
        assert actual.attrs["error"].equals(expected.attrs["error"].sel(t=tslice))
    if "dropped" in expected.attrs:
        assert actual.attrs["dropped"].equals(expected.attrs["dropped"].sel(t=tslice))
        if "error" in expected.attrs:
            assert (
                actual.attrs["dropped"]
                .attrs["error"]
                .equals(expected.attrs["dropped"].attrs["error"].sel(t=tslice))
            )


def assert_values_binned_equal(actual, expected, max_freq):
    """Checks actual array is consistent with binning the expected one so
    as to satisfy the maximum frequency requirement.

    """
    min_width = 1 / max_freq
    times = actual.coords["t"]
    original_times = expected.coords["t"]
    spacing = original_times[1] - original_times[0]
    half_width = 0.5 * np.ceil(min_width / spacing) * spacing
    # if not np.all(expected.isel(t=-1).values == expected.isel(t=0).values):
    #     breakpoint()
    for t in times:
        tslice = slice(t - half_width, t + half_width)
        count = len(expected.coords["t"].sel(t=tslice))
        assert actual.sel(t=t).values == approx(
            expected.sel(t=tslice).mean("t").values, nan_ok=True
        )
        if "error" in expected.attrs:
            assert actual.attrs["error"].sel(t=t).values == approx(
                np.sqrt(
                    (expected.attrs["error"].sel(t=tslice) ** 2).mean("t") / count
                ).values,
                nan_ok=True,
            )
        if "dropped" in expected.attrs:
            assert actual.attrs["dropped"].sel(t=t).values == approx(
                expected.attrs["dropped"].sel(t=tslice).mean("t").values
            )
            if "error" in expected.attrs:
                assert actual.attrs["dropped"].attrs["error"].sel(t=t).values == approx(
                    np.sqrt(
                        (
                            expected.attrs["dropped"].attrs["error"].sel(t=tslice) ** 2
                        ).mean("t")
                        / count
                    ).values
                )


def assert_data_arrays_equal(actual, expected, tstart, tend, max_freq):
    """Performs various assertions to confirm that the two DataArray objects
    are equivalent."""
    assert actual.name == expected.name
    assert actual.attrs["datatype"] == expected.attrs["datatype"]
    times = actual.coords["t"]
    original_times = expected.coords["t"]
    assert np.all(times >= tstart - 0.5)
    assert np.all(times <= tend + 0.5)
    assert np.all(np.unique(times) == times)
    assert actual.dims == expected.dims
    if len(times) > 1:
        actual_freq = float((len(times) - 1) / (times[-1] - times[0]))
        original_freq = float(
            (len(original_times) - 1) / (original_times[-1] - original_times[0])
        )
        assert actual_freq <= max_freq
        assert (
            original_freq == approx(actual_freq)
            or (round(original_freq / actual_freq) + 1) * actual_freq > max_freq
        )
    assert actual.attrs["transform"] == expected.attrs["transform"]
    if len(expected.coords["t"].sel(t=slice(tstart, tend))) == len(actual.coords["t"]):
        assert_values_equal(actual, expected, tstart, tend)
    else:
        assert_values_binned_equal(actual, expected, max_freq)


def _check_calls_equivalent(actual_args, expected_args):
    """Checks two calls to a mock are equivalent. Unlike the standard,
    implementation, this one is designed to work with numpy arrays."""
    if len(actual_args[0]) != len(expected_args[0]):
        return False
    for a, e in zip(actual_args[0], expected_args[0]):
        if isinstance(a, np.ndarray) or isinstance(e, np.ndarray):
            val = np.all(a == e)
        else:
            val = a == e
        if not val:
            return False
    if len(actual_args[1]) != len(expected_args[1]):
        return False
    if actual_args[1].keys() != expected_args[1].keys():
        return False
    for key in actual_args[1]:
        a = actual_args[1][key]
        e = expected_args[1][key]
        if isinstance(a, np.ndarray) or isinstance(e, np.ndarray):
            val = np.all(a == e)
        else:
            val = a == e
        if not val:
            return False
    return True


def assert_any_call(mock, *expected_args, **expected_kwargs):
    """Checks the mock has been called with these arguments at some point.
    Unlike the standard implementation, this works with numpy arrays.
    """
    for call in mock.call_args_list:
        if _check_calls_equivalent(call, (expected_args, expected_kwargs)):
            return
    assert False, "No matching calls."


def assert_called_with(mock, *expected_args, **expected_kwargs):
    """Checks the most recent call to the mock was made with these arguments.
    Unlike the standard implementation, this works with numpy arrays.
    """
    assert mock.call_args is not None
    assert _check_calls_equivalent(mock.call_args, (expected_args, expected_kwargs))


# Ignore warnings when an empty array
pytestmark = mark.filterwarnings("ignore:Mean of empty slice")


@given(
    text().flatmap(
        lambda instrument: tuples(
            expected_data(
                transect_coordinates_and_axes(coord_name=instrument + "_coord"),
                ("ne", ("number_density", "electrons")),
                ("te", ("temperature", "electrons")),
            ),
            just(instrument),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_thomson_scattering(data_instrument, uid, revision, time_range, max_freq):
    """Test the get_thomson_scattering method correctly combines and processes
    raw data."""
    data, instrument = data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_thomson_scattering(uid, instrument, revision, quantities)
    reader._get_thomson_scattering.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "thomson_scattering",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    tuples(sampled_from(sorted(ELEMENTS)), text()).flatmap(
        lambda elem_instrument: tuples(
            expected_data(
                transect_coordinates_and_axes(
                    coord_name=elem_instrument[1] + "_coords"
                ),
                ("angf", ("angular_freq", elem_instrument[0])),
                ("conc", ("concentration", elem_instrument[0])),
                ("ti", ("temperature", elem_instrument[0])),
            ),
            just(elem_instrument[1]),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_charge_exchange(data_instrument, uid, revision, time_range, max_freq):
    """Test the get_charge_exchange method correctly combines and processes
    raw data."""
    data, instrument = data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_charge_exchange(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_charge_exchange(uid, instrument, revision, quantities)
    reader._get_charge_exchange.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call()
        assert_any_call(
            reader.create_provenance,
            "charge_exchange",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    text().flatmap(
        lambda instrument: tuples(
            expected_data(
                magnetic_coordinates_and_axes(name=instrument),
                ("te", ("temperature", "electrons")),
            ),
            just(instrument),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_cyclotron_emissions(data_instrument, uid, revision, time_range, max_freq):
    """Test the get_cyclotron_emissions method correctly combines and processes
    raw data."""
    data, instrument = data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_thomson_scattering(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_cyclotron_emissions(uid, instrument, revision, quantities)
    reader._get_cyclotron_emissions.assert_called_once_with(
        uid, instrument, revision, quantities
    )
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "cyclotron_emissions",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    tuples(text(), machine_dimensions()).flatmap(
        lambda instrument_dims: tuples(
            just(instrument_dims[1]),
            expected_data(
                los_coordinates_and_axes(
                    instrument_dims[1],
                    domain_as_dims=True,
                    name=instrument_dims[0],
                ),
                ("h", ("luminous_flux", "sxr")),
                ("t", ("luminous_flux", "sxr")),
                ("v", ("luminous_flux", "sxr")),
                unique_transforms=True,
            ),
            just(instrument_dims[0]),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
@mark.filterwarnings("ignore:invalid.*sqrt")
def test_sxr(dims_data_instrument, uid, revision, time_range, max_freq):
    """Test the get_radiation method correctly combines and processes
    raw SXR data."""
    machine_dims, data, instrument = dims_data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq, machine_dims)
    reader.set_radiation(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_radiation(uid, instrument, revision, quantities)
    reader._get_radiation.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "radiation",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    tuples(machine_dimensions(), text()).flatmap(
        lambda dims_instrument: tuples(
            just(dims_instrument[0]),
            expected_data(
                los_coordinates_and_axes(
                    dims_instrument[0], domain_as_dims=True, name=dims_instrument[1]
                ),
                ("kb5h", ("luminous_flux", "bolometric")),
                ("kb5v", ("luminous_flux", "bolometric")),
                unique_transforms=True,
            ),
            just(dims_instrument[1]),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
def test_bolometry(dims_data_instrument, uid, revision, time_range, max_freq):
    """Test the get_radiation method correctly combines and processes
    raw bolometry data."""
    machine_dims, data, instrument = dims_data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq, machine_dims)
    reader.set_radiation(next(iter(data.values())), data)
    quantities = set(data)
    results = reader.get_radiation(uid, instrument, revision, quantities)
    reader._get_radiation.assert_called_once_with(uid, instrument, revision, quantities)
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "radiation",
            uid,
            instrument,
            revision,
            q,
            [],
            find_dropped_channels(expected, expected.dims[1]),
        )


@given(
    tuples(machine_dimensions(), text()).flatmap(
        lambda dims_instrument: tuples(
            just(dims_instrument[0]),
            expected_data(
                los_coordinates_and_axes(
                    dims_instrument[0],
                    min_los=1,
                    max_los=1,
                    domain_as_dims=True,
                    toroidal_skew=False,
                    name=dims_instrument[1],
                ),
                ("h", ("effective_charge", "plasma")),
                ("v", ("effective_charge", "plasma")),
                unique_transforms=True,
            ),
            just(dims_instrument[1]),
        )
    ),
    text(),
    integers(),
    times,
    max_freqs,
)
@mark.filterwarnings("ignore:divide.*double_scalars")
def test_bremsstrahlung_spectroscopy(
    dims_data_instrument, uid, revision, time_range, max_freq
):
    """Test the get_bremsstrahlung_spectroscopy method correctly combines
    and processes raw data.

    """
    machine_dims, data, instrument = dims_data_instrument
    for quantity, array in data.items():
        array.name = instrument + "_" + quantity
    reader = MockReader(True, True, *time_range, max_freq, machine_dims)
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
        assert_any_call(
            reader.create_provenance,
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
                "rbnd",
                "zbnd",
            ]
        ),
        unique=True,
        min_size=1,
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
):
    """Test the get_equilibrium method correctly combines and processes raw
    data.

    """
    for key in data:
        data[key].name = calculation + "_" + data[key].name
    reader = MockReader(True, True, *time_range, max_freq)
    reader.set_equilibrium(data["ftor"], data)
    results = reader.get_equilibrium(uid, calculation, revision, set(quantities))
    for q, actual, expected in [(q, results[q], data[q]) for q in quantities]:
        assert_data_arrays_equal(actual, expected, *time_range, max_freq)
        assert_any_call(
            reader.create_provenance,
            "equilibrium",
            uid,
            calculation,
            revision,
            q,
            [],
            [],
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
    floats(),
    floats(),
    floats(),
    dictionaries(text(), text()),
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
    assert isinstance(reader.agent, prov.ProvAgent)
    assert hasattr(reader, "entity")
    assert isinstance(reader.entity, prov.ProvEntity)
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


@contextmanager
def cachedir():
    """Set up a fake cache directory for testing getting of channels to
    drop.

    """
    import indica.readers.abstractreader as areader

    old_cache = areader.CACHE_DIR
    userdir = os.path.expanduser("~")
    with TemporaryDirectory(dir=userdir) as new_cache:
        areader.CACHE_DIR = os.path.relpath(new_cache, userdir)
        try:
            yield areader.CACHE_DIR
        finally:
            areader.CACHE_DIR = old_cache


@mark.filterwarnings("ignore:loadtxt")
@given(
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    text(),
    sampled_from(
        [
            integers(-2147483647, 2147483647),
            floats(allow_nan=False, allow_infinity=False),
        ]
    ).flatmap(
        lambda strat: tuples(
            lists(strat, unique=True),
            lists(strat, unique=True),
            lists(strat, unique=True),
        )
    ),
)
def test_select_channels(key, dim, channel_args):
    """Check selecting channels properly handles caching."""
    bad_channels, expected1, expected2 = channel_args
    data = MagicMock()
    selector = MagicMock()
    data.coords[dim].dtype = (
        type(expected1[0])
        if len(expected1) > 0
        else type(expected2[0])
        if len(expected2) > 0
        else type(bad_channels[0])
        if len(bad_channels) > 0
        else float
    )
    with cachedir() as cdir:
        selector.return_value = expected1
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector)
        cachefile = os.path.expanduser(
            os.path.join("~", cdir, reader.__class__.__name__, key)
        )
        print(cachefile)
        if os.path.isfile(cachefile):
            os.remove(cachefile)
        # Test when no cache file is present
        channels = reader._select_channels(key, data, dim, bad_channels)
        assert np.all(channels == expected1)
        assert_called_with(selector, data, dim, bad_channels, [])
        assert os.path.isfile(cachefile)
        # Check when cache file present but select different channels
        creation_time = os.path.getctime(cachefile)
        selector.return_value = expected2
        sleep(1e-2)
        channels = reader._select_channels(key, data, dim, bad_channels)
        assert np.all(channels == expected2)
        assert_called_with(selector, data, dim, bad_channels, expected1)
        mod_time1 = os.path.getmtime(cachefile)
        assert creation_time < mod_time1
        # Check when cache file present and reuse those channels
        selector.side_effect = lambda data, dim, bad, cached: cached
        data.coords[dim].dtype = type(expected2[0]) if len(expected2) > 0 else float
        sleep(1e-2)
        channels = reader._select_channels(key, data, dim, bad_channels)
        assert np.all(channels == expected2)
        assert_called_with(selector, data, dim, bad_channels, expected2)
        mod_time2 = os.path.getmtime(cachefile)
        assert mod_time1 < mod_time2
