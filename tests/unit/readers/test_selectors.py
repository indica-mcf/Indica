"""Test the functions used for choosing which channels to ignore."""

# Programmatically trigger click events to check correct channels are picked

# Check clicking a point twice will still include it

# Check passing in existing unselected channels
from contextlib import contextmanager
import json
import os
from tempfile import TemporaryDirectory
from time import sleep
from unittest import mock
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from hypothesis import given
from hypothesis.strategies import floats
from hypothesis.strategies import from_regex
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
from pytest import mark

from indica.readers.selectors import ignore_channels_from_dict
from indica.readers.selectors import ignore_channels_from_file
from indica.readers.selectors import use_cached_ignore_channels
from .mock_reader import ConcreteReader
from .test_abstract_reader import _check_calls_equivalent


def assert_called_with(mock, *expected_args, **expected_kwargs):
    """Checks the most recent call to the mock was made with these arguments.
    Unlike the standard implementation, this works with numpy arrays.
    """
    assert mock.call_args is not None
    assert _check_calls_equivalent(mock.call_args, (expected_args, expected_kwargs))


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
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
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
            lists(strat, unique=True),
        )
    ),
)
def test_select_channels(category, uid, instrument, quantity, dim, channel_args):
    """Check selecting channels properly handles caching."""
    bad_channels, intrinsic_bad_channels, expected1, expected2 = channel_args
    data = MagicMock()
    selector = MagicMock()
    data.coords[dim].dtype = (
        type(expected1[0])
        if len(expected1) > 0
        else type(expected2[0])
        if len(expected2) > 0
        else type(bad_channels[0])
        if len(bad_channels) > 0
        else type(intrinsic_bad_channels[0])
        if len(intrinsic_bad_channels) > 0
        else float
    )
    with cachedir() as cdir, patch.object(
        ConcreteReader, "_get_bad_channels"
    ) as get_bad:
        selector.return_value = expected1
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector)
        get_bad.return_value = intrinsic_bad_channels
        cache_key = reader._RECORD_TEMPLATE.format(
            reader._reader_cache_id, category, instrument, uid, quantity
        )
        cachefile = os.path.expanduser(
            os.path.join("~", cdir, reader.__class__.__name__, cache_key)
        )
        if os.path.isfile(cachefile):
            os.remove(cachefile)
        # Test when no cache file is present
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected1)
        assert_called_with(
            selector,
            data,
            dim,
            intrinsic_bad_channels + bad_channels,
            intrinsic_bad_channels,
        )
        assert os.path.isfile(cachefile)
        # Check when cache file present but select different channels
        creation_time = os.path.getctime(cachefile)
        selector.return_value = expected2
        sleep(1e-2)
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected2)
        assert_called_with(
            selector, data, dim, intrinsic_bad_channels + bad_channels, expected1
        )
        mod_time1 = os.path.getmtime(cachefile)
        assert creation_time < mod_time1
        # Check when cache file present and reuse those channels
        selector.side_effect = lambda data, dim, bad, cached: cached
        data.coords[dim].dtype = type(expected2[0]) if len(expected2) > 0 else float
        sleep(1e-2)
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected2)
        assert_called_with(
            selector, data, dim, intrinsic_bad_channels + bad_channels, expected2
        )
        mod_time2 = os.path.getmtime(cachefile)
        assert mod_time1 < mod_time2


@mark.filterwarnings("ignore:loadtxt")
@given(
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
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
        )
    ),
)
def test_use_cached_ignore_channels(
    category, uid, instrument, quantity, dim, channel_args
):
    bad_channels, intrinsic_bad_channels = channel_args
    data = MagicMock()
    selector = use_cached_ignore_channels
    data.coords[dim].dtype = (
        type(bad_channels[0])
        if len(bad_channels) > 0
        else type(intrinsic_bad_channels[0])
        if len(intrinsic_bad_channels) > 0
        else float
    )
    data.name = f"{instrument}_{quantity}"
    with cachedir() as cdir, patch.object(
        ConcreteReader, "_get_bad_channels"
    ) as get_bad:
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector)
        get_bad.return_value = intrinsic_bad_channels
        cache_key = reader._RECORD_TEMPLATE.format(
            reader._reader_cache_id, category, instrument, uid, quantity
        )
        cachefile = os.path.expanduser(
            os.path.join("~", cdir, reader.__class__.__name__, cache_key)
        )
        if os.path.isfile(cachefile):
            os.remove(cachefile)
        # Test when no cache file is present, should return nothing
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == intrinsic_bad_channels)
        assert os.path.isfile(cachefile)
        # Check when cache file present
        creation_time = os.path.getctime(cachefile)
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector)
        sleep(1e-2)
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == intrinsic_bad_channels)
        mod_time1 = os.path.getmtime(cachefile)
        assert creation_time < mod_time1


@mark.filterwarnings("ignore:loadtxt")
@given(
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
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
            lists(strat, unique=True),
        )
    ),
)
def test_ignore_channels_from_dict(
    category, uid, instrument, quantity, dim, channel_args
):
    bad_channels, intrinsic_bad_channels, expected1, expected2 = channel_args
    data = MagicMock()
    channel_dict1 = {f"{instrument}_{quantity}": expected1}
    channel_dict2 = {f"{instrument}_{quantity}": expected2}
    selector1 = ignore_channels_from_dict(ignore_dict=channel_dict1)
    selector2 = ignore_channels_from_dict(ignore_dict=channel_dict2)
    data.coords[dim].dtype = (
        type(expected1[0])
        if len(expected1) > 0
        else type(expected2[0])
        if len(expected2) > 0
        else type(bad_channels[0])
        if len(bad_channels) > 0
        else type(intrinsic_bad_channels[0])
        if len(intrinsic_bad_channels) > 0
        else float
    )
    data.name = f"{instrument}_{quantity}"
    with cachedir() as cdir, patch.object(
        ConcreteReader, "_get_bad_channels"
    ) as get_bad:
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector1)
        get_bad.return_value = intrinsic_bad_channels
        cache_key = reader._RECORD_TEMPLATE.format(
            reader._reader_cache_id, category, instrument, uid, quantity
        )
        cachefile = os.path.expanduser(
            os.path.join("~", cdir, reader.__class__.__name__, cache_key)
        )
        if os.path.isfile(cachefile):
            os.remove(cachefile)
        # Test when no cache file is present
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected1)
        assert os.path.isfile(cachefile)
        # Check when cache file present but select different channels
        creation_time = os.path.getctime(cachefile)
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector2)
        sleep(1e-2)
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected2)
        mod_time1 = os.path.getmtime(cachefile)
        assert creation_time < mod_time1


@mark.filterwarnings("ignore:loadtxt")
@given(
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
    from_regex(r"[a-zA-Z0-9_]+", fullmatch=True),
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
            lists(strat, unique=True),
        )
    ),
)
def test_ignore_channels_from_json(
    category, uid, instrument, quantity, dim, channel_args
):
    bad_channels, intrinsic_bad_channels, expected1, expected2 = channel_args
    data = MagicMock()
    with mock.patch(
        "builtins.open",
        mock_open(read_data=json.dumps({f"{instrument}_{quantity}": expected1})),
    ):
        selector1 = ignore_channels_from_file(filename="/dev/null")
    with mock.patch(
        "builtins.open",
        mock_open(read_data=json.dumps({f"{instrument}_{quantity}": expected2})),
    ):
        selector2 = ignore_channels_from_file(filename="/dev/null")
    data.coords[dim].dtype = (
        type(expected1[0])
        if len(expected1) > 0
        else type(expected2[0])
        if len(expected2) > 0
        else type(bad_channels[0])
        if len(bad_channels) > 0
        else type(intrinsic_bad_channels[0])
        if len(intrinsic_bad_channels) > 0
        else float
    )
    data.name = f"{instrument}_{quantity}"
    with cachedir() as cdir, patch.object(
        ConcreteReader, "_get_bad_channels"
    ) as get_bad:
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector1)
        get_bad.return_value = intrinsic_bad_channels
        cache_key = reader._RECORD_TEMPLATE.format(
            reader._reader_cache_id, category, instrument, uid, quantity
        )
        cachefile = os.path.expanduser(
            os.path.join("~", cdir, reader.__class__.__name__, cache_key)
        )
        if os.path.isfile(cachefile):
            os.remove(cachefile)
        # Test when no cache file is present
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected1)
        assert os.path.isfile(cachefile)
        # Check when cache file present but select different channels
        creation_time = os.path.getctime(cachefile)
        reader = ConcreteReader(0.0, 1.0, 100.0, MagicMock(), selector2)
        sleep(1e-2)
        channels = reader._select_channels(
            category, uid, instrument, quantity, data, dim, bad_channels
        )
        assert np.all(channels == expected2)
        mod_time1 = os.path.getmtime(cachefile)
        assert creation_time < mod_time1
