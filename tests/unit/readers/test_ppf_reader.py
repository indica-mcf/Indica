"""Test reading from SAL database."""

from contextlib import nullcontext
import pathlib
import re
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis import settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
import pytest
import sal.client
import sal.core.exception
import scipy.constants as sc

from indica.readers import PPFReader
from .fake_salclient import fake_sal_client


@pytest.fixture(scope="module")
def fake_sal():
    """Loads data to create a fake SALClient class."""
    return fake_sal_client(pathlib.Path(__file__).parent.absolute() / "ppf_samples.pkl")


pulses = integers(1, 99999)
times = lists(floats(30.0, 80.0), min_size=2, max_size=2).map(sorted)
errors = floats(0.0001, 0.2)
max_freqs = floats(2.0, 1000.0)
revisions = integers(0)
actual_revisions = integers(1)
edited_revisions = lists(actual_revisions, min_size=1, unique=True).map(sorted)
lines_of_sight = tuples(
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
    arrays(float, 35, elements=floats(allow_infinity=False, allow_nan=False)),
)


def trim_lines_of_sight(los, n):
    """Return a new tuple with the LOS arrays trimmed to length n."""
    return tuple(d[:n] for d in los)


def get_record(reader, pulse, uid, instrument, dtype, revision):
    """Gets the path for the requested recrod, with the correct revision for
    the data actually heald in the database."""
    path = f"/pulse/{pulse:d}/ppf/signal/{uid}/{instrument}/{dtype}"
    rev = reader._client.list(path + f":{revision}").revision_current
    return path + f":{rev}"


def test_needs_authentication():
    """Test that the if needs_authentication is true then will not be able
    to fetch data.

    TODO: consider whether I should use mocking so both possibilities are tested.
    """
    reader = PPFReader(90272, 0.0, 100.0, selector=MagicMock(), session=MagicMock())
    if reader.requires_authentication:
        with pytest.raises(sal.core.exception.AuthenticationFailed):
            reader._get_thomson_scattering("jetppf", "hrts", 0, {"te"})
    else:
        reader._get_thomson_scattering("jetppf", "hrts", 0, {"te"})


@given(pulses, times, errors, max_freqs, text(), text())
def test_authentication(fake_sal, pulse, time_range, error, freq, user, password):
    """Test authentication method on client get called."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
        assert reader.authenticate(user, password)
        reader._client.authenticate.assert_called_once_with(user, password)


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    sampled_from(["hrts", "lidr"]),
    revisions,
    edited_revisions,
    lists(sampled_from(["te", "ne"]), min_size=1, unique=True).map(set),
)
@settings(report_multiple_bugs=False)
def test_get_thomson_scattering(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
):
    """Test quantities returned by _get_thomson_scattering are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    with pytest.raises(sal.core.exception.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_thomson_scattering(uid, instrument, revision, quantities)
    if bad_rev:
        return
    z_signal = reader._client.DATA[f"{instrument}/z"]
    assert np.all(z_signal.data == results["z"])
    assert len(z_signal.data) == results["length"]
    assert np.all(z_signal.dimensions[0].data == results["R"])
    records = [get_record(reader, pulse, uid, instrument, "z", revision)]
    for q in quantities:
        signal = reader._client.DATA[f"{instrument}/{q}"]
        assert np.all(results[q] == signal.data)
        assert np.all(results["times"] == signal.dimensions[0].data)
        if instrument == "lidr":
            error_signal = reader._client.DATA[f"{instrument}/{q}u"]
            assert np.all(results[q] + results[q + "_error"] == error_signal.data)
        else:
            error_signal = reader._client.DATA[f"{instrument}/d{q}"]
            assert np.all(results[q + "_error"] == error_signal.data)
        assert np.all(results["times"] == error_signal.dimensions[0].data)
        expected = sorted(
            records
            + list(
                map(
                    lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                    [q, q + "u" if instrument == "lidr" else "d" + q],
                )
            )
        )
        assert sorted(results[q + "_records"]) == expected


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("cgiroud"),
    just("cxg6"),
    revisions,
    edited_revisions,
    lists(sampled_from(["angf", "conc", "ti"]), min_size=1, unique=True).map(set),
)
def test_get_charge_exchange(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
):
    """Test quantities returned by _get_charge_exchange are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    with pytest.raises(sal.core.exception.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_charge_exchange(uid, instrument, revision, quantities)
    if bad_rev:
        return
    z_signal = reader._client.DATA[f"{instrument}/pos"]
    assert np.all(z_signal.data == results["z"])
    assert len(z_signal.data) == results["length"]
    assert np.all(reader._client.DATA[f"{instrument}/rpos"] == results["R"])
    assert np.all(reader._client.DATA[f"{instrument}/texp"].data == results["texp"])
    records = ["pos", "rpos", "texp", "mass"].maps(
        lambda q: get_record(reader, pulse, uid, instrument, q, revision)
    )
    uncertainties = {"angf": "afhi", "conc": "cohi", "ti": "tihi"}
    for q in quantities:
        signal = reader._client.DATA[f"{instrument}/{q}"]
        assert np.all(results[q] == signal.data)
        assert np.all(results["times"] == signal.dimensions[0].data)
        error_signal = reader._client.DATA[f"{instrument}/{uncertainties[q]}"]
        assert np.all(results[q + "_error"] + results[q] == error_signal.data)
        assert np.all(results["times"] == error_signal.dimensions[0].data)
        assert sorted(results[q + "_records"]) == sorted(
            records
            + map(
                lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                [q, uncertainties[q]],
            )
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    sampled_from(["efit", "eftp"]),
    revisions,
    edited_revisions,
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
                "rmag",
                "rbnd",
                "vjac",
                "zmag",
                "zbnd",
            ]
        ),
        min_size=1,
        unique=True,
    ).map(set),
)
def test_get_equilibrium(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
):
    """Test quantities returned by _get_equilibrium are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    with pytest.raises(sal.core.exception.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_equilibrium(uid, instrument, revision, quantities)
    if bad_rev:
        return
    signal = reader._client.DATA[f"{instrument}/f"]
    if len({"f", "ftor", "vjac", "rmji", "rmjo"} & quantities) > 0:
        assert np.all(signal.dimensions[1].data == results["psin"])
    for q in quantities:
        signal = reader._client.DATA[f"{instrument}/{q}"]
        assert np.all(results[q].flatten() == signal.data.flatten())
        assert np.all(results["times"] == signal.dimensions[0].data)
        if q == "psi":
            assert sorted(results[q + "_records"]) == sorted(
                map(
                    lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                    ["psi", "psir", "psiz"],
                )
            )
        else:
            assert results[q + "_records"] == [
                get_record(reader, pulse, uid, instrument, q, revision)
            ]


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    just("kk3"),
    just({"te"}),
    revisions,
    edited_revisions,
    lines_of_sight,
)
def test_get_cyclotron_emissions(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    quantities,
    revision,
    available_revisions,
    los,
):
    """Test quantities returned by _get_cyclotrons_emissions are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    mock_surf = MagicMock(return_value=trim_lines_of_sight(los, 1))
    with patch("indica.readers.surf_los.read_surf_los", mock_surf), pytest.raises(
        sal.core.exception.NodeNotFound
    ) if bad_rev else nullcontext():
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    assert results["z"] == los[2][0]
    # TODO: determine how best to describe the SURF data for PROV
    records = [
        "surf_los.dat",
        get_record(reader, pulse, uid, instrument, "gen", revision),
    ]
    gen = reader._client.DATA[f"{instrument}/gen"]
    for q in quantities:
        emissions = results[q]
        channel_names = [
            key.split("/")[-1]
            for key in reader._client.DATA
            if re.search(rf"{q}\d\d$", key, re.I)
        ]
        channel_indices = [int(c[-2:]) - 1 for c in channel_names]
        for i, name in enumerate(channel_names):
            assert np.all(
                emissions[:, i] == reader._client.DATA[f"{instrument}/{name}"]
            )
        assert np.all(
            results["Btot"] * sc.e * gen[11, channel_indices] / (2 * np.pi * sc.m_e)
            == pytest.approx(gen[15, channel_indices])
        )
        assert np.all(results[q + "_error"] == pytest.approx(0.1 * emissions))
        bad_channels = np.argwhere(np.isin(results["Btot"], results["bad_channels"]))
        assert np.all(gen[18, bad_channels] == 0)
        assert np.all(gen[19, bad_channels] == 0)
        assert np.all(np.delete(gen[18, :], bad_channels) != 0)
        assert np.all(np.delete(gen[19, :], bad_channels) != 0)
        assert sorted(results[q + "_records"]) == sorted(
            records
            + map(
                lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                channel_names,
            )
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    just("sxr"),
    revisions,
    edited_revisions,
    lists(sampled_from(["h", "t", "v"]), min_size=1, unique=True).map(set),
    lines_of_sight,
)
@settings(report_multiple_bugs=False)
def test_get_sxr(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    los,
):
    """Test SXR quantities returned by _get_radiation are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    LOS_LENS = {"sxr/h": 17, "sxr/t": 35, "sxr/v": 35}
    mock_surf = MagicMock(
        side_effect=lambda f, p, inst: trim_lines_of_sight(los, LOS_LENS[inst])
    )
    with patch("indica.readers.surf_los.read_surf_los", mock_surf), pytest.raises(
        sal.core.exception.NodeNotFound
    ) if bad_rev else nullcontext():
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    assert results["machine_dims"] == ((1.83, 3.9), (-1.75, 2.0))
    # TODO: determine how best to describe the SURF data for PROV
    records = ["surf_los.dat"]
    for q in quantities:
        radiation = results[q]
        assert results["length"][q] == radiation.shape[1]
        channel_names = [
            key.split("/")[-1]
            for key in reader._client.DATA
            if re.search(rf"{q}\d\d$", key, re.I)
        ]
        channel_indices = [int(c[-2:]) - 1 for c in channel_names]
        for i, name in enumerate(channel_names):
            signal = reader._client.DATA[f"{instrument}/{name}"]
            assert np.all(radiation[:, i] == signal.data)
            assert np.all(results[q + "_times"] == signal.dimensions[0].data)
        assert np.all(results[q + "_Rstart"] == los[0][channel_indices])
        assert np.all(results[q + "_Rstop"] == los[1][channel_indices])
        assert np.all(results[q + "_zstart"] == los[2][channel_indices])
        assert np.all(results[q + "_zstop"] == los[3][channel_indices])
        assert np.all(results[q + "_error"] == pytest.approx(error * radiation))
        assert sorted(results[q + "_records"]) == sorted(
            records
            + list(
                map(
                    lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                    channel_names,
                )
            )
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    just("bolo"),
    revisions,
    edited_revisions,
    lists(sampled_from(["kb5h", "kb5v"]), min_size=1, unique=True).map(set),
    lines_of_sight,
)
def test_get_radiation(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    los,
):
    """Test bolometric quantities returned by _get_radiation are correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    LOS_LENS = {"bolo/kb5v": 32, "bolo/kb5h": 24}
    mock_surf = MagicMock(
        side_effect=lambda f, p, inst: trim_lines_of_sight(los, LOS_LENS[inst])
    )
    with patch("indica.readers.surf_los.read_surf_los", mock_surf), pytest.raises(
        sal.core.exception.NodeNotFound
    ) if bad_rev else nullcontext():
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    # TODO: determine how best to describe the SURF data for PROV
    records = ["surf_overlays.db"]
    for q in quantities:
        radiation = results[q]
        length = results["length"][q]
        assert length == radiation.shape[1]
        signal = reader._client.DATA[f"{instrument}/{q}"]
        assert np.all(radiation == signal.data)
        assert np.all(results[q + "_times"] == signal.dimensions[0].data)
        assert np.all(results[q + "_Rstart"] == los[0][:length])
        assert np.all(results[q + "_Rstop"] == los[1][:length])
        assert np.all(results[q + "_zstart"] == los[2][:length])
        assert np.all(results[q + "_zstop"] == los[3][:length])
        assert np.all(results[q + "_error"] == pytest.approx(error * radiation))
        assert sorted(results[q + "_records"]) == sorted(
            records + [get_record(reader, pulse, uid, instrument, q, revision)]
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    just("jetppf"),
    just("ks3"),
    revisions,
    edited_revisions,
    lists(sampled_from(["zefh", "zefv"]), min_size=1, unique=True).map(set),
)
def test_get_bremsstrahlung_spectroscopy(
    fake_sal,
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
):
    """Test data returned by _get_bremsstrahlung_spectroscopy is correct."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    reader._client._revisions = available_revisions
    bad_rev = revision != 0 and revision < available_revisions[0]
    with pytest.raises(sal.core.exception.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_charge_exchange(uid, instrument, revision, quantities)
    if bad_rev:
        return
    for q in quantities:
        signal = reader._client.DATA[f"{instrument}/{q}"]
        assert np.all(results[q] == signal.data)
        assert np.all(results["times"] == signal.dimensions[0].data)
        error_signal = reader._client.DATA[f"{instrument}/{q[0]}{q[-1]}hi"]
        assert np.all(
            results["q"] + results[q + "_error"] == pytest.approx(error_signal.data)
        )
        assert np.all(results["times"] == error_signal.dimensions[0].data)
        los = reader._client.DATA[f"edg7/los{q[-1]}"]
        assert results[q + "Rstart"] == los[0]
        assert results[q + "Rend"] == los[1]
        assert results[q + "zstart"] == los[2]
        assert results[q + "zend"] == los[3]
        assert sorted(results[q + "_records"]) == sorted(
            map(
                lambda x: get_record(reader, pulse, uid, x[0], x[1], revision),
                [
                    ("edg7", f"los{q[-1]}"),
                    (instrument, q),
                    (instrument, f"{q[0]}{q[-1]}hi"),
                ],
            )
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    text(),
    sampled_from(PPFReader.DDA_METHODS),
    revisions,
    lists(text(), min_size=1, unique=True).map(set),
)
def test_general_get(
    fake_sal, pulse, time_range, error, freq, uid, instrument, revision, quantities
):
    """Test the generic get method to ensure it calls the correct things."""
    with patch.object("indica.reader.PPFReader.get_thomson_scattering"), patch.object(
        "indica.reader.PPFReader.get_charge_exchange"
    ), patch.object("indica.reader.PPFReader.get_equilibrium"), patch.object(
        "indica.reader.PPFReader.get_cyclotron_emissions"
    ), patch.object(
        "indica.reader.PPFReader.get_radiation"
    ), patch.object(
        "indica.reader.PPFReader.get_bremsstrahlung_spectroscopy"
    ):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
        results = reader.get(uid, instrument, revision, quantities)
        assert isinstance(results, MagicMock)
        getattr(reader, reader.DDA_METHODS[instrument]).assert_called_once_with(
            uid, instrument, revision, quantities
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    text(),
    sampled_from(PPFReader.DDA_METHODS),
    revisions,
)
def test_get_defaults(
    fake_sal, pulse, time_range, error, freq, uid, instrument, revision
):
    """Test the generic get method uses appropriate default quantities."""
    with patch("indica.readers.ppfreader.SALClient", fake_sal):
        reader = PPFReader(
            pulse,
            *time_range,
            default_error=error,
            max_freq=freq,
            selector=MagicMock(),
            session=MagicMock(),
        )
    results = reader.get(uid, instrument, revision)
    assert set(results) == set(reader.available_quantities(instrument))
