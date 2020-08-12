"""Test reading from SAL database."""

from contextlib import nullcontext
import pathlib
import re
from unittest.mock import MagicMock
from unittest.mock import patch

from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import lists
from hypothesis.strategies import only
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
from hypothesis.strategies import tuples
import numpy as np
import pytest
import sal.client
import sal.core.exceptions
import scipy.constants as sc

from indica.readers import PPFReader
import indica.readers.surf_los
from .fake_salclient import fake_sal_client


@pytest.fixture(scope="module")
def fake_sal():
    """Loads data to create a fake SALClient class."""
    return fake_sal_client(pathlib.Path(__file__).parent.absolute() / "sample_ppfs.pkl")


@pytest.fixture
def patch_sal(fake_sal, monkeypatch):
    """Monkeypatches the SALClient class with the fake version."""
    monkeypatch.setattr(sal.client, "SALClient", fake_sal)


pulses = integers(1, 99999)
times = lists(floats(30.0, 80.0), min_size=2, max_size=2).map(sorted)
errors = floats(0.0001, 0.2)
max_freqs = floats(2.0, 1000.0)
revisions = integers(0)
edited_revisions = lists(revisions, min_size=1, unique=True).map(sorted)
lines_of_sight = tuples(
    arrays(float, 96), arrays(float, 96), arrays(float, 96), arrays(float, 96)
)


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
    if reader.needs_authentication:
        with pytest.raises(sal.core.exceptions.PermissionError):
            reader._get_thomson_scattering("jetppf", "hrts", 0, {"te"})
    else:
        reader._get_thomson_scattering("jetppf", "hrts", 0, {"te"})


@given(pulses, times, errors, max_freqs, text(), text())
def test_authentication(pulse, time_range, error, freq, user, password, fake_sal):
    """Test authentication method on client get called."""
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
    only("jetppf"),
    sampled_from(["hrts", "lidr"]),
    revisions,
    edited_revisions,
    lists(sampled_from(["te", "ne"]), min_size=1, unique=True).map(set),
)
def test_get_thomson_scattering(
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    patch_sal,
):
    """Test quantities returned by _get_thomson_scattering are correct."""
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
    with pytest.raises(sal.core.exceptions.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_thomson_scattering(uid, instrument, revision, quantities)
    if bad_rev:
        return
    z_signal = reader._client.DATA[f"{instrument}/z"]
    assert np.all(z_signal.data == results["z"])
    assert len(z_signal.data) == results["length"]
    assert np.all(z_signal.dimensions[0].data == results["R"])
    records = [get_record(reader, pulse, uid, instrument, "z", revision)]
    for q in quantities:
        assert np.all(results[q] == reader.client.DATA[f"{instrument}/{q}"].data)
        assert np.all(
            results[q + "_error"] == reader.client.DATA[f"{instrument}/d{q}"].data
        )
        assert sorted(results[q + "_records"]) == sorted(
            records
            + map(
                lambda x: get_record(reader, pulse, uid, instrument, x, revision),
                [q, "d" + q],
            )
        )


@given(
    pulses,
    times,
    errors,
    max_freqs,
    only("cgiroud"),
    only("cxg6"),
    revisions,
    edited_revisions,
    lists(sampled_from(["angf", "conc", "ti"]), min_size=1, unique=True).map(set),
)
def test_get_charge_exchange(
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    patch_sal,
):
    """Test quantities returned by _get_charge_exchange are correct."""
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
    with pytest.raises(sal.core.exceptions.NodeNotFound) if bad_rev else nullcontext():
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
        assert np.all(results[q] == reader.client.DATA[f"{instrument}/{q}"].data)
        assert np.all(
            results[q + "_error"] + results[q]
            == reader.client.DATA[f"{instrument}/{uncertainties[q]}"].data
        )
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
    only("jetppf"),
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
                "rsep",
                "vjac",
                "zmag",
                "zsep",
            ]
        ),
        min_size=1,
        unique=True,
    ).map(set),
)
def test_get_equilibrium(
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    patch_sal,
):
    """Test quantities returned by _get_equilibrium are correct."""
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
    with pytest.raises(sal.core.exceptions.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_equilibrium(uid, instrument, revision, quantities)
    if bad_rev:
        return
    signal = reader._client.DATA[f"{instrument}/f"]
    assert np.all(signal.dimensions[1] == results["psin"])
    for q in quantities:
        assert np.all(results[q] == reader.client.DATA[f"{instrument}/{q}"].data)
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
    only("jetppf"),
    only("kk3"),
    only({"te"}),
    revisions,
    edited_revisions,
    lines_of_sight,
)
def test_get_cyclotron_emissions(
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
    patch_sal,
    monkeypatch,
):
    """Test quantities returned by _get_cyclotrons_emissions are correct."""
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
    with monkeypatch.context() as m, pytest.raises(
        sal.core.exceptions.NodeNotFound
    ) if bad_rev else nullcontext():
        m.setattr(
            indica.readers.surf_los,
            "read_surf_los",
            mock_surf=MagicMock(return_value=los),
        )
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    assert results["z"] == los[2][0]
    # TODO: determine how best to describe the SURF data for PROV
    records = [
        "surf_overlays.db",
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
    only("jetppf"),
    only("sxr"),
    revisions,
    edited_revisions,
    lists(sampled_from(["h", "t", "v"]), min_size=1, unique=True).map(set),
    lines_of_sight,
)
def test_get_sxr(
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
    patch_sal,
    monkeypatch,
):
    """Test SXR quantities returned by _get_radiation are correct."""
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
    with monkeypatch.context() as m, pytest.raises(
        sal.core.exceptions.NodeNotFound
    ) if bad_rev else nullcontext():
        m.setattr(
            indica.readers.surf_los,
            "read_surf_los",
            mock_surf=MagicMock(return_value=los),
        )
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    # TODO: determine how best to describe the SURF data for PROV
    records = ["surf_overlays.db"]
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
            assert np.all(
                radiation[:, i] == reader._client.DATA[f"{instrument}/{name}"].data
            )
        assert np.all(results[q + "_Rstart"] == los[0][channel_indices])
        assert np.all(results[q + "_Rstop"] == los[1][channel_indices])
        assert np.all(results[q + "_zstart"] == los[2][channel_indices])
        assert np.all(results[q + "_zstop"] == los[3][channel_indices])
        assert np.all(results[q + "_error"] == pytest.approx(error * radiation))
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
    only("jetppf"),
    only("bolo"),
    revisions,
    edited_revisions,
    lists(sampled_from(["kb5h", "kb5v"]), min_size=1, unique=True).map(set),
    lines_of_sight,
)
def test_get_radiation(
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
    patch_sal,
    monkeypatch,
):
    """Test bolometric quantities returned by _get_radiation are correct."""
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
    with monkeypatch.context() as m, pytest.raises(
        sal.core.exceptions.NodeNotFound
    ) if bad_rev else nullcontext():
        m.setattr(
            indica.readers.surf_los,
            "read_surf_los",
            mock_surf=MagicMock(return_value=los),
        )
        results = reader._get_radiation(uid, instrument, revision, quantities)
    if bad_rev:
        return
    # TODO: determine how best to describe the SURF data for PROV
    records = ["surf_overlays.db"]
    for q in quantities:
        radiation = results[q]
        length = results["length"][q]
        assert length == radiation.shape[1]
        assert np.all(radiation == reader._client.DATA[f"{instrument}/{q}"])
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
    only("jetppf"),
    only("ks3"),
    revisions,
    edited_revisions,
    lists(sampled_from(["zefh", "zefv"]), min_size=1, unique=True).map(set),
)
def test_get_bremsstrahlung_spectroscopy(
    pulse,
    time_range,
    error,
    freq,
    uid,
    instrument,
    revision,
    available_revisions,
    quantities,
    patch_sal,
):
    """Test data returned by _get_bremsstrahlung_spectroscopy is correct."""
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
    with pytest.raises(sal.core.exceptions.NodeNotFound) if bad_rev else nullcontext():
        results = reader._get_charge_exchange(uid, instrument, revision, quantities)
    if bad_rev:
        return
    for q in quantities:
        assert np.all(results[q] == reader._client.DATA[f"{instrument}/{q}"])
        assert np.all(
            results["q"] + results[q + "_error"]
            == pytest.approx(reader._client.DATA[f"{instrument}/{q[0]}{q[-1]}hi"])
        )
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
    pulse, time_range, error, freq, uid, instrument, revision, quantities, patch_sal
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
    pulse, time_range, error, freq, uid, instrument, revision, patch_sal
):
    """Test the generic get method uses appropriate default quantities."""
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
