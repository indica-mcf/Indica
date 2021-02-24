"""Check PROV/session tracking functionality."""

from unittest import mock

from hypothesis import given
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import emails
from hypothesis.strategies import lists
from hypothesis.strategies import text
import prov.model as prov

from indica import session
from indica.session import hash_vals
from indica.session import Session


@composite
def sessions(draw):
    """Generates :py:class:`indica.session.Session` objects."""
    # TODO: add option to draw ORCID identifiers
    return Session(draw(emails()))


@given(lists(dictionaries(text(), text()), min_size=2, max_size=2, unique_by=str))
def test_hash_vals_unique(inputs):
    hash1 = hash_vals(**inputs[0])
    hash2 = hash_vals(**inputs[1])
    assert hash1 != hash2


@given(text(), text(), text(), text(), emails())
def test_session_initialises_prov(os, directory, host, python, email):
    from indica.session import platform

    with mock.patch.multiple(
        platform, node=lambda: host, platform=lambda: os, python_version=lambda: python
    ), mock.patch("os.getcwd", lambda: directory):
        session = Session(email)
    associations = list(session.prov.get_records(prov.ProvAssociation))
    assert len(associations) == 1
    assert associations[0].get_attribute("prov:activity") == {
        session.session.identifier
    }
    assert associations[0].get_attribute("prov:agent") == {session._user[0].identifier}
    assert session._user[0].identifier.localpart == email
    assert session._user[0].identifier.namespace.uri == "https://ccfe.ukaea.uk/"
    assert session.session.get_attribute("os") == {os}
    assert session.session.get_attribute("directory") == {directory}
    assert session.session.get_attribute("host") == {host}
    assert session.session.get_attribute("python") == {python}
    # Test dependency data


def test_session_initialise_orcid():
    orcid = "0000-0000-0000-0000"
    session = Session(orcid)
    assert session._user[0].identifier.localpart == orcid
    assert session._user[0].identifier.namespace.uri == "https://orcid.org/"


def test_session_context_manager():
    new_session = Session("rand.m.person@ukaea.uk")
    old_session = session.global_session
    assert new_session != old_session
    with new_session:
        assert session.global_session == new_session
    assert session.global_session == old_session


# Check session behaves properly as context manager

# Try pushing/popping/accessing PROV agents (RuleBasedStateMachine)

# Test begin() factory method

# Test exporting/reloading data produces same result

# Test export/reload properly interacts with data in __main__
