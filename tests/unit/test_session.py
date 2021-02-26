"""Check PROV/session tracking functionality."""

from pathlib import Path
from unittest import mock

from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import composite
from hypothesis.strategies import dictionaries
from hypothesis.strategies import emails
from hypothesis.strategies import from_regex
from hypothesis.strategies import lists
from hypothesis.strategies import text
import prov.model as prov

import indica
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


@settings(deadline=None)
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


@given(emails())
def test_session_begin(email):
    old_global = session.global_session
    try:
        Session.begin(email)
        assert session.global_session != old_global
        assert session.global_session._user[0].identifier.localpart == email
    finally:
        session.global_session = old_global


@given(
    emails(), lists(from_regex("[a-z0-9]+", fullmatch=True), unique=True, min_size=1)
)
def test_push_pop_agents(email, agent_ids):
    assume(all(ident != email for ident in agent_ids))
    session = Session(email)
    agents = []
    original_agent = session._user[0]
    for ident in agent_ids:
        agents.append(session.prov.agent(ident))
        session.push_agent(agents[-1])
        assert session.agent is agents[-1]
    for agent in agents[::-1]:
        popped = session.pop_agent()
        assert popped is agent
        delegation = next(
            iter(
                filter(
                    lambda x: x.get_attribute("prov:delegate") == {agent.identifier},
                    session.prov.get_records(prov.ProvDelegation),
                )
            )
        )
        assert delegation.get_attribute("prov:responsible") == {
            session.agent.identifier
        }
    assert session.agent is original_agent


@given(emails(), from_regex("[a-z0-9]+", fullmatch=True))
def test_new_agent_context(email, agent_id):
    assume(email != agent_id)
    session = Session(email)
    agent = session.prov.agent(agent_id)
    old_agent = session.agent
    with session.new_agent(agent):
        assert session.agent is agent
        delegation = next(
            iter(
                filter(
                    lambda x: x.get_attribute("prov:delegate") == {agent.identifier},
                    session.prov.get_records(prov.ProvDelegation),
                )
            )
        )
        assert delegation.get_attribute("prov:responsible") == {old_agent.identifier}
    assert session.agent is old_agent


def test_dependency_provenance():
    session = Session("rand.m.person@ukaea.uk")
    dependencies = list(
        filter(
            lambda x: x.get_attribute("prov:generatedEntity")
            == {session.indica_prov.identifier},
            session.prov.get_records(prov.ProvDerivation),
        )
    )
    package_dirs = [
        Path(next(iter(dep.get_attribute("prov:usedEntity"))).localpart)
        for dep in dependencies
    ]
    package_names = [path.name for path in package_dirs]
    assert len(session.prov.get_record("pypi:indica")) == 1
    assert "xarray" in package_names
    assert "scipy" in package_names
    assert "numpy" in package_names
    assert "prov" in package_names
    assert "sal" in package_names
    assert "matplotlib" in package_names
    assert "netCDF4" in package_names
    assert session.prov.get_record("pypi:pandas")
    indica_records = session.prov.get_record(f"local:{Path(indica.__file__).parent}")
    commit = indica_records[0].get_attribute("git_commit")
    assert len(commit) == 1
    assert "UNKNOWN" not in commit


# Test exporting/reloading data produces same result

# Test export/reload properly interacts with data in __main__
