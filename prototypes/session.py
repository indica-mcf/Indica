"""Experimental design for handling provenance using W3C PROV.

"""

import datetime
import hashlib
import typing

import prov.model as prov

__author__ = "Marco Sertoli"
__credits__ = ["Chris MacMackin", "Marco Sertoli"]


global_session = None


def get_dependency_data():
    """A generator for provenance data on dependencies."""
    raise NotImplementedError("TODO: write this function")


def hash_vals(**kwargs: typing.Dict[str, typing.Any]) -> str:
    """Produces an SHA256 hash from the key-value pairs passed as
    arguments.

    Parameters
    ---------
    kwargs
        The data to use for the hash.

    Returns
    -------
    str
        A hexadecimal representation of the hash.
    """
    hash_result = hashlib.sha256()
    for key, val in kwargs:
        hash_result.update(bytes(key, encoding="utf-8"))
        hash_result.update(b":")
        hash_result.update(bytes(str(val), encoding="utf-8"))
        hash_result.update(b",")
    return hash_result.hexdigest()


class Session:
    """A class handling information about the use of the software. Mostly
    this is for purposes of provenance.

    Parameters
    ----------
    user_orcid: str
        The ORCiD ID for the person using the software.

    Attributes
    ----------
    prov: prov.model.ProvDocument
        The document containing all of the provenance information for this
        session.
    session: prov.model.ProvActivity
        The provenance Activity object representing this session. It should
        contain information about versions of different libraries being used.

    """

    def __init__(self, user_orcid: str):
        self.prov = prov.ProvDocument()
        self.prov.set_default_namespace('https://ccfe.ukaea.uk/')
        self.prov.add_namespace('orcid', 'https://orcid.org/')
        self._user = [self.prov.agent('orcid:' + user_orcid)]
        date = datetime.datetime.now()
        session_properties = {'os': None, 'directory': None, 'host': None}
        session_id = hash_vals(startTime=date, **session_properties)
        self.session = self.prov.activity(session_id, date, None,
                                          session_properties)
        self.prov.association(self.session, self._user[0])

    def __enter__(self):
        global global_session
        self.old_global_session = global_session
        global_session = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global global_session
        global_session = self.old_global_session
        return False

    @property
    def agent(self) -> prov.ProvAgent:
        """The agent (person or piece of software) currently in immediate
        control of execution.

        :returntype: prov.model.ProvAgent

        """
        return self._user[-1]

    def push_agent(self, agent: prov.ProvAgent):
        """Delegate responsibility to another agent.

        They will appear to be in control of execution now and will be
        returned by the :py:meth:`agent` property.

        Parameters
        ----------
        agent
            The new agent to delegate responsibilityt to.

        """
        agent.actedOnBehalfOf(self._user[-1])
        self._user.append(agent)

    def pop_agent(self) -> prov.ProvAgent:
        """Take responsibility back from the Agent that it was most recently
        delegated to.

        The Agent which the responsibility was delegated by will now
        appear to be in control of execution and will be the one
        returned by the :py:meth:`agent` property.

        Returns
        -------
        prov.ProvAgent
            The agent that responsibility was taken away from.

        """
        return self._user.pop()

    @classmethod
    def begin(cls, user_orcid: str):
        """Sets up a global session, without bothering with a context
        manager.

        Parameters
        ----------
        user_orcid
            The ORCiD ID for the person using the software.

        """
        global global_session
        global_session = cls(user_orcid)
