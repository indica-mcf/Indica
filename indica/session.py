"""Experimental design for handling provenance using W3C PROV.

"""

from contextlib import contextmanager
from contextlib import redirect_stderr
import datetime
from functools import wraps
import hashlib
import importlib
import io
import os
from pathlib import Path
import platform
import re
import subprocess
import typing

import pkg_resources
import prov.model as prov
from xarray import DataArray
from xarray import Dataset

from .utilities import positional_parameters

if typing.TYPE_CHECKING:
    from .readers import DataReader
    from .equilibrium import Equilibrium
    from .operators import Operator

__author__ = "Marco Sertoli"
__credits__ = ["Chris MacMackin", "Marco Sertoli"]


ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{4}$")

global_session: "Session"


def get_dependency_data():
    """A generator for provenance data on dependencies."""
    raise NotImplementedError("TODO: write this function")


def hash_vals(**kwargs: typing.Any) -> str:
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
    # TODO: include date/time in hash
    hash_result = hashlib.sha256()
    for key, val in kwargs.items():
        hash_result.update(bytes(key, encoding="utf-8"))
        hash_result.update(b":")
        hash_result.update(bytes(str(val), encoding="utf-8"))
        hash_result.update(b",")
    return hash_result.hexdigest()


def package_provenance(
    doc: prov.ProvDocument, package_name: str
) -> typing.Tuple[prov.ProvEntity, prov.ProvEntity]:
    """Returns provenance for the requested package. This provenance will
    include version information for all dependencies. Returns a tuple
    of the provenance for the package in general and the specific
    installation being used here.

    """
    doc.add_namespace("pypi", "https://pypi.org/project/")
    doc.add_namespace("local", "file://")
    package = pkg_resources.working_set.find(
        pkg_resources.Requirement.parse(package_name)
    )
    assert isinstance(package, pkg_resources.Distribution)
    general_entity = doc.entity(
        f"pypi:{package.project_name}",
        {"pypi:package": package.project_name},
    )
    version_entity = doc.entity(
        f"pypi:{package.project_name}/{package.version}",
        {"pypi:version": package.version},
    )
    version_entity.specializationOf(general_entity)
    # Some modules print things when imported, so capture this
    tmp_output = io.StringIO()
    try:
        with redirect_stderr(tmp_output), redirect_stderr(tmp_output):
            path = Path(importlib.import_module(package.project_name).__file__).parent
        # Check this directory and the parent directory for git repository
        # TODO: Check all parent directories, but only if the child directory
        # is not ignored.
        # if any((p / ".git").exists() for p in [path] + path.parents):
        if (path / ".git").exists() or (path.parent / ".git").exists():
            git_hash = subprocess.check_output(
                ["git", "describe", "--always"], cwd=path, text=True
            ).strip()
            git_diff = subprocess.check_output(
                ["git", "diff", "HEAD", "--", "indica"], text=True
            ).strip()
            if len(git_diff) > 0:
                git_hash += "-dirty"
        elif (path / "git_version").exists():
            with (path / "git_version").open() as f:
                git_hash = f.read()
        else:
            git_hash = "UNKNOWN"
    except ModuleNotFoundError:
        path = Path(package.location) / package.project_name
        git_hash = "UNKNOWN"
    except Exception:
        tmp_output.seek(0)
        print(tmp_output.read())
        raise
    installed_entity = doc.entity(
        f"local:{path}", {"host": platform.node(), "git_commit": git_hash}
    )
    installed_entity.specializationOf(version_entity)
    for dep in package.requires():
        dep_general_entity, dep_installed_entity = package_provenance(
            doc, dep.project_name
        )
        version_entity.wasDerivedFrom(dep_general_entity)
        installed_entity.wasDerivedFrom(dep_installed_entity)
    return general_entity, installed_entity


class Session:
    """Manages the a particular run of the software.

    Has the following uses:
    - keep information about version of package and dependencies
    - hold provenance information
    - track the data read/calculated and operators instantiated
    - allow that data to be exported and reloaded

    TODO: Consider whether some of these behaviours should be spun off
    into separate classes which are then aggregated into this one.

    Parameters
    ----------
    user_id: str
        Something with which to identify the user. Recommend either an email
        address or an ORCiD ID.

    Attributes
    ----------
    data: typing.Dict[str, DataArray]
        All of the data which has been read in or calculated during this
        session.
    equilibria: typing.Dict[str, Equilibrium]
        All of the equilibrium objects which have been created during this
        session.
    operators: typing.Dict[str, AbstractOperator]
        All of the operators which have been instantiated during this session.
    prov: prov.model.ProvDocument
        The document containing all of the provenance information for this
        session.
    readers: typing.Dict[str, DataReader]
    session: prov.model.ProvActivity
        The provenance Activity object representing this session. It should
        contain information about versions of different libraries being used.

    """

    def __init__(self, user_id: str):
        self.prov = prov.ProvDocument()
        self.prov.set_default_namespace("https://ccfe.ukaea.uk/")
        if ORCID_RE.match(user_id):
            self.prov.add_namespace("orcid", "https://orcid.org/")
            self._user = [self.prov.agent("orcid:" + user_id)]
        else:
            self._user = [
                self.prov.agent(user_id if user_id else "example@example.com")
            ]
        date = datetime.datetime.now()
        session_properties = {
            "os": platform.platform(),
            "directory": os.getcwd(),
            "host": platform.node(),
            "python": platform.python_version(),
        }
        session_id = hash_vals(startTime=date, **session_properties)
        self.session = self.prov.activity(session_id, date, None, session_properties)
        # Use an empty ID to short-circuit all of the provenance
        # calculation. This is useful to prevent provenance being
        # built whenever this module is imported.
        if user_id != "":
            self.indica_prov = package_provenance(self.prov, "indica")[1]
            self.session.used(self.indica_prov)
        self.prov.association(self.session, self._user[0])

        self.data: typing.Dict[str, typing.Union[DataArray, Dataset]] = {}
        self.equilibria: typing.Dict[str, Equilibrium] = {}
        self.operators: typing.Dict[str, Operator] = {}
        self.readers: typing.Dict[str, DataReader] = {}

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

    @contextmanager
    def new_agent(self, agent: prov.ProvAgent) -> prov.ProvAgent:
        """A context manager for temporarily adding an agent to the
        session. This is useful to ensure the agent will be removed even if
        there is an exception thrown.

        """
        self.push_agent(agent)
        try:
            yield agent
        finally:
            self.pop_agent()

    def export(self, filename: str):
        """Write all of the data and operators from this session into a file,
        for reuse later.
        """
        pass

    @classmethod
    def begin(cls, user_id: str):
        """Sets up a global session, without bothering with a context
        manager.

        Parameters
        ----------
        user_id
            An identifier, such as an email address or ORCiD ID, for the person
            using the software.

        """
        global global_session
        global_session = cls(user_id)

    @classmethod
    def reload(cls, filename: str) -> "Session":
        """Create a session from a saved which was written to
        ``filename``. Thanks to some Python voodoo, any local
        variables in ``__main__`` will be recreated.

        """
        pass


global_session = Session("")


def generate_prov(pass_sess: bool = False):
    """Decorator to be applied to functions generating
    :py:class:`xarray.DataArray` output. It will produce PROV data and
    attach it as an attribute.

    This should only be applied to stateless functions, as the PROV
    data it generates will not accurately describe anything else.

    Parameters
    ----------
    pass_sess
        Indicates whether, if a keyword argument called ``sess`` is present,
        it should be passed to ``func``.

    """

    def outer_wrapper(func):
        param_names, var_positional = positional_parameters(func)
        num_positional = len(param_names)

        @wraps(func)
        def prov_generator(*args, **kwargs):
            session = kwargs.get("sess", global_session)
            if "sess" in kwargs and not pass_sess:
                kwargs = dict(kwargs)
                del kwargs["sess"]
            start_time = datetime.datetime.now()

            result = func(*args, **kwargs)

            end_time = datetime.datetime.now()
            args_prov = []
            activity_attrs = {prov.PROV_TYPE: func.__name__}
            id_attrs = {}
            for i, arg in enumerate(args):
                if i < num_positional:
                    argname = param_names[i]
                else:
                    argname = var_positional + str(i - num_positional)
                if isinstance(arg, DataArray):
                    args_prov.append(arg.attrs["provenance"])
                    id_attrs[argname] = args_prov[-1].identifier
                else:
                    args_prov.append(str(arg))
                    activity_attrs[argname] = str(arg)
            for key, val in kwargs.items():
                if isinstance(arg, DataArray):
                    args_prov.append(val.attrs["provenance"])
                    id_attrs[key] = args_prov[-1].identifier
                else:
                    args_prov[key] = str(key)
                    activity_attrs[val] = str(arg)
            generated_array = False
            activity_id = hash_vals(agent=session.agent, date=end_time, **id_attrs)
            activity = session.prov.activity(
                activity_id, start_time, end_time, activity_attrs
            )
            if isinstance(result, DataArray):
                entity_id = hash_vals(
                    activity=activity_id, name=result.name, **id_attrs
                )
                entity = session.prov.entity(entity_id)
                entity.wasGeneratedBy(activity, end_time)
                entity.wasAttributedTo(session.agent)
                result.attrs["provenance"] = entity
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if isinstance(r, DataArray):
                        entity_id = hash_vals(
                            activity=activity_id,
                            position=str(i),
                            name=r.name,
                            **id_attrs,
                        )
                        entity = session.prov.entity(entity_id)
                        entity.wasGeneratedBy(activity, end_time)
                        entity.wasAttributedTo(session.agent)
                        r.attrs["provenance"] = entity
            if not generated_array:
                raise ValueError(
                    "No DataArray object was produced by the "
                    "function. Can not assign PROV data."
                )
            return result

        return prov_generator

    return outer_wrapper
