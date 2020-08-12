"""Provides a fake implementaion of the SALClient, that reads data
from a file on the disk."""

from datetime import datetime
import pickle
from unittest.mock import Mock

from sal.client import SALClient
from sal.core.exception import NodeNotFound
from sal.core.object import Branch
from sal.core.object import BranchReport
from sal.core.object import LeafReport
from sal.core.object import ObjectReport


def fake_sal_client(datafile):
    """Create a fake SALClient class which returns data contained in
    ``datafile``. ``datafile`` should contain a pickled dictionary mapping
    keys with the format "dda/dtype" to SAL Signal objects.

    """
    with open(datafile, "rb") as df:

        class FakeSALClient(SALClient):
            """Fake implementation of the SALClient class. It only implements the
            subset of functionality used by the
            :py:class:`indica.readers.PPFReader` class.

            """

            DATA = pickle.load(df)

            def __init__(self, url, verify_https_cert=True):
                self.url = url
                self._blacklist = []
                self._revisions = [1]

            authenticate = Mock()

            def get(self, path, summary=False):
                """Gets the node at the specified path."""
                path_components = path.split("/")
                if path_components[-1] and len(path_components) == 8:
                    # Leaf node
                    dda = path_components = [-2]
                    dtype = path_components[-1].split(":")[0]
                    key = f"{dda}/{dtype}"
                    if key in self._blacklist or key not in self.DATA:
                        raise NodeNotFound(f"Node {path} does not exist.")
                    signal = self.DATA[key]
                    return signal.summary() if summary else signal
                else:
                    # Branch node
                    if len(path_components) >= 8 or not path_components[0]:
                        raise NodeNotFound(f"Node {path} does not exist.")
                    return Branch(f"Branch {path}")

            def list(self, path):
                """Returns node meta data for the specified path."""
                path_components = path.split("/")
                revision = (
                    int(path_components[-1].split(":")[1])
                    if ":" in path_components[-1]
                    else 0
                )
                if revision == 0:
                    revision = self._revisions[-1]
                elif revision < self._revisions[0]:
                    raise NodeNotFound(f"Node {path} does not exist.")
                else:
                    revision = next(filter(lambda r: r <= revision, self._revisions))
                key = "/".join(path_components[-2:])
                if len(path_components) < 6:
                    return Mock()
                elif len(path_components) == 6:
                    return BranchReport(
                        f"Branch {path}",
                        list({key.split("/")[0] for key in self.DATA}),
                        [],
                        datetime.now(),
                        revision,
                        self._revisions[-1],
                        self._revisions,
                    )
                elif len(path_components) == 7:
                    return BranchReport(
                        f"Branch {path}",
                        [],
                        [
                            (
                                key.split["/"][1],
                                ObjectReport(value.CLASS, value.GROUP, value.VERSION),
                            )
                            for key, value in self.DATA.items()
                            if key.split["/"][0] == path_components[-1]
                        ],
                        datetime.now(),
                        revision,
                        self._revisions[-1],
                        self._revisions,
                    )
                elif len(path_components) == 8 and key in self.DATA:
                    node = self.DATA[key]
                    return LeafReport(
                        f"Leaf {path}",
                        node.CLASS,
                        node.GROUP,
                        node.VERSION,
                        datetime.now(),
                        revision,
                        self._revisions[-1],
                        self._revisions,
                    )
                else:
                    raise NodeNotFound(f"Node {path} does not exist.")

    return FakeSALClient
