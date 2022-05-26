"""Provides a fake implementaion of the SALClient, that reads data
from a file on the disk."""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union
from unittest.mock import Mock

import numpy as np
from sal.client import SALClient
from sal.core.exception import InvalidPath
from sal.core.exception import NodeNotFound
from sal.core.object import Branch
from sal.core.object import BranchReport
from sal.core.object import LeafReport
from sal.core.object import ObjectReport
from sal.dataclass import Signal


class FakeSALClient(SALClient):
    """Fake implementation of the SALClient class. It only implements the
    subset of functionality used by the
    :py:class:`indica.readers.PPFReader` class.

    """

    def __init__(self, url, verify_https_cert=True):
        self.url = url
        self._blacklist = []
        self._revisions = [1]
        self.authenticate = Mock(return_value=True)
        with open(self.data_file, "r") as f:
            self.data: Dict[str, Tuple[int]] = json.load(f)
        self.constructed_data: Dict[str, Signal] = {
            key: Signal(
                data=np.random.randn(*data_shape),
                dimensions=[np.random.random(size) for size in data_shape],
            )
            for key, data_shape in self.data.items()
        }

    @property
    def data_file(self) -> Union[str, Path]:
        raise NotImplementedError(
            "Data source file not implemented, use subclass of FakeSALClient"
        )

    def get(self, path, summary=False):
        """Gets the node at the specified path."""
        if len(self.data) == 0:
            raise UserWarning("Data not loaded")
        path_components = path.split("/")
        if path_components[-1] and len(path_components) == 8:
            # Leaf node
            dda = path_components[-2]
            dtype = path_components[-1].split(":")[0]
            key = f"{dda}/{dtype}"
            if key in self._blacklist or key not in self.constructed_data:
                raise NodeNotFound(f"Node {path} does not exist.")
            signal = self.constructed_data[key]
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
            int(path_components[-1].split(":")[1]) if ":" in path_components[-1] else 0
        )
        if revision == 0:
            revision = self._revisions[-1]
        elif revision < 0:
            raise InvalidPath(
                f"The supplied path {path} does not conform to the data system"
                " path specification."
            )
        elif revision < self._revisions[0]:
            raise NodeNotFound(f"Node {path} does not exist.")
        else:
            revision = list(filter(lambda r: r <= revision, self._revisions))[-1]
        key = f"{path_components[-2]}/{path_components[-1].split(':')[0]}"
        if len(path_components) < 6:
            return Mock()
        elif len(path_components) == 6:
            return BranchReport(
                f"Branch {path}",
                list({key.split("/")[0] for key in self.constructed_data}),
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
                        key.split("/")[1],
                        ObjectReport(value.CLASS, value.GROUP, value.VERSION),
                    )
                    for key, value in self.constructed_data.items()
                    if key.split("/")[0] == path_components[-1]
                ],
                datetime.now(),
                revision,
                self._revisions[-1],
                self._revisions,
            )
        elif len(path_components) == 8 and key in self.constructed_data:
            node = self.constructed_data[key]
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
