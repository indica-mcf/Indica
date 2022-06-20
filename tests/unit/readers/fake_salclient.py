"""Provides a fake implementaion of the SALClient, that reads data
from a file on the disk."""

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from hashlib import sha1
import json
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng
from sal.client import SALClient
from sal.core.exception import InvalidPath
from sal.core.exception import NodeNotFound
from sal.core.object import Branch
from sal.core.object import BranchReport
from sal.core.object import LeafReport
from sal.core.object import ObjectReport
from sal.dataclass import Signal
from sal.dataclass.signal.dimension import ArrayDimension


class BaseFakeSALClient(SALClient, ABC):
    """
    Base fake implementation of the SALClient class. It only implements the
    subset of functionality used by the
    :py:class:`indica.readers.PPFReader` class.

    Reads data from data_file, subclass and assign this to a json for testing.
    """

    def __init__(self, url, verify_https_cert=True):
        self.url = url
        self._blacklist = []
        self._revisions = [1]
        self.authenticate = Mock(return_value=True)

        with open(self.data_file, "r") as f:
            self.data_specs: Dict[str, Tuple[int]] = json.load(f)

    @property
    @abstractmethod
    def data_file(self) -> Union[str, Path]:
        raise NotImplementedError(
            "Data source file not implemented, use subclass of BaseFakeSALClient"
        )

    def construct_signal(self, quantity):
        spec = self.data_specs[quantity]
        if "data" in spec:
            # data is stored e.g. cxg6/zqnn and cxg6/mass
            generated_data = spec["data"]
        else:
            # data not stored, generate it based on quantity name
            generated_data = BaseFakeSALClient.generate_data(
                quantity,
                tuple(spec["shape"]),
                spec["dtype"],
                spec["min"],
                spec["max"],
                self.data_file,
            )

        dimensions = [ArrayDimension(np.linspace(0, 1, size)) for size in spec["shape"]]
        return Signal(dimensions, generated_data)

    @staticmethod
    def generate_data(quantity, shape, dtype, min_val, max_val, data_file):
        # Set the PRNG state using the hash of the diagnotic/dda string
        # Will reproduce the same data/dimensions for the same pair
        hash = sha1(quantity.encode("utf-8"))
        rng = default_rng(int.from_bytes(hash.digest()[-4:], "big"))

        if dtype == "float64":
            return rng.uniform(size=shape, low=min_val, high=max_val)
        elif dtype == "int64":
            return rng.integers(size=shape, low=min_val, high=max_val)
        else:
            raise ValueError(f"dtype {dtype} in {data_file} unrecognised.")

    def get(self, path, summary=False):
        """Gets the node at the specified path."""
        if len(self.data_specs) == 0:
            raise UserWarning("Data not loaded")

        path_components = path.split("/")
        if path_components[-1] and len(path_components) == 8:
            # Leaf node
            dda = path_components[-2]
            dtype = path_components[-1].split(":")[0]
            key = f"{dda}/{dtype}"
            if key in self._blacklist or key not in self.data_specs:
                raise NodeNotFound(f"Node {path} does not exist.")
            signal = self.construct_signal(key)
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
                list({key.split("/")[0] for key in self.data_specs}),
                [],
                datetime.now(),
                revision,
                self._revisions[-1],
                self._revisions,
            )
        elif len(path_components) == 7:
            leaves = []
            for key in self.data_specs:
                if key.split("/")[0] == path_components[-1]:
                    signal = self.construct_signal(key)
                    leaves.append(
                        (
                            key.split("/"),
                            ObjectReport(signal.CLASS, signal.GROUP, signal.VERSION),
                        )
                    )

            return BranchReport(
                f"Branch {path}",
                [],
                leaves,
                datetime.now(),
                revision,
                self._revisions[-1],
                self._revisions,
            )
        elif len(path_components) == 8 and key in self.data_specs:
            node = self.construct_signal(key)
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
