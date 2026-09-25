from typing import List
from typing import Tuple

import mdsthin
from mdsthin import TreeNNF
import numpy as np

from indica import BaseIO
from ..numpy_typing import RevisionLike


# this will be baseio class instead. what is defauly pulse?
class MDSUtils(BaseIO):
    def __init__(
        self,
        pulse,
        server: str = "smaug",
        tree: str = "ST40",
    ):
        self.tree: str = tree
        self.pulse: int = pulse
        self.conn: mdsthin.Connection = mdsthin.Connection(server)
        self.conn.openTree(self.tree, self.pulse)

    def close(self) -> None:
        self.conn.disconnect()

    @property
    def requires_authentication(self) -> bool:
        return False

    def _read_mds_value(self, path: str):
        value = self.conn.get(path)
        if hasattr(value, "data"):
            return value.data()
        return value

    def get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, str]:
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""
        path, path_check = self.get_mds_path(uid, instrument, quantity, revision)
        _data = self._read_mds_value(path)

        if quantity.lower() == ":best_run":
            data = str(_data)
        else:
            data = np.array(_data)

        return data, path

    def get_signal_dims(
        self,
        mds_path: str,
        ndims: int,
    ) -> Tuple[List[np.array], List[str]]:
        """
        Gets the dimensions of a signal given the path to the signal
        and the number of dimensions
        TODO: try/except is required if data not written to MDS+ with dimensions
        """

        dimensions = []
        paths = []
        for dim in range(ndims):
            path = f"dim_of({mds_path},{dim})"
            try:
                _dimension = np.array(self._read_mds_value(path))
            except Exception as e:
                _dimension = None
                print(f"No dimensions for {mds_path}: {e}")

            paths.append(path)
            dimensions.append(_dimension)
        return dimensions, paths

    def get_signal_units(
        self,
        mds_path: str,
    ) -> str:
        """Gets the units of a signal given the path to the signal
        and the number of dimensions"""

        path = f"units_of({mds_path})"
        unit = self._read_mds_value(path)

        return unit

    def get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array], str, str]:
        """Gets the signal and its coordinates for the given INSTRUMENT, at the
        given revision."""
        data, _path = self.get_signal(uid, instrument, quantity, revision)
        dims, _ = self.get_signal_dims(_path, len(data.shape))
        unit = self.get_signal_units(_path)

        return data, dims, unit, _path

    def revision_name(self, revision: RevisionLike) -> RevisionLike:
        """Return string defining RUN## or BEST if revision = 0"""

        if isinstance(revision, int):
            _revision = int(revision)
            if _revision < 0:
                rev_str = ""
            elif _revision == 0:
                rev_str = "best"
            elif _revision < 10:
                rev_str = f"run0{int(_revision)}"
            else:
                rev_str = f"run{int(_revision)}"
        else:
            rev_str = f"{revision}"

        return rev_str.upper()

    def get_best_revision(
        self,
        uid: str,
        instrument: str,
        revision_name: str = "best",
    ):
        """
        Return revision name to which BEST is pointing to
        """
        best_revision, _ = self.get_signal(uid, instrument, ".best_run", revision_name)
        return best_revision

    def get_revision(
        self, uid: str, instrument: str, revision: RevisionLike
    ) -> tuple[RevisionLike, bool]:
        """
        Return revision name given
        """
        revision_name = self.revision_name(revision)
        is_best = False
        if "BEST" in revision_name:
            try:
                revision_name = self.get_best_revision(uid, instrument, revision_name)
                is_best = True
            except TreeNNF:
                is_best = False

        return revision_name, is_best

    def get_mds_path(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[str, str]:
        """Return the path in the MDS+ database to for the given INSTRUMENT/CODE

        uid: currently redundant --> set to empty string ""
        instrument: e.g. "efit"
        quantity: e.g. ".global:cr0" # minor radius
        revision: if 0 --> looks for "best", else "run##"
        """
        revision_name = self.revision_name(revision)
        mds_path = ""
        if len(uid) > 0:
            mds_path += f".{uid}".upper()
        if len(instrument) > 0 and instrument.upper() != self.tree.upper():
            mds_path += f".{instrument}".upper()
        mds_path += f".{revision_name}{quantity}".upper()
        return mds_path, self.mdsCheck(mds_path)

    def mdsCheck(self, mds_path):
        """Return FAILED if node doesn't exist or other error
        Return FAILED if: lenght(data)==1 and data==nan"""

        mds_path_test = (
            f"_dummy = IF_ERROR (IF ((SIZE ({mds_path})==1), "
            + f'IF ({mds_path}+1>{mds_path}, {mds_path}, "FAILED"),'
            + f' {mds_path}), "FAILED")'
        )

        return mds_path_test
