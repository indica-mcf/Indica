from typing import List
from typing import Tuple

from MDSplus import Connection
import numpy as np

from indica import BaseIO
from ..numpy_typing import RevisionLike


class MDSError(Exception):
    """An exception which occurs when trying to read MDS+ data which would
    not be caught by the lower-level MDSplus library. An example would be
    failing to find any valid channels for an instrument when each channel
    is a separate DTYPE.

    """


class MDSWarning(UserWarning):
    """A warning that occurs while trying to read MDS+ data. Typically
    related to caching in some way.

    """


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
        self.conn: Connection = Connection(server)
        self.conn.openTree(self.tree, self.pulse)

    def close(self) -> None:
        del self.conn

    @property
    def requires_authentication(self) -> bool:
        return False

    def get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, str]:
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""

        path, path_check = self.get_mds_path(uid, instrument, quantity, revision)
        if quantity.lower() == ":best_run":
            data = str(self.conn.get(path))
        else:
            data = np.array(self.conn.get(path))

        return data, path

    def get_signal_dims(
        self,
        mds_path: str,
        ndims: int,
    ) -> Tuple[List[np.array], List[str]]:
        """Gets the dimensions of a signal given the path to the signal
        and the number of dimensions"""

        dimensions = []
        paths = []
        for dim in range(ndims):
            path = f"dim_of({mds_path},{dim})"
            dim_tmp = self.conn.get(path).data()

            paths.append(path)
            dimensions.append(np.array(dim_tmp))
        return dimensions, paths

    def get_signal_units(
        self,
        mds_path: str,
    ) -> str:
        """Gets the units of a signal given the path to the signal
        and the number of dimensions"""

        path = f"units_of({mds_path})"
        unit = self.conn.get(path).data()

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

    def revision_name(self, revision: RevisionLike) -> str:
        """Return string defining RUN## or BEST if revision = 0"""

        if type(revision) == int:
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

    def get_best_revision(self, uid: str, instrument: str):
        """
        Return revision name to which BEST is pointing to
        """
        best_revision, _ = self.get_signal(uid, instrument, ".best_run", "best")
        return best_revision

    def get_revision(self, uid: str, instrument: str, revision: RevisionLike) -> str:
        """
        Return revision name given
        """
        revision_name = self.revision_name(revision)
        if revision_name == "BEST":
            revision_name = self.get_best_revision(uid, instrument)

        return revision_name

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

        if len(instrument) > 0 and instrument.upper() != self.tree.upper()
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
