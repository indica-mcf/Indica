from MDSplus import Connection
from MDSplus.mdsExceptions import TreeNNF
from MDSplus.mdsExceptions import TreeNODATA

from .abstractreader import DataReader
from .. import session


from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from ..numpy_typing import RevisionLike

import numpy as np



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

class MDSReader(DataReader): 
        
    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "smaug",
        tree: str = "ST40",
        default_error: float = 0.05,
        max_freq: float = 1e6,
        session: session.Session = session.global_session,
    ): 
        self.tree: str = tree
        self.conn: Connection = Connection(server)
        self.conn.openTree(self.tree, self.pulse)




    def get_mds_path_dims(self, mds_path: str, dim: int):
        """Gets the dimensions' path given an mds_path"""

        dims_path = f"dim_of({mds_path},{dim})"
        return dims_path
    

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, str]:
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""
        path, path_check = self.get_mds_path(uid, instrument, quantity, revision)
        if quantity.lower() == ":best_run":
            data = str(self.conn.get(path))
        else:
            data = np.array(self.conn.get(path))
            # data = np.array(self.conn.get(path_check))

        return data, path
    
    def _get_signal_dims(
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
            dim_tmp = self.conn.get(self.mdsCheck(path)).data()

            paths.append(path)
            dimensions.append(np.array(dim_tmp))
        return dimensions, paths
    

    def _get_signal_units(
        self,
        mds_path: str,
    ) -> str:
        """Gets the units of a signal given the path to the signal
        and the number of dimensions"""

        path = f"units_of({mds_path})"
        unit = self.conn.get(path).data()

        return unit
    
    def get_mds_path(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[str, str]:
        """Return the path in the MDS+ database to for the given INSTRUMENT/CODE

        uid: currently redundant --> set to empty string ""
        instrument: e.g. "efit"
        quantity: e.g. ".global:cr0" # minor radius
        revision: if 0 --> looks for "best", else "run##"
        """
        revision_name = self.get_revision_name(revision)
        mds_path = ""
        if len(uid) > 0:
            mds_path += f".{uid}".upper()
        if len(instrument) > 0 and instrument.upper() != self.tree.upper():
            mds_path += f".{instrument}".upper()
        mds_path += f"{revision_name}{quantity}".upper()
        return mds_path, self.mdsCheck(mds_path)

    def get_mds_path_dims(self, mds_path: str, dim: int):
        """Gets the dimensions' path given an mds_path"""

        dims_path = f"dim_of({mds_path},{dim})"
        return dims_path