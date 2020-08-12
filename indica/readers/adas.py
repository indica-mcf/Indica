"""Base class for reading in ADAS atomic data."""

from abc import abstractmethod
import datetime
from typing import Dict
from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
from xarray import Dataset

from ..abstractio import BaseIO
from ..datatypes import ArrayType
from ..session import global_session
from ..session import hash_vals
from ..session import Session


# TODO: Evaluate this location
DEFAULT_PATH: str = ""


class ADASReader(BaseIO):
    """Class for reading atomic data from ADAS files.0

    Parameters
    ----------
    path: str
        Location from which relative paths should be evaluated.
        Default is installation directory containing distributed
        data files.
    sesssion: Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    def __init__(self, path: str = DEFAULT_PATH, sess: Session = global_session):
        self.path = path
        self.session = sess
        self.openadas = path == DEFAULT_PATH
        self.session.prov.add_namespace("adas", "https://open.adas.ac.uk/detail/adf11/")
        self.session.prov.add_namespace("localfile", "file://")
        self.prov_id = hash_vals(path=path)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.actedOnBehalfOf(self.agent, self.session.agent)
        self.entity = self.session.prov.entity(self.prov_id, {"path": path})
        self.session.prov.generation(
            self.entity, self.session.session, time=datetime.datetime.now()
        )
        self.session.prov.attribution(self.entity, self.session.agent)

    def close(self):
        """Closes connections to database/files. For this class it does not
        need to do anything."""
        pass

    def get(self, filename: str) -> Dataset:
        """Read data from the specified ADAS file.

        Parameters
        ----------
        filename
            The ADF11 file to read.

        Returns
        -------
        :
            The data in the specified file. Dimensions are density and
            temperature. Each members of the dataset correspond to a
            different charge state.

        """

    @abstractmethod
    def _get(
        self, absolute_path: str
    ) -> Tuple[Dict[Union[int, str], np.ndarray], ArrayType]:
        """Parse the ADAS file, returning its data in a dictionary.

        Parameters
        ----------
        absolute_path
            The path to the file to be parsed.

        Returns
        -------
        values : Dict[Union[int, str], np.ndarray]
            A dictionary containing the following items:

            densities
                The densities at which the data is provided
            temperatures
                The temperatures at which the data is provided
            _int_
                The ADAS data for this charge state

        data : DataType
            The type of data being read in (quantity and element)
        """
        raise NotImplementedError(
            "{} does not implement a '_get' " "method.".format(self.__class__.__name__)
        )

    @property
    def requires_authentication(self) -> Literal[False]:
        """Reading ADAS data never requires authentication."""
        return False
