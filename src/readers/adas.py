"""Function for reading in ADAS atomic data in ADF11 format."""

import datetime

from xarray import Dataset

from ..session import global_session
from ..session import hash_vals
from ..session import Session


# TODO: Evaluate this location
DEFAULT_PATH: str = ""


class ADASReader:
    """Class for reading atomic data from ADAS files in ADF11 format.

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

    def get(self, filename: str) -> Dataset:
        """Read the specified ADF11 ADAS file.

        Parameters
        ----------
        filename
            The ADF11 file to read.

        Returns
        -------
            The data in the specified file. Dimensions are density and
            temperature. Each members of the dataset correspond to a
            different charge state.

        """
