"""Base class for reading in ADAS atomic data."""

import datetime
from pathlib import Path
from typing import Literal
from typing import TextIO
from typing import Union
from urllib.request import urlretrieve

import prov.model as prov
from xarray import Dataset

from ..abstractio import BaseIO
from ..session import global_session
from ..session import hash_vals
from ..session import Session


# TODO: Evaluate this location
DEFAULT_PATH = Path("")
CACHE_DIR = ".indica"


class ADASReader(BaseIO):
    """Class for reading atomic data from ADAS files.

    Parameters
    ----------
    path: Union[str, Path]
        Location from which relative paths should be evaluated.
        Default is to download files from OpenADAS, storing them
        in your home directory for later use.
    sesssion: Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    def __init__(
        self, path: Union[str, Path] = DEFAULT_PATH, sess: Session = global_session
    ):
        path = Path(path)
        self.session = sess
        self.openadas = path == DEFAULT_PATH
        if path == DEFAULT_PATH:
            self.namespace = "openadas"
            self.session.prov.add_namespace(
                self.namespace, "https://open.adas.ac.uk/detail/adf11/"
            )
            self.path = Path.home() / CACHE_DIR / "adas"
        else:
            self.path = path
            self.namespace = "localadas"
            self.session.prov.add_namespace(
                self.namespace, "file:/" + str(self.path.resolve())
            )
        self.prov_id = hash_vals(path=self.path)
        self.agent = self.session.prov.agent(self.prov_id)
        self.session.prov.delegation(self.session.agent, self.agent)
        self.entity = self.session.prov.entity(
            self.prov_id, {"path": str(self.path.resolve())}
        )
        self.session.prov.generation(
            self.entity, self.session.session, time=datetime.datetime.now()
        )
        self.session.prov.attribution(self.entity, self.session.agent)

    def close(self):
        """Closes connections to database/files. For this class it does not
        need to do anything."""
        pass

    def get_adf11(self, quantity: str, element: str, year: int) -> Dataset:
        """Read data from the specified ADAS file.

        Parameters
        ----------
        quantity
            The type of data to retrieve. Options: scd, acd, plt, prb,
            plsx, prsx.
        element
            The atomic symbol for the element which will be retrieved.
        year
            The year the data is from. Only two digits are needed.

        Returns
        -------
        :
            The data in the specified file. Dimensions are density and
            temperature. Each members of the dataset correspond to a
            different charge state.

        """

    def create_provenance(
        self, filename: Path, start_time: datetime.datetime
    ) -> prov.ProvEntity:
        """Create a provenance entity for the given ADAS file.

        Note that this method just creates the provenance data
        appropriate for the arguments it has been provided with. It
        does not check that these arguments are actually valid and
        that the provenance corresponds to actually existing data.

        """
        end_time = datetime.datetime.now()
        entity = self.session.prov.entity(
            hash_vals(filename=filename, start_time=start_time)
        )
        activity = self.session.prov.activity(
            hash_vals(agent=self.prov_id, date=start_time),
            start_time,
            end_time,
            {prov.PROV_TYPE: "ReadData"},
        )
        self.session.prov.association(activity, self.agent)
        self.session.prov.association(activity, self.session.agent)
        self.session.prov.communication(activity, self.session.session)
        self.session.prov.derivation(entity, f"{self.namespace}:{filename}", activity)
        self.session.prov.generation(entity, activity, end_time)
        self.session.prov.attribution(entity, self.agent)
        self.session.prov.attribution(entity, self.session.agent)
        return entity

    def _get_file(
        self, dataclass: Union[str, Path], filename: Union[str, Path]
    ) -> TextIO:
        """Retrieves an ADAS file, downloading it from OpenADAS if
        necessary. It will cache any downloads for later use.

        Parameters
        ----------
        dataclass
            The format of ADAS data in this file (e.g., ADF11).
        filename
            Name of the file to get.

        Returns
        -------
        :
            A file-like object from which the data can be read.

        """
        filepath = self.path / dataclass / filename
        if self.openadas and not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(
                f"https://open.adas.ac.uk/download/{dataclass}/{filename}", filepath
            )
        return filepath.open("r")

    @property
    def requires_authentication(self) -> Literal[False]:
        """Reading ADAS data never requires authentication."""
        return False
