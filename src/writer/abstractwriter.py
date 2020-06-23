"""Provides an abstract class defining the interface for writing out data."""

from abc import ABC
from abc import abstractmethod
from typing import Literal
from typing import Union

from xarray import DataArray
from xarray import Dataset


class DataWriter(ABC):
    """An abstract class defining the interface for writing data to the
    disk or datatbases.

    """

    def __enter__(self) -> "DataWriter":
        """Called at beginning of a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        """Close reader at end of context manager. Don't try to handle
        exceptions."""
        self.close()
        return False

    def authenticate(self, name: str, password: str) -> bool:
        """Confirms user has permission to write data.

        This must be called before writing data to some locations. The
        default implementation does nothing. If the value of
        `py:meth:requires_authentication` is ``False`` then it does
        not need to be called.

        Parameters
        ----------
        name
            Username to authenticate against.
        password
            Password for that user.

        Returns
        -------
        :
            Indicates whether authentication was succesful.

        """
        return True

    @abstractmethod
    def write(self, uid: str, name: str, *data: Union[Dataset, DataArray]):
        """Write data out to the desired format/database.

        The exact location will be implementation-dependent but will
        include the ``uid`` and ``name`` arguments.

        Parameters
        ----------
        uid
            User ID (i.e., user that created or wrote this data)
        name
            Name to store this data under, such as a DDA
        data
            The data to be written out. The data will be written as though it
            had been merged into a single :py:class:`xarray.Dataset`

        """
        raise NotImplementedError(
            "{} does not implement a `write` method".format(self.__class__.__name__)
        )

    @property
    @abstractmethod
    def requires_authentication(self) -> bool:
        """Indicates whether authentication is required to write data.

        Returns
        -------
        :
            True if authenticationis needed, otherwise false.
        """
        raise NotImplementedError(
            "{} does not implement a "
            "'requires_authentication' "
            "property.".format(self.__class__.__name__)
        )

    @abstractmethod
    def close(self) -> None:
        """Closes connection to whatever backend (file, database, server,
        etc.) to which data is being written."""
        raise NotImplementedError(
            "{} does not implement a 'close' method.".format(self.__class__.__name__)
        )
