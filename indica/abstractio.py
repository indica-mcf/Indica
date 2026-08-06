"""Base class for IO objects."""

from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Literal
from typing import Tuple

import numpy as np

from indica.numpy_typing import RevisionLike


class BaseIO(ABC):
    """An abstract class defining methods needed by all IO objects."""

    def __enter__(self) -> "BaseIO":
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

    @property
    @abstractmethod
    def requires_authentication(self) -> bool:
        """Indicates whether authentication is required to read data.

        Returns
        -------
        :
            True if authentication is needed, otherwise false.
        """
        raise NotImplementedError(
            "{} does not implement a 'requires_authentication' property.".format(
                self.__class__.__name__
            )
        )

    @abstractmethod
    def close(self) -> None:
        """Closes connection to whatever backend (file, database, server,
        etc.) to which data is being written."""
        raise NotImplementedError(
            "{} does not implement a 'close' method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array], str, str]:
        """Fetch a quantity, it's coordinates, and it's units from a database."""
        raise NotImplementedError(
            "{} does not implement a 'get_data' method.".format(self.__class__.__name__)
        )

    @abstractmethod
    def get_error(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array], str, str]:
        """Fetch the errors associated with a quantity from a database."""
        raise NotImplementedError(
            "{} does not implement a 'get_error' method.".format(
                self.__class__.__name__
            )
        )

    @abstractmethod
    def get_revision(
        self, uid: str, instrument: str, revision: RevisionLike
    ) -> tuple[RevisionLike, bool]:
        """Get actual revision that's being read from database, converts relative
        revision (e.g. 0, latest) to absolute, and return if latest/best revision
        according to source.

        """
        raise NotImplementedError(
            "{} does not implement a 'get_revision' method.".format(
                self.__class__.__name__
            )
        )
