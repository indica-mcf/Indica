"""Base class for IO objects.
"""

from abc import ABC
from abc import abstractmethod
from typing import Literal


class BaseIO(ABC):
    """An abstract class defining methods needed by all IO objects.

    """

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
