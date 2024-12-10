"""Experimental design for performing mathematical operations on data.
"""

from abc import ABC
from abc import abstractmethod
import datetime
from typing import Any
from typing import Union

from xarray import DataArray
from xarray import Dataset

Data = Union[DataArray, Dataset]


class OperatorError(Exception):
    """An Exception class raised by :py:class:`operator.Operator` when
    receiving erroneous arguments.
    """


class Operator(ABC):
    """Abstract base class for performing calculations with data."""

    def __init__(self, **kwargs: Any):
        """Creates a provenance - currently not very elaborate."""
        self._init_time: datetime.datetime

    @abstractmethod
    def __call__(self, *args: DataArray) -> Union[DataArray, Dataset]:
        """The invocation of the operator.

        The exact number of arguments should be determined by the
        subclass. However, it is anticipated that these would all be
        :py:class:`xarray.DataArray` objects.

        Unfortunately, we can not use Mypy static type-checking for
        this routine or its overriding implementations, as the number
        of arguments will vary.

        """
        raise NotImplementedError(
            "{} does not implement a "
            "'__call__' method.".format(self.__class__.__name__)
        )
