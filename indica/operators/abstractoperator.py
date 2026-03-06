"""Experimental design for performing mathematical operations on data."""
from abc import ABC
from abc import abstractmethod
import datetime
from typing import Any
from typing import Union

from xarray import DataArray
from xarray import Dataset

from indica.converters import CoordinateTransform
from indica.plasma import Plasma

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

    def set_transform(self, transform: CoordinateTransform):
        """
        Line-of-sight or Transect coordinate transform
        """
        # TODO: types attribute set during initialisation!
        self.transform = transform

    def set_plasma(self, plasma: Plasma):
        """
        Assign Plasma class to use for computation of forward model
        """
        self.plasma = plasma

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

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
