"""Experimental design for performing mathematical operations on data.
"""

from abc import ABC
from abc import abstractmethod
import datetime
from itertools import zip_longest
from typing import Any
from typing import cast
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from xarray import DataArray
from xarray import Dataset

from indica.utilities import hash_vals
from ..datatypes import DatasetType
from ..datatypes import DataType

Data = Union[DataArray, Dataset]

if TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)


class OperatorError(Exception):
    """An Exception class raised by :py:class:`operator.Operator` when
    receiving erroneous arguments.

    """


class Operator(ABC):

    """Abstract base class for performing calculations with data.

    Note that the "Parameters" section below describes the paramters
    used when calling an object of this class and *not* when
    constructing a new object as would normally be the case.

    Parameters
    ----------
    kwargs: Any
        Any other arguments which should be recorded.

    Attributes
    ----------
    ARGUMENT_TYPES: ClassVar[List[DataType]]
        Ordered list of the types of data expected for each argument of the
        operator. If there are variadic positional arguments then their type is
        given by the final element of the list.
    RETURN_TYPES: ClassVar[List[DataType]]
        Ordered list of the types of data returned by the operator.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    def __init__(self, **kwargs: Any):
        """Creates a provenance - currently not very elaborate."""
        self.prov_id = hash_vals(operator_type=self.__class__.__name__, **kwargs)
        self._end_time: datetime.datetime

    def _ellipsis_type(self, arg: Data) -> DataType:
        """Given the argument corresponding to the penultimate argument type,
        return the type required for all further variadic arguments.
        """
        if isinstance(arg, DataArray):
            return arg.attrs["datatype"]
        else:
            dtype = arg.attrs["datatype"]
            return dtype[0], {
                k: dtype[1][k] for k in cast(DatasetType, self.ARGUMENT_TYPES[-2])[1]
            }

    def validate_arguments(self, *args: Data):
        """Checks that arguments to the operator are of the expected types.

        Parameters
        ----------
        args
            All of the arguments to be used in the operation.

        """
        self._start_time = datetime.datetime.now()
        arg_len = len(args)
        expected_len = len(self.ARGUMENT_TYPES)
        if expected_len > 0 and self.ARGUMENT_TYPES[-1] == Ellipsis:
            iterator = zip_longest(
                args,
                self.ARGUMENT_TYPES[:-1],
                fillvalue=self._ellipsis_type(args[expected_len - 2]),
            )
        elif arg_len != expected_len:
            message = (
                "Operator of class {} received {} arguments but "
                "expected {}".format(self.__class__.__name__, arg_len, expected_len)
            )
            raise OperatorError(message)
        else:
            # MyPy complaining since iterator is set to type zip_longest earlier in the
            # code, and is set to type zip here even though the two assignments are in
            # two mutually exclusive branches!(if-else branches not git branches)
            # Ignoring for now.
            iterator = zip(args, self.ARGUMENT_TYPES)  # type: ignore
        for i, (arg, expected) in enumerate(iterator):
            if isinstance(arg, DataArray):
                datatype = arg.attrs["datatype"]
                if expected[0] and datatype[0] != expected[0]:
                    message = (
                        "Argument {} of wrong general data type for operator {}: "
                        "expected {}, received {}.".format(
                            i + 1,
                            self.__class__.__name__,
                            expected[0],
                            datatype[0],
                        )
                    )
                    raise OperatorError(message)
                if expected[1] and datatype[1] != expected[1]:
                    message = (
                        "Argument {} of wrong specific data type for operator {}: "
                        "expected to be for {}, received {}.".format(
                            i + 1,
                            self.__class__.__name__,
                            expected[1],
                            datatype[1],
                        )
                    )
                    raise OperatorError(message)
            elif isinstance(arg, Dataset):
                datatype = arg.attrs["datatype"]
                if expected[0] and datatype[0] != expected[0]:
                    message = (
                        "Argument {} of wrong specific data type for operator {}: "
                        "expected {}, received {}.".format(
                            i + 1,
                            self.__class__.__name__,
                            expected[0],
                            datatype[0],
                        )
                    )
                    raise OperatorError(message)

    @abstractmethod
    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        """Indicates the datatypes of the results when calling the operator
        with arguments of the given types. It is assumed that the
        argument types are valid.

        Parameters
        ----------
        args
            The datatypes of the parameters which the operator is to be called with.

        Returns
        -------
        :
            The datatype of each result that will be returned if the operator is
            called with these arguments.

        """
        raise NotImplementedError(
            "{} does not implement a "
            "'return_types' method.".format(self.__class__.__name__)
        )

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
