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
from warnings import warn

import prov.model as prov
from xarray import DataArray
from xarray import Dataset

from .. import session
from ..datatypes import DatasetType
from ..datatypes import DataType
from ..datatypes import DatatypeWarning
from ..datatypes import GENERAL_DATATYPES
from ..datatypes import SPECIFIC_DATATYPES

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
    sess: session.Session
        An object representing the session being run. Contains information
        such as provenance data.
    kwargs: Any
        Any other arguments which should be recorded in the PROV entity for
        the reader.

    Attributes
    ----------
    ARGUMENT_TYPES: ClassVar[List[DataType]]
        Ordered list of the types of data expected for each argument of the
        operator. If there are variadic positional arguments then their type is
        given by the final element of the list.
    RETURN_TYPES: ClassVar[List[DataType]]
        Ordered list of the types of data returned by the operator.
    prov_id: str
        The hash used to identify this object in provenance documents.
    agent: prov.model.ProvAgent
        An agent representing this object in provenance documents.
        DataArray objects can be attributed to it.
    entity: prov.model.ProvEntity
        An entity representing this object in provenance documents. It is used
        to provide information on the object's own provenance.

    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = []

    def __init__(self, sess: session.Session = session.global_session, **kwargs: Any):
        """Creates a provenance entity/agent for the operator object. Also
        checks arguments and results are of valid datatypes. Should be
        called by initialisers in subclasses.

        """
        self._session = sess
        # TODO: also include library version and, ideally, version of
        # relevent dependency in the hash
        self.prov_id = session.hash_vals(
            operator_type=self.__class__.__name__, **kwargs
        )
        self.agent = self._session.prov.agent(self.prov_id)
        self._session.prov.actedOnBehalfOf(self.agent, self._session.agent)
        self.entity = self._session.prov.entity(self.prov_id, kwargs)
        self._session.prov.generation(
            self.entity, self._session.session, time=datetime.datetime.now()
        )
        self._session.prov.attribution(self.entity, self._session.agent)
        self._input_provenance: List[prov.ProvEntity] = []
        self._prov_count = 0
        self._end_time: datetime.datetime
        self.activity: prov.ProvActivity
        for i, datatype in enumerate(self.ARGUMENT_TYPES):
            if isinstance(datatype, EllipsisType):
                if i + 1 != len(self.ARGUMENT_TYPES):
                    raise TypeError(
                        (
                            "Operator class {} uses ellipsis dots as a type for"
                            " argument {}. Only supported in final position."
                        ).format(self.__class__.__name__, i + 1)
                    )
                else:
                    continue
            if datatype[0] not in GENERAL_DATATYPES:
                warn(
                    "Operator class {} expects argument {} to have "
                    "unrecognised general datatype '{}'".format(
                        self.__class__.__name__, i + 1, datatype[0]
                    ),
                    DatatypeWarning,
                )
            if datatype[1] not in SPECIFIC_DATATYPES:
                warn(
                    "Operator class {} expects argument {} to have "
                    "unrecognised specific datatype '{}'".format(
                        self.__class__.__name__, i + 1, datatype[1]
                    ),
                    DatatypeWarning,
                )

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

        Also gathers provenance information for use later.

        Parameters
        ----------
        args
            All of the arguments to be used in the operation.

        """
        self._start_time = datetime.datetime.now()
        self._input_provenance = [
            arg.attrs["provenance"] for arg in args if "provenance" in arg.attrs
        ]
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
            iterator = zip(args, self.ARGUMENT_TYPES)
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
                for key, general_type in expected[1].items():
                    if key not in datatype[1]:
                        message = (
                            "Variable {} required by operator {} is missing from "
                            "argument {}.".format(
                                key,
                                self.__class__.__name__,
                                i + 1,
                            )
                        )
                        raise OperatorError(message)
                    if datatype[1][key] != general_type:
                        message = (
                            "Variable {} of argument {} of wrong general data type for "
                            "operator {}: expected {}, received {}.".format(
                                key,
                                i + 1,
                                self.__class__.__name__,
                                general_type,
                                datatype[1][key],
                            )
                        )
                        raise OperatorError(message)
            else:
                raise OperatorError(
                    "Argument {} is not a DataArray or Dataset".format(arg)
                )

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

    def assign_provenance(self, data: Union[DataArray, Dataset]) -> prov.ProvEntity:
        """Create and assign a provenance entity to the argument. This argument
        should be one of the results of the operator.

        This should only be called after
        :py:meth:`validate_arguments`, as it relies on that routine to
        collect information about the inputs to the operator. It
        should not be called until after all calculations are
        finished, as the first call will be used to determine the
        end-time of the calculation.

        Returns
        -------
        :
            A provenance entity for the newly calculated data.

        """
        # TODO: Generate multiple pieces of PROV data for multiple return values
        if self._prov_count == 0:
            self.end_time = datetime.datetime.now()
            activity_id = session.hash_vals(agent=self.prov_id, date=self.end_time)
            self.activity = self._session.prov.activity(
                activity_id,
                self._start_time,
                self.end_time,
                {prov.PROV_TYPE: "Calculation"},
            )
            self.activity.wasAssociatedWith(self._session.agent)
            self.activity.wasAssociatedWith(self.agent)
            self.activity.wasInformedBy(self._session.session)
            for arg in self._input_provenance:
                self.activity.used(arg)
        entity_id = session.hash_vals(
            creator=self.prov_id,
            date=self.end_time,
            result_number=self._prov_count,
            **{
                "arg" + str(i): p.identifier
                for i, p in enumerate(self._input_provenance)
            }
        )
        self._prov_count += 1
        if isinstance(data, Dataset):
            entity = self._session.prov.collection(
                entity_id, {prov.PROV_TYPE: "Dataset"}
            )
            for array in data.data_vars.values():
                if "provenance" not in array.attrs:
                    print("Creating provenenace")
                    self.assign_provenance(array)
                entity.hadMember(array.attrs["provenance"])
        else:
            entity = self._session.prov.entity(
                entity_id,
                {
                    prov.PROV_TYPE: "DataArray",
                    prov.PROV_VALUE: ",".join(data.attrs["datatype"]),
                },
            )
        entity.wasGeneratedBy(self.activity, self.end_time)
        entity.wasAttributedTo(self._session.agent)
        entity.wasAttributedTo(self.agent)
        for arg in self._input_provenance:
            entity.wasDerivedFrom(arg)
        if isinstance(data, Dataset):
            data.attrs["provenance"] = entity
        else:
            data.attrs["partial_provenance"] = entity
            if data.indica.equilibrium:
                data.indica._update_prov_for_equilibrium(
                    data.attrs["transform"].equilibrium
                )
            else:
                data.attrs["provenance"] = entity

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
