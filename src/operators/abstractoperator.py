"""Experimental design for performing mathematical operations on data.
"""

from abc import ABC, abstractmethod
import datetime
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import prov.model as prov
from xarrays import DataArray

import session


DataType = Tuple[str, str]


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
    args
        The arguments for the calculation, with number and meaning to be 
        determined by the subclass.

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
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

    ARGUMENT_TYPES: ClassVar[List[DataType]] = []
    RESULT_TYPES: ClassVar[List[DataType]] = []

    def __init__(self, sess: session.Session = session.global_session,
                 **kwargs: Dict[str, Any]):
        """Creates a provenance entity/agent for the operator object. Should
        be called by initialisers in subclasses.

        """
        self._start_time = None
        self._session = sess
        # TODO: also include library version and, ideally, version of
        # relevent dependency in the hash
        self.prov_id = session.hash_vals(operator_type=self.__class__.__name__,
                                         **kwargs)
        self.agent = self._session.prov.agent(self.prov_id)
        self._session.prov.actedOnBehalfOf(self.agent, self._session.agent)
        self.entity = self._session.prov.entity(self.prov_id, kwargs)
        self._session.prov.generation(self.entity, self._session.session,
                                     time=datetime.datetime.now())
        self._session.prov.attribution(self.entity, self._session.agent)
        self._input_provenance: List[prov.ProvEntity] = []

    def validate_arguments(self, *args: Sequence[DataArray]):
        """Checks that arguments to the operator are of the expected types.

        Also gathers provenance information for use later.

        Parameters
        ----------
        args
            All of the arguments to be used in the operation.

        """
        self._start_time = datetime.datetime.now()
        self._input_provenance = [arg.attrs['provenance'] for arg in args]
        arg_len = len(args)
        expected_len = len(self.ARGUMENT_TYPES)
        if arg_len != expected_len:
            message = "Operator of class {} received {} arguments but " \
                "expected {}".format(self.__class__.__name__, arg_len,
                                     expected_len)
            raise OperatorError(message)
        for i, arg, expected in enumerate(zip(args, self.ARGUMENT_TYPES)):
            datatype = arg.attrs['datatype']
            if datatype[0] != expected[0]:
                message = "Argument {} of wrong data type for operator {}: " \
                    "expected {}, received {}.".format(i+1,
                                                       self.__class__.__name__,
                                                       expected[0],
                                                       datatype[0])
                raise OperatorError(message)
            if expected[1] and datatype[1] != expected[1]:
                message = "Argument {} of wrong type of {} for operator {}: " \
                    "expected to be for {}, received {}.".format(
                        i+1, expected[0], self.class_.__name__, expected[1],
                        datatype[1])
                raise OperatorError(message)

    def create_provenance(self) -> prov.ProvEntity:
        """Create a provenance entity for the result of the operator.

        This should only be called after :py:meth:`validate_arguments`, as it
        relies on that routine to collect information about the inputs
        to the operator.

        Note that the results of successive calls to the same operator
        will have different provenance data (with a different
        identifier), as this accounts for the creation-time.

        Returns
        -------
        :
            A provenance entity for the newly calculated data.

        """
        end_time = datetime.datetime.now()
        entity_id = session.hash_vals(creator=self.prov_id, date=end_time,
                                      **{"arg" + str(i): p.identifier for i, p
                                         in enumerate(self._input_provenance)})
        activity_id = session.hash_vals(agent=self.prov_id, date=end_time)
        # TODO: Should each subclass specify its own PROV_TYPE?
        activity = self._session.prov.activity(activity_id, self._start_time,
                                              end_time,
                                              {prov.PROV_TYPE: "Calculation"})
        activity.wasAssociatedWith(self._session.agent)
        activity.wasAssociatedWith(self.agent)
        activity.wasInformedBy(self._session.session)
        # TODO: Should I include any attributes?
        entity = self._session.prov.entity(entity_id)
        entity.wasGeneratedBy(activity, end_time)
        entity.wasAttributedTo(self._session.agent)
        entity.wasAttributedTo(self.agent)
        for arg in self._input_provenance:
            entity.wasDerivedFrom(arg)
            activity.used(arg)
        return entity

    @abstractmethod
    def __call__(self, *args):
        """The invocation of the operator.

        The exact number of arguments should be determined by the
        subclass. However, it is anticipated that these would all be
        :py:class:`xarray.DataArray` objects.

        """
        raise NotImplementedError("{} does not implement a "
                                  "'__call__' method.".format(
                                      self.__class__.__name__))


class CalcZeff(Operator):
    """Calculate effective charge of ions in plasma.

    This is intended for illustrative purposes only and will likely
    undergo considerable refactoring prior to inclusion in the
    codebase

    Note that the "Parameters" section below describes the paramters
    used when calling an object of this class and *not* when
    constructing a new object as would normally be the case.

    Parameters
    ----------
    n_e
        Number density of electrons.
    n_Be
        Number density of Beryllium ions.
    T_e
        Temperature of electrons.

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.
    """

    ARGUMENT_TYPES = ['number_desnity', 'electrons',
                      'number_density', 'beryllium',
                      'temperature', 'electrons']
    RESULT_TYPES = [('effective_charge', 'plasma')]

    def __init__(self, adas_data: str,
                 sess: session.Session = session.global_session):
        """Creates a provenance entity/agent for the operator object.

        Parameters
        ----------
        adas_data
            String indicating what source of atomic data to use. (Details TBC)

        sess
            Object representing this session of calculations with the library.
            Holds and communicates provenance information.

        """
        super(self).__init__(sess, adas_data=adas_data)
        self.adas_data = adas_data

    def __call__(self, n_e: DataArray, n_Be: DataArray,
                 T_e: DataArray) -> DataArray:
        """Perform the calculation."""
        self.validate_arguments(n_e, n_Be, T_e)
        q_Be = None  # TODO: get this from ADAS data and T_e
        # TODO: make sure all arguments are mapped to same coordinate system
        result = (n_e + n_Be*(q_Be**2 - q_Be))/n_e
        result.name = "Zeff"
        result.attrs["generate_mappers"] = n_e.attrs["generate_mappers"]
        result.attrs["map_to_master"] = n_e.attrs["map_to_master"]
        result.attrs["map_from_master"] = n_e.attrs["map_from_master"]
        result.attrs["datatype"] = self.RESULT_TYPES[0]
        result.attrs["provenance"] = self.create_provenance()
        return result
