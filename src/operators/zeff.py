"""Simple example of an operator calculating $Z_{eff}$.
"""

from typing import ClassVar
from typing import List

from .abstractoperator import Operator
from ..datatypes import DataType
from ..session import global_session
from ..session import Session


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

    ARGUMENT_TYPES: ClassVar[List[DataType]] = [
        ("number_desnity", "electrons"),
        ("number_density", "beryllium"),
        ("temperature", "electrons"),
    ]
    RESULT_TYPES = [("effective_charge", "plasma")]

    def __init__(self, adas_data: str, sess: Session = global_session):
        """Creates a provenance entity/agent for the operator object.

        Parameters
        ----------
        adas_data
            String indicating what source of atomic data to use. (Details TBC)

        sess
            Object representing this session of calculations with the library.
            Holds and communicates provenance information.

        """
        super().__init__(sess, adas_data=adas_data)
        self.adas_data = adas_data

    def __call__(self, n_e, n_Be, T_e):
        """Perform the calculation."""
        self.validate_arguments(n_e, n_Be, T_e)
        q_Be = 1  # TODO: get this from ADAS data and T_e
        # TODO: make sure all arguments are mapped to same coordinate system
        result = (n_e + n_Be * (q_Be ** 2 - q_Be)) / n_e
        # TODO: Properly propagate uncertainty
        result.name = "Zeff"
        result.attrs["generate_mappers"] = n_e.attrs["generate_mappers"]
        result.attrs["map_to_master"] = n_e.attrs["map_to_master"]
        result.attrs["map_from_master"] = n_e.attrs["map_from_master"]
        result.attrs["datatype"] = self.RESULT_TYPES[0]
        result.attrs["provenance"] = self.create_provenance()
        return result
