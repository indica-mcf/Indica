"""Class for reading ADAS data in the ADF11 format."""

from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np

from .adas import ADASReader
from ..datatypes import ArrayType


class ADF11Reader(ADASReader):
    """Class for reading in ADF11 formatted ADAS data.

    Parameters
    ----------
    path: str
        Location from which relative paths should be evaluated.
        Default is installation directory containing distributed
        data files.
    sesssion: Session
        An object representing the session being run. Contains information
        such as provenance data.

    """

    def _get(
        self, absolute_path: str
    ) -> Tuple[Dict[Union[int, str], np.ndarray], ArrayType]:
        """Parse the ADAS file, returning its data in a dictionary.

        Parameters
        ----------
        absolute_path
            The path to the file to be parsed.

        Returns
        -------
        values : Dict[Union[int, str], np.ndarray]
            A dictionary containing the following items:

            densities
                The densities at which the data is provided
            temperatures
                The temperatures at which the data is provided
            _int_
                The ADAS data for this charge state

        data : DataType
            The type of data being read in (quantity and element)
        """
        raise NotImplementedError(
            "{} does not implement a '_get' " "method.".format(self.__class__.__name__)
        )
