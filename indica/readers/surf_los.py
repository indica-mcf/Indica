"""Routines for getting start and end points of lines of sight using data from
Surf data.

"""

from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np


class SURFException(Exception):
    """Exception raised when trying to get line of sight data for an
    instrument that does not exist or a pulse number for which data is
    not available.

    """


def read_surf_los(
    filename: Union[str, Path], pulse: int, instrument: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read beginning and ends of lines of sight from Surf data.

    Optionally scale the lines of sight so they are all the same
    length as the longest one.

    Parameters
    ----------
    filename
        Name of the file containing the FLUSH data.
    pulse
        The pulse number the data is required for.
    instrument
        Which instrument to get the lines of sight for. Format within SURF file
        is inconsistent, but this routine will endeavour to support at least
        the following: SXR/H, SXR/T, SXR/V, KK3, BOLO/KB5H, BOLO/KB5V. For any
        other instrument, try a string which is present within its heading in
        the SURF file.

    Retruns
    -------
    Rstart
        Major radius for the start of the line of sight for each channel.
    Rend
        Major radius for the end of the line of sight for each channel.
    Zstart
        Vertical position for the start of the line of sight for each channel.
    Zend
        Vertical position for the end of the line of sight for each channel.
    """
    pass
