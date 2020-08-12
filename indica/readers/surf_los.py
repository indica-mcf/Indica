"""Routines for getting start and end points of lines of sight using data from
Surf data.

"""

from typing import Tuple

import numpy as np


def read_surf_los(
    filename: str, pulse: int, instrument: str
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
        Which instrument to get the lines of sight for.

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
