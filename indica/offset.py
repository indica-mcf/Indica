"""Callback functions to choose an offset for equilibrium data."""

from typing import Callable
from typing import Tuple

from xarray import DataArray

from .numpy_typing import ArrayLike

ConvertToFlux = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
OffsetPicker = Callable[[float, DataArray, ConvertToFlux], Tuple[float, bool]]


def interactive_offset_choice(
    guess: float, T_e: DataArray, converter: ConvertToFlux
) -> Tuple[float, bool]:
    """Plots electron temperature against flux surface when using best
    guess for location of magnetic axis. User then has a chance to accept this
    value or specify their own alternative. The function can be called again
    with the new value, if necessary.

    Because user input is prompted on the command-line, this function is not
    suitable for use within a GUI.

    Parameters
    ----------
    guess
        Estimate for appropriate offset of the magnetic axis location.
    T_e
        Electron temperature data (from HRTS on JET).
    converter
        Function that can get the flux surfece value \rho from (R,z,t)
        coordinates.

    Returns
    -------
    offset : float
        The offset along the major radius to apply to equilibrium data accepted
        or suggested by the user.
    accept : bool
        Whether the user accepted the offset value or whether, instead, this is
        a new suggestion. If the latter the function can be called again with
        the new guess.

    """
    pass
