"""Callback functions to choose an offset for equilibrium data."""

from typing import Callable

from xarray import DataArray

from .numpy_typing import ArrayLike

ConvertToFlux = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
OffsetPicker = Callable[[DataArray, ConvertToFlux], float]


def interactive_offset_choice(T_e: DataArray, converter: ConvertToFlux) -> float:
    """Automatically calculates optimum offset, plots results, and then gives
    user a chance to specify their own choice.

    The user can choose to accept the default value. If they reject it
    they will be prompted to provide their own offset value. The plots
    will then be regenerated using the new value and the user once
    again prompted whether they want to accept it. This process is
    repeated until a value is accepted.

    Because user input is prompted on the command-line, this function is not
    suitable for use within a GUI.

    Parameters
    ----------
    T_e
        Electron temperature data (from HRTS on JET).
    converter
        Function that can get the flux surfece value \rho from (R,z,t)
        coordinates.

    Returns
    -------
    :
        The offset along the major radius to apply to equilibrium data.
    """
    pass
