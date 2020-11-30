"""Callback functions to choose an offset for equilibrium data."""

from typing import Callable
from typing import cast
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
from xarray import DataArray
from xarray import Dataset

OffsetPicker = Callable[
    [float, DataArray, DataArray, Optional[DataArray]], Tuple[float, bool]
]


def interactive_offset_choice(
    guess: float, T_e: DataArray, flux: DataArray, offset_at_time: Optional[DataArray]
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
    flux
        Normalised flux surface for each channel at each times.
    offset_at_time
        The optimal R-offset for each time. If present can be used for an
        additional plot.

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
    print(
        f"Displaying electron temperatures as a function of normalised magnetic "
        f"flux\nfor the specified R-offset of {guess}m. Once finished inspecting "
        "the plot(s),\nclose them."
    )

    etemps = Dataset(
        {
            "T_e": (T_e / 1e3).assign_attrs(
                long_name=r"Electron Temperature $T_{\rm e}$", units="keV"
            ),
            "rho_poloidal": flux.assign_attrs(
                long_name=r"Poloidal Magnetic Flux $\rho_{\rm pol}$"
            ),
        }
    ).isel(t=slice(None, None, 10))
    etemps.plot.scatter(x="rho_poloidal", y="T_e")
    plt.tight_layout()
    if offset_at_time is not None:
        plt.figure()
        offset_at_time = cast(
            DataArray,
            offset_at_time.assign_attrs(long_name="Optimal R-offset", units="m"),
        )
        offset_at_time.plot()
        plt.xlabel("Time [s]")
        plt.tight_layout()
    plt.show()
    accept = ask_user(f"Use R-offset of {guess}m?")
    if not accept:
        while True:
            try:
                new_guess = float(input("Specify a new R-offset: "))
                break
            except ValueError:
                print("Please enter a valid floating-point number.")
                continue
    else:
        new_guess = guess
    return new_guess, accept


def ask_user(question: str) -> bool:
    check = str(input(f"{question} (Y/N): ")).lower().strip()
    try:
        if check[0] == "y":
            return True
        elif check[0] == "n":
            return False
        else:
            print("Invalid Input")
            return ask_user(question)
    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user(question)
