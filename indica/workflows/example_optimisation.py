import numpy as np
from indica.models.plasma import Plasma
from scipy.optimize import least_squares


def match_interferometer_los_int(
    models: dict,
    plasma: Plasma,
    data: dict,
    t: float,
    optimise_for: dict = {"smmh1": ["ne"]},
    guess: float = 5.0e19,
    bounds: tuple = (1.0e17, 1.0e21),
    bckc: dict = {},
):
    def residuals(guess):
        plasma.Ne_prof.y0 = guess
        plasma.assign_profiles("electron_density", t=t)

        for instrument in optimise_for.keys():
            if instrument in models.keys():
                bckc[instrument] = models[instrument](t=t)

        resid = []
        for instrument in optimise_for.keys():
            if instrument in models.keys():
                for quantity in optimise_for[instrument]:
                    resid.append(
                        data[instrument][quantity].sel(t=t) - bckc[instrument][quantity]
                    )

        return (np.array(resid) ** 2).sum()

    least_squares(residuals, guess, bounds=bounds, method="dogbox")

    return plasma, bckc


def match_helike_spectroscopy_line_ratios(
    models: dict,
    plasma: Plasma,
    data: dict,
    t: float,
    optimise_for: dict = {"xrcs": ["int_k/int_w"]},
    guess: float = 1.0e3,
    bounds: tuple = (1.0e2, 10e4),
    bckc: dict = {},
):
    def residuals(guess):
        plasma.Te_prof.y0 = guess
        plasma.assign_profiles("electron_temperature", t=t)
        plasma.Ti_prof.y0 = guess
        plasma.assign_profiles("ion_temperature", t=t)

        for instrument in optimise_for.keys():
            if instrument in models.keys():
                bckc[instrument] = models[instrument](t=t)

        resid = []
        for instrument in optimise_for.keys():
            if instrument in models.keys():
                for quantity in optimise_for[instrument]:
                    resid.append(
                        data[instrument][quantity].sel(t=t) - bckc[instrument][quantity]
                    )

        return (np.array(resid) ** 2).sum()

    least_squares(residuals, guess, bounds=bounds, method="dogbox")

    return plasma, bckc


def match_helike_spectroscopy_ion_temperature(
    models: dict,
    plasma: Plasma,
    data: dict,
    t: float,
    optimise_for: dict = {"xrcs": ["ti_w"]},
    guess: float = 1.0e3,
    bounds: tuple = (1.0e2, 1.5e3),
    bckc: dict = {},
):
    def residuals(guess):
        plasma.Ti_prof.y0 = guess
        plasma.assign_profiles("ion_temperature", t=t)

        for instrument in optimise_for.keys():
            if instrument in models.keys():
                bckc[instrument] = models[instrument](t=t)

        resid = []
        for instrument in optimise_for.keys():
            if instrument in models.keys():
                for quantity in optimise_for[instrument]:
                    resid.append(
                        data[instrument][quantity].sel(t=t) - bckc[instrument][quantity]
                    )

        return (np.array(resid) ** 2).sum()

    least_squares(residuals, guess, bounds=bounds, method="dogbox")

    return plasma, bckc


def match_helike_spectroscopy_intensity(
    models: dict,
    plasma: Plasma,
    data: dict,
    t: float,
    optimise_for: dict = {"xrcs": ["int_w"]},
    guess: float = 1.0e15,
    bounds: tuple = (1.0e13, 1.0e19),
    bckc: dict = {},
    element: str = "ar",
):
    def residuals(guess):
        plasma.Nimp_prof.y0 = guess
        plasma.assign_profiles("impurity_density", t=t, element=element)

        for instrument in optimise_for.keys():
            if instrument in models.keys():
                bckc[instrument] = models[instrument](t=t)

        resid = []
        for instrument in optimise_for.keys():
            if instrument in models.keys():
                for quantity in optimise_for[instrument]:
                    resid.append(
                        data[instrument][quantity].sel(t=t) - bckc[instrument][quantity]
                    )

        return (np.array(resid) ** 2).sum()

    least_squares(residuals, guess, bounds=bounds, method="dogbox")

    return plasma, bckc
