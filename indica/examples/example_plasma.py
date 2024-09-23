from pathlib import Path
import pickle
from typing import Tuple

import numpy as np

from indica.models import Plasma
from indica.models.plasma import PlasmaProfiler
from indica.profilers.profiler_gauss import initialise_gauss_profilers


def example_plasma(
    machine: str = "st40",
    pulse: int = None,
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    main_ion="h",
    impurities: Tuple[str, ...] = ("c", "ar", "he"),
    load_from_pkl: bool = True,
    **kwargs,
):
    default_plasma_file = (
        f"{Path(__file__).parent.parent}/data/{machine}_default_plasma_phantom.pkl"
    )

    if load_from_pkl and pulse is not None:
        try:
            print(f"\n Loading phantom plasma class from {default_plasma_file}. \n")
            return pickle.load(open(default_plasma_file, "rb"))
        except FileNotFoundError:
            print(
                f"\n\n No phantom plasma class file {default_plasma_file}. \n"
                f" Building it and saving to file. \n\n"
            )

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        **kwargs,
    )
    plasma.build_atomic_data()

    profilers = initialise_gauss_profilers(
        plasma.rho,
        profile_names=[
            "electron_density",
            "ion_temperature",
            "toroidal_rotation",
            "electron_temperature",
            "impurity_density:ar",
            "impurity_density:c",
            "impurity_density:he",
        ],
    )
    plasma_profiler = PlasmaProfiler(
        plasma=plasma,
        profilers=profilers,
    )

    # Assign profiles to time-points
    nt = len(plasma.t)
    ne_peaking = np.linspace(1, 2, nt)
    te_peaking = np.linspace(1, 2, nt)
    _y0 = plasma_profiler.profilers["toroidal_rotation"].y0
    vrot0 = np.linspace(
        _y0 * 1.1,
        _y0 * 2.5,
        nt,
    )
    vrot_peaking = np.linspace(1, 2, nt)

    _y0 = plasma_profiler.profilers["ion_temperature"].y0
    ti0 = np.linspace(_y0 * 1.1, _y0 * 2.5, nt)

    _y0 = plasma_profiler.profilers[f"impurity_density:{impurities[0]}"].y0
    nimp_y0 = _y0 * 5 * np.linspace(1, 8, nt)
    nimp_peaking = np.linspace(1, 5, nt)
    nimp_wcenter = np.linspace(0.4, 0.1, nt)
    for i, t in enumerate(plasma.t):
        parameters = {
            "electron_temperature.peaking": te_peaking[i],
            "ion_temperature.peaking": te_peaking[i],
            "ion_temperature.y0": ti0[i],
            "toroidal_rotation.peaking": vrot_peaking[i],
            "toroidal_rotation.y0": vrot0[i],
            "electron_density.peaking": ne_peaking[i],
            "impurity_density:ar.peaking": nimp_peaking[i],
            "impurity_density:ar.y0": nimp_y0[i],
            "impurity_density:ar.wcenter": nimp_wcenter[i],
        }

        plasma_profiler(parameters=parameters, t=t)

    if load_from_pkl and pulse is not None:
        print(f"\n Saving phantom plasma class in {default_plasma_file} \n")
        pickle.dump(plasma, open(default_plasma_file, "wb"))

    return plasma


example_plasma()
