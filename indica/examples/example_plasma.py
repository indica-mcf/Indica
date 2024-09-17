from pathlib import Path
import pickle
from typing import Tuple

from indica.models import Plasma
from indica.profilers.profiler_gauss import initialise_gauss_profilers
from indica.workflows.bda.plasma_profiler import PlasmaProfiler


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
    )
    plasma_profiler = PlasmaProfiler(plasma, profilers)
    plasma_profiler()

    if load_from_pkl and pulse is not None:
        print(f"\n Saving phantom plasma class in {default_plasma_file} \n")
        pickle.dump(plasma, open(default_plasma_file, "wb"))

    return plasma
