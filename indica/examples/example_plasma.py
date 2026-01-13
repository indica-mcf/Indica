import numpy as np

from indica import Plasma
from indica import PlasmaProfiler
from indica.numpy_typing import Tuple
from indica.profilers.profiler_gauss import initialise_gauss_profilers


def example_plasma(
    tstart=0.02,
    tend=0.1,
    dt=0.01,
    machine="st40",
    main_ion="h",
    impurities: Tuple[str, ...] = ("c", "ar", "he"),
    **kwargs,
):
    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        machine=machine,
        main_ion=main_ion,
        impurities=impurities,
        full_run=True,
    )
    plasma.build_atomic_data()

    profile_names = [
        "electron_density",
        "ion_temperature",
        "toroidal_rotation",
        "electron_temperature",
    ]
    for imp in impurities:
        profile_names.append(f"impurity_density:{imp}")
    profilers = initialise_gauss_profilers(plasma.rhop, profile_names=profile_names)
    plasma_profiler = PlasmaProfiler(
        plasma=plasma,
        profilers=profilers,
    )
    plasma_profiler()

    # Make profiles evolve in time
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

    return plasma
