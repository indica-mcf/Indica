from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica import Equilibrium
from indica import Plasma
from indica import PlasmaProfiler
from indica.configs import ST40Conf
from indica.configs.workflows.composition.default import COMPOSITION_SETTINGS
from indica.models import Interferometer
from indica.models import PinholeCamera
from indica.plotters import DataPlotter
from indica.profilers.profiler_gauss import initialise_gauss_profilers
from indica.readers import ReaderProcessor
from indica.readers import ST40Reader
from indica.readers.modelreader import ModelReader

MODELS = {"smmh": Interferometer, "blom_xy1": PinholeCamera, "sxrc_xy1": PinholeCamera}
full_run = True

def test_new_code(pulse: int = 11314, save_fig: bool = False):
    # 13706 #11418 #13565 #12201 #13661 # #13663 #13645 #13643 #
    settings = COMPOSITION_SETTINGS[pulse]

    prad_instr = settings["prad_instr"]
    try:
        rad_instr = settings["prad_instr"].split("t1d_")[1]
    except Exception as e:
        print(f"{e}: using 2D inversion")
        rad_instr = settings["prad_instr"].split("t2d_")[1]

    to_read = {
        "smmh": "ne_int",
        "ts": "ne",
        "ts": "te",
        "ppts": "ne_rhop",
        "zeff_brems": "zeff",
        prad_instr: "emission_rhop",
        rad_instr: "brightness",
    }

    keys = list(MODELS.keys())
    for key in keys:
        if key not in to_read:
            MODELS.pop(key)

    tstart = settings["tstart"]
    tend = settings["tend"]
    dt = settings["dt"]
    t = settings["t"]
    prad_mult = settings["prad_mult"]
    const_zeff = settings["const_zeff"]
    zeff_const = settings["zeff_const"]
    imp_from_zeff = settings["imp_from_zeff"]
    main_ion = settings["main_ion"]
    impurities = settings["impurities"]
    concentration = settings["concentration"]
    zeff_rhop_lim = settings["zeff_rhop_lim"]
    prad_instr = settings["prad_instr"]
    nplot = settings["nplot"]
    concentration_ratio = [conc / concentration[0] for conc in concentration]

    # Init readers and processors
    # Init readers and processors
    conf = ST40Conf()
    processor = ReaderProcessor()
    st40 = ST40Reader(pulse, 0, tend + 2 * dt)

    # Read gsfit and adapt time range to where equilibrium data is present
    gsfit = st40.get("", "gsfit", 0)
    equil = Equilibrium(gsfit)
    if t is None:
        t = np.arange(tstart, tend, dt)
    else:
        t = np.array(t, ndmin=1)

    t = t[np.where((t >= np.min(equil.t.data)) * (t <= np.max(equil.t.data)))[0]]
    tstart = float(t[0])
    tend = float(t[-1])
    # TODO: error unless this is initialised before the Plasma !
    plotter = DataPlotter(pulse, t, tstart, tend, nplot=nplot)

    # Instatiate plasma class
    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        full_run=True,
    )
    plasma.set_equilibrium(equil)
    plasma.build_atomic_data()
    profile_names = ["neutral_density", "residence_time",
                     "electron_density", "electron_temperature"]
    for imp in impurities:
        profile_names.append(f"impurity_density:{imp}")
    profilers = initialise_gauss_profilers(
        plasma.rhop,
        profile_names=profile_names,
    )
    plasma_profiler = PlasmaProfiler(
        plasma=plasma,
        profilers=profilers,
    )
    _params = plasma_profiler.profilers["electron_density"].get_parameters()

    plasma_profiler(
        {
            "neutral_density.y0": 1.0e14,
            "neutral_density.y1": 1.e16,
            "neutral_density.yend": 1.0e16,
            "neutral_density.peaking": 2.,
            "neutral_density.wcenter": 0.2,
            "neutral_density.wped": 8,
        }
    )

    # plasma_profiler(
    #     {
    #         "residence_time.y0": 0.1,
    #         "residence_time.y1": 0.005,
    #         "residence_time.yend": 0.001,
    #         "residence_time.peaking": 2.,
    #         "residence_time.wcenter": 0.25,
    #         "residence_time.wped": 0.4,
    #     }
    # )
    #

    # Read all experimental data and post-process
    raw, transforms = {}, {}
    instruments = np.unique(list(to_read.keys()))
    print("Reading data")
    raw = st40(instruments)
    for instrument, quantity in to_read.items():
        if instrument not in raw:
            continue

        print(f"Assigning transforms {instrument}")
        try:
            raw[instrument][quantity].transform.set_equilibrium(equil)
            transforms[instrument] = raw[instrument][quantity].transform
        except Exception as e:
            print(e)
            print(f"{instrument}.{quantity} has no transform")
    raw[prad_instr]["emission_rhop"] *= prad_mult

    processed = processor(raw, tstart=tstart, tend=tend, dt=dt)
    for t in plasma.t:
        if np.any(raw["ppts"]["ne_rhop"].sel(t=t, method="nearest") > 1.0e16):
            _value = raw["ppts"]["ne_rhop"].sel(t=t, method="nearest").interp(rhop=plasma.rhop)
        else:
            _value = np.nan
        plasma.electron_density.loc[dict(t=t)] = _value

        if np.any(raw["ppts"]["te_rhop"].sel(t=t, method="nearest") > 100.0):
            _value = raw["ppts"]["te_rhop"].sel(t=t, method="nearest").interp(rhop=plasma.rhop)
        else:
            _value = np.nan
        plasma.electron_temperature.loc[dict(t=t)] = _value

    # Flattened Zeff for rhop outside of range zeff_rhop_lim
    if "zeff" in raw["zeff_brems"].keys():
        _zeff = deepcopy(processed["zeff_brems"]["zeff"]).interp(
            rhop=plasma.rhop, method="nearest", kwargs={"fill_value": "extrapolate"}
        )
        zeff = xr.where(
            (_zeff.rhop <= zeff_rhop_lim[1])
            * (_zeff.rhop >= zeff_rhop_lim[0])
            * (_zeff > 1.5),
            _zeff,
            np.nan,
        )
    else:
        zeff = xr.full_like(plasma.zeff.sum("element"), zeff_const)

    for t in zeff.t:
        val = zeff.sel(t=t)
        if const_zeff:
            zeff.loc[dict(t=t)] = val.mean("rhop", skipna=True)
        else:
            fin = np.where(np.isfinite(val))[0]
            if len(fin) > 0:
                zeff.loc[dict(t=t)] = xr.where(
                    (zeff.rhop >= zeff_rhop_lim[1]), val[fin[-1]], zeff.sel(t=t)
                )
                zeff.loc[dict(t=t)] = xr.where(
                    (zeff.rhop <= zeff_rhop_lim[0]), val[fin[0]], zeff.sel(t=t)
                )

    if imp_from_zeff:
        # Function to calculate impurity density from zeff
        def calc_imp_from_zeff(
            effective_charge: DataArray, element: str, const_dens: bool = False
        ):
            _imp_dens = (
                plasma.electron_density
                * (effective_charge - 1)
                / (
                    plasma.meanz.sel(element=element) ** 2
                    - plasma.meanz.sel(element=element)
                )
            )
            if const_dens:
                _const_val = (_imp_dens).mean("rhop")
                _imp_dens = _imp_dens / _imp_dens * _const_val

            return _imp_dens

        # First guess: whole Zeff is caused by first impurity
        # then iterate a couple of times
        _zeff = deepcopy(zeff)

        for _iter in range(3):
            # Lightest impurity
            _imp_dens = calc_imp_from_zeff(_zeff, impurities[0])

            # Assign impurity density to plasma class
            # Other impurites = desired fraction of first
            imp_dens = []
            for i, imp in enumerate(impurities):
                imp_dens.append(_imp_dens * concentration_ratio[i])
                for t in plasma.t:
                    plasma.impurity_density.loc[dict(element=imp, t=t)] = (
                        imp_dens[i].sel(t=t).data
                    )

            # Recalculate Zeff of 1st impurity subtracting other contributions
            _zeff = deepcopy(zeff)
            for imp in impurities[1:]:
                _zeff -= plasma.zeff.sel(element=imp)
        # Recalculate lightest impurity subtracting other Zeff contributions
        imp_dens[0] = calc_imp_from_zeff(_zeff, impurities[0])
        for t in plasma.t:
            for i, elem in enumerate(impurities):
                plasma.impurity_density.loc[dict(element=elem, t=t)] = (
                    imp_dens[i].sel(t=t).data
                )
    else:
        for i, elem in enumerate(impurities):
            plasma.impurity_density.loc[dict(element=elem)] = (
                plasma.electron_density * concentration[i]
            )

    # Back-calculate expected diagnostic measurements
    print("Running diagnostic models")
    model_kwargs = {rad_instr: {"power_loss": deepcopy(plasma.power_loss_tot)}}
    model_reader = ModelReader(MODELS, model_kwargs=model_kwargs)
    model_reader.set_plasma(plasma)
    model_reader.set_geometry_transforms(transforms, equil)
    bckc = model_reader()

    # Plot
    plt.ioff()
    for instrument, quantity in to_read.items():
        print(f"Plotting {instrument}")
        if quantity not in raw[instrument]:
            continue
        plot_method = conf.INSTRUMENT_METHODS[instrument].replace("get_", "plot_")

        if instrument in bckc:
            _save_fig = False
        else:
            _save_fig = save_fig

        if instrument == rad_instr:
            _to_plot = processed[instrument]
        else:
            _to_plot = raw[instrument]
        getattr(plotter, plot_method)(
            instrument, _to_plot, new_fig=True, save_fig=_save_fig, linestyle="solid"
        )
        if instrument in bckc:
            getattr(plotter, plot_method)(
                instrument,
                bckc[instrument],
                linestyle="dashed",
                marker="x",
                new_fig=False,
                save_fig=save_fig,
                label="Back-calculated",
            )
            plt.legend()

    for elem in impurities:
        plotter.plot_plasma_attribute(
            "ion_concentration",
            plasma,
            save_fig=save_fig,
            element=elem,
            sci=False,
            to_plot="profiles",
        )

    plotter.plot_plasma_attribute(
        "neutral_density", plasma, save_fig=save_fig, ylog=True, to_plot="profiles"
    )

    plotter.plot_plasma_attribute(
        "residence_time", plasma, save_fig=save_fig, ylog=True, to_plot="profiles"
    )

    plotter.plot_plasma_attribute(
        "zeff", plasma, save_fig=save_fig, ylog=False, to_plot="profiles"
    )

    plot_method = conf.INSTRUMENT_METHODS[prad_instr].replace("get_", "plot_")
    getattr(plotter, plot_method)(prad_instr, raw[prad_instr], to_plot="profiles")
    plotter.plot_plasma_attribute(
        "total_radiation",
        plasma,
        save_fig=save_fig,
        to_plot=("profiles",),
        new_fig=False,
        linestyle="dashed",
        label="Back-calculated",
    )

    plot_method = conf.INSTRUMENT_METHODS[prad_instr].replace("get_", "plot_")
    getattr(plotter, plot_method)(
        prad_instr,
        raw[prad_instr],
        to_plot=("integral",),
        color="k",
        label="Exp.",
    )

    plotter.plot_plasma_attribute(
        "prad_tot",
        plasma,
        save_fig=save_fig,
        to_plot=("integral",),
        new_fig=False,
        linestyle="dashed",
        color="r",
        label="Back-calculated",
    )
    plt.legend()

    if "zeff" in raw["zeff_brems"].keys():
        plot_method = conf.INSTRUMENT_METHODS["zeff_brems"].replace("get_", "plot_")
        getattr(plotter, plot_method)(
            "zeff_brems",
            raw["zeff_brems"],
            to_plot="profiles",
            sci=False,
        )
        plotter.plot_plasma_attribute(
            "zeff",
            plasma,
            save_fig=save_fig,
            to_plot=("profiles",),
            new_fig=False,
            linestyle="dashed",
            label="Back-calculated",
            sci=False,
        )

    for elem in impurities:
        plotter.plot_plasma_attribute("fz", plasma, element=elem, save_fig=save_fig)

    if not save_fig:
        plt.show()
    else:
        plt.close("all")

    return plasma, plotter
