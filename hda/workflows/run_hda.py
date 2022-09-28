import xarray as xr
from xarray import Dataset
import hda.hda_tree as hda_tree
from hda.models.xray_crystal_spectrometer import XRCSpectrometer
from hda.models.interferometer import Interferometer
from hda.models.diode_filter import Diode_filter
from hda.manage_data import bin_data_in_time, map_on_equilibrium, initialize_bckc
from hda.models.plasma import Plasma
from hda.read_st40 import ST40data
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates, LinesOfSightTransform
from indica.converters.line_of_sight import LineOfSightTransform
from indica.provenance import get_prov_attribute
from hda.optimizations.interferometer import match_interferometer_los_int
from hda.optimizations.xray_crystal_spectrometer import (
    match_line_ratios,
    match_ion_temperature,
    match_intensity,
)
import hda.plots as plots
import os
import pickle

from hda.profiles import profile_scans

import matplotlib.pylab as plt
import numpy as np
from copy import deepcopy

plt.ion()

"""
Refactored HDA workflows to check against hdatests.test_hda
"""

INTERFEROMETERS = ["smmh1", "nirh1"]


def run_default(pulse: int = 10009, use_ref=True):
    """
    Run HDA workflow with default settings

    Parameters
    ----------
    pulse
        Pulse number
    use_ref
        Model Ti profile based on Te as explained in Profiles class

    Returns
    -------
    plasma class, binned data and back-calculated values from optimisations, raw experimental data

    """
    plasma, raw_data, data, bckc = initialize_workflow(pulse)
    run_optimisations(plasma, data, bckc, use_ref=use_ref)

    # plot_results(plasma, data, bckc, raw_data)
    plots.compare_data_bckc(data, bckc, raw_data=raw_data, pulse=plasma.pulse)
    # plots.profiles(plasma, data=data, bckc=bckc)
    # plots.time_evol(plasma, data, bckc=bckc)

    return plasma, data, bckc, raw_data


def scan_profiles(
    pulse: int = 9780, write=False, run_add="REF", modelling=True, force=True
):

    profs = profile_scans()

    plasma, raw_data, data, bckc = initialize_workflow(pulse)

    pulse = plasma.pulse
    if modelling:
        pulse_to_write = pulse + 25000000
    else:
        pulse_to_write = pulse

    run = 60
    run_tmp = deepcopy(run)
    pl_dict = {}
    bckc_dict = {}
    run_dict = {}
    iteration = 0
    for kNe, Ne in profs["Ne"].items():
        plasma.Ne_prof = deepcopy(Ne)
        for kTe, Te in profs["Te"].items():
            plasma.Te_prof = deepcopy(Te)
            for kTi, Ti in profs["Ti"].items():
                Ti = deepcopy(Te)
                Ti.datatype = ("temperature", "ion")
                plasma.Ti_prof = Ti
                if kTi == "broad":
                    use_ref = False
                else:
                    use_ref = True
                for kNimp, Nimp in profs["Nimp"].items():
                    if kNimp != "peaked":
                        Nimp = deepcopy(Ne)
                        Nimp.datatype = ("density", "impurity")
                        Nimp.y1 = (
                            Nimp.yspl.sel(rho_poloidal=0.7, method="nearest").values
                            / 1.5
                        )
                        Nimp.yend = Nimp.y1
                        Nimp.build_profile()
                    plasma.Nimp_prof = deepcopy(Nimp)

                    run_tmp += 1

                    run_name = f"RUN{run_tmp}{run_add}"
                    descr = f"{kTe} Te, {kTi} Ti, {kNe} Ne, {kNimp} Cimp"
                    print(f"\n{descr}\n")
                    run_dict[run_name] = descr

                    run_optimisations(plasma, data, bckc, use_ref=use_ref)

                    pl_dict[run_name] = deepcopy(plasma)
                    bckc_dict[run_name] = deepcopy(bckc)

                    if write:
                        hda_tree.write(
                            plasma,
                            pulse_to_write,
                            "HDA",
                            data=data,
                            bckc=bckc,
                            descr=descr,
                            run_name=run_name,
                            force=force,
                        )

                        save_to_pickle(
                            plasma,
                            raw_data,
                            data,
                            bckc,
                            pulse=plasma.pulse,
                            name=run_name,
                            force=force,
                        )

                    iteration += 1

    return pl_dict, raw_data, data, bckc_dict, run_dict


def run_optimisations(plasma, data, bckc, use_ref=False):
    opt = plasma.optimisation

    # Initialize some optimisation parameters
    te0 = 1.0e3
    te0_ref = None

    # Initialize other variables TODO: put these somewhere else...
    xrcs = plasma.forward_models["xrcs"]

    for i, t in enumerate(plasma.t):
        print(float(t))
        # Match chosen interferometer
        _, Ne_prof = match_interferometer_los_int(
            plasma.forward_models[opt["el_dens"]["diagnostic"]],
            plasma.Ne_prof,
            data[opt["el_dens"]["diagnostic"]],
            t,
            quantities=opt["el_dens"]["quantities"],
        )
        plasma.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
        # Back-calculate the LOS-integral of all the interferometers for consistency checks
        for diagnostic in INTERFEROMETERS:
            bckc_tmp, _ = plasma.forward_models[diagnostic].integrate_on_los(
                plasma.el_dens.sel(t=t), t=t,
            )
            for quantity in opt["el_dens"]["quantities"]:
                bckc[diagnostic][quantity].loc[dict(t=t)] = bckc_tmp[quantity].values

        # Optimize electron temperature for XRCS line ratios
        plasma.calc_imp_dens(t=t)
        if t > plasma.t.min():
            te0 = plasma.el_temp.sel(t=plasma.t[i - 1], rho_poloidal=0).values / 2.0

        Ne = plasma.el_dens.sel(t=t)
        Nimp = plasma.imp_dens.sel(t=t)
        Nh = plasma.neutral_dens.sel(t=t)
        tau = plasma.tau.sel(t=t)
        if not np.any(tau > 0):
            tau = None

        _, Te_prof = match_line_ratios(
            xrcs,
            plasma.Te_prof,
            data[opt["el_temp"]["diagnostic"]],
            t,
            Ne,
            Nimp=Nimp,
            Nh=Nh,
            tau=tau,
            quantities=opt["el_temp"]["quantities"],
            te0=te0,
        )
        plasma.el_temp.loc[dict(t=t)] = Te_prof.yspl.values

        if use_ref:
            te0_ref = plasma.el_temp.sel(t=t, rho_poloidal=0).values

        _, Ti_prof = match_ion_temperature(
            xrcs,
            plasma.Ti_prof,
            data[opt["ion_temp"]["diagnostic"]],
            t,
            quantities=opt["ion_temp"]["quantities"],
            lines=opt["ion_temp"]["lines"],
            te0_ref=te0_ref,
        )
        for elem in plasma.elements:
            plasma.ion_temp.loc[dict(t=t, element=elem)] = Ti_prof.yspl.values

        _, Nimp_prof = match_intensity(
            xrcs,
            plasma.Nimp_prof,
            data[opt["ar_dens"]["diagnostic"]],
            t,
            quantities=opt["ar_dens"]["quantities"],
            lines=opt["ar_dens"]["lines"],
        )
        plasma.imp_dens.loc[
            dict(t=t, element=opt["ar_dens"]["element"])
        ] = Nimp_prof.yspl.values

        assign_xrcs_bckc(plasma, t, xrcs, bckc)

        Bremss = plasma.forward_models["lines"]
        Bremss.calculate_emission(
            plasma.el_temp.sel(t=t),
            plasma.el_dens.sel(t=t),
            plasma.zeff.sum("element").sel(t=t),
        )
        los_integral, _ = Bremss.integrate_on_los(t=t)
        bckc["lines"]["brems"].loc[dict(t=t)] = los_integral.values

    return plasma, data, bckc


def initialize_workflow(pulse):
    plasma = initialize_plasma_object()
    raw_data, data, bckc = initialize_plasma_data(pulse=pulse, plasma=plasma)
    initialize_forward_models(plasma, data)
    initialize_optimisations(plasma, data)

    return plasma, raw_data, data, bckc


def initialize_plasma_object(
    pulse: int = 9780,
    tstart=0.02,
    tend=0.10,
    dt=0.01,
    main_ion="h",
    impurities: tuple = ("c", "ar", "he"),
    imp_conc: tuple = (0.03, 0.001, 0.01),
    full_run=False,
):

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        main_ion=main_ion,
        impurities=impurities,
        imp_conc=imp_conc,
        pulse=pulse,
        full_run=full_run,
    )
    plasma.build_atomic_data(default=True)
    plasma.forward_models = {}

    return plasma


def initialize_plasma_data(
    pulse: int, plasma: Plasma = None, equilibrium_diagnostic="efit",
):
    """
    Read ST40 data, initialize Equilibrium class and Flux Transforms,
    map all diagnostics to equilibrium

    Parameters
    ----------
    plasma
        Plasma class
    equilibrium_diagnostic
        Diagnostic to be used as Equilibrium

    Returns
    -------
    Raw data and binned data dictionaries
    """
    plasma.pulse = pulse
    raw = ST40data(pulse, plasma.tstart - plasma.dt / 2, plasma.tend + plasma.dt / 2)
    raw_data = raw.get_all()

    equilibrium_data = raw_data[equilibrium_diagnostic]
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)
    plasma.calculate_geometry()

    data = {}
    for kinstr in raw_data.keys():
        data[kinstr] = bin_data_in_time(
            raw_data[kinstr], plasma.tstart, plasma.tend, plasma.dt,
        )
        map_on_equilibrium(data[kinstr], flux_transform=plasma.flux_transform)
    bckc = initialize_bckc(data)

    revision = get_prov_attribute(
        equilibrium_data[list(equilibrium_data)[0]].provenance, "revision"
    )
    plasma.optimisation["equil"] = f"{equilibrium_diagnostic}:{revision}"

    return raw_data, data, bckc


def initialize_forward_models(
    plasma: Plasma, data: dict, marchuk=True, extrapolate="constant",  # None
):

    for diag in INTERFEROMETERS:
        plasma.forward_models[diag] = Interferometer(name=diag)
        plasma.forward_models[diag].set_los_transform(
            data[diag]["ne"].attrs["transform"]
        )

    plasma.forward_models["xrcs"] = XRCSpectrometer(
        marchuk=marchuk, extrapolate=extrapolate, fract_abu=plasma.fract_abu,
    )
    plasma.forward_models["xrcs"].set_los_transform(
        data["xrcs"][list(data["xrcs"])[0]].attrs["transform"]
    )

    plasma.forward_models["lines"] = Diode_filter("brems")
    plasma.forward_models["lines"].set_los_transform(
        data["lines"]["brems"].attrs["transform"]
    )


def initialize_optimisations(
    plasma: Plasma,
    data: dict,
    diagnostic_ne="smmh1",
    diagnostic_te="xrcs",
    diagnostic_ti="xrcs",
    diagnostic_ar="xrcs",
    quantities_ne=["ne"],
    quantities_te=[
        "int_n3/int_tot"
    ],  # ["int_k/int_w", "int_n3/int_w", "int_n3/int_tot"]
    quantities_ti=["ti_w"],
    lines_ti=["w"],
    quantities_ar=["int_w"],
    lines_ar=["w"],
):

    revision = get_prov_attribute(
        data[diagnostic_ne][list(data[diagnostic_ne])[0]].provenance, "revision"
    )
    plasma.optimisation["el_dens"] = {
        "diagnostic": diagnostic_ne,
        "quantities": quantities_ne,
        "rev": revision,
    }

    revision = get_prov_attribute(
        data[diagnostic_te][list(data[diagnostic_te])[0]].provenance, "revision"
    )
    plasma.optimisation["el_temp"] = {
        "diagnostic": diagnostic_te,
        "quantities": quantities_te,
        "rev": revision,
    }

    revision = get_prov_attribute(
        data[diagnostic_ti][list(data[diagnostic_ti])[0]].provenance, "revision"
    )
    plasma.optimisation["ion_temp"] = {
        "diagnostic": diagnostic_ti,
        "quantities": quantities_ti,
        "lines": lines_ti,
        "rev": revision,
    }

    revision = get_prov_attribute(
        data[diagnostic_ar][list(data[diagnostic_ar])[0]].provenance, "revision"
    )
    plasma.optimisation["ar_dens"] = {
        "diagnostic": diagnostic_ar,
        "quantities": quantities_ar,
        "lines": lines_ar,
        "element": "ar",
        "rev": revision,
    }


def assign_xrcs_bckc(plasma: Plasma, t: float, xrcs: XRCSpectrometer, bckc: dict):
    """
    Map xrcs forward model results to desired bckc dictionary structure
    """
    xrcs_keys = list(bckc["xrcs"].keys())

    # Initialize missing variables
    if "emiss" not in bckc["xrcs"][xrcs_keys[0]].attrs.keys():
        emiss = []
        for _ in plasma.time:
            emiss.append(xrcs.emission["w"] * 0.0)
        emiss = xr.concat(emiss, "t").assign_coords(t=plasma.time)

        pos = xr.full_like(plasma.time, np.nan)
        for k in xrcs_keys:
            bckc["xrcs"][k].attrs["emiss"] = deepcopy(emiss)
            bckc["xrcs"][k].attrs["pos"] = {
                "value": deepcopy(pos),
                "err_in": deepcopy(pos),
                "err_out": deepcopy(pos),
            }

    # Map quantities
    los_int, _ = xrcs.integrate_on_los(t=t)
    emiss_keys = xrcs.emission.keys()
    for quantity in xrcs_keys:
        tmp = quantity.split("_")
        measurement = tmp[0]

        if measurement != "te" and measurement != "ti" and measurement != "int":
            continue

        line = tmp[-1]
        if line not in emiss_keys:
            continue

        if "/" in quantity:
            tmp = quantity.split("/")
            line = tmp[0].split("_")[1] + tmp[1].split("_")[1]
        if line not in emiss_keys:
            continue

        bckc["xrcs"][quantity].emiss.loc[dict(t=t)] = xrcs.emission[line]

        if measurement == "te":
            bckc["xrcs"][quantity].loc[dict(t=t)] = xrcs.moment_analysis(
                plasma.el_temp.sel(t=t), t, line=line
            )
        if measurement == "ti":
            bckc["xrcs"][quantity].loc[dict(t=t)] = xrcs.moment_analysis(
                plasma.ion_temp.sel(t=t, element=plasma.main_ion), t, line=line
            )

        if measurement == "int":
            # TODO: make this clearer...still too messy
            if quantity in los_int.keys():
                bckc["xrcs"][quantity].loc[dict(t=t)] = los_int[quantity]
            else:
                bckc["xrcs"][quantity].loc[dict(t=t)] = los_int[line]

        pos, err_in, err_out = xrcs.calculate_emission_position(t, line=line)
        bckc["xrcs"][quantity].pos["value"].loc[dict(t=t)] = pos
        bckc["xrcs"][quantity].pos["err_in"].loc[dict(t=t)] = err_in
        bckc["xrcs"][quantity].pos["err_out"].loc[dict(t=t)] = err_out


def calc_centrifugal_asymmetry(plasma: Plasma, vtor0=500.0e3):
    plasma.vtor.values = (plasma.ion_temp / plasma.ion_temp.max() * vtor0).values
    plasma.calc_centrifugal_asymmetry()


def simulate_new_interferometer(
    plasma: Plasma, data: dict, bckc: dict, name: str = "smmh2"
):
    # Add invented interferometer with different LOS 20 cm above the SMMH1
    # using the new LOS transform from Jon

    opt = plasma.optimisation

    plasma.forward_models[name] = Interferometer(name="smmh2")
    _trans = data["smmh1"]["ne"].attrs["transform"]
    los_transform = LinesOfSightTransform(
        x_start=_trans.x_start.values,
        y_start=_trans.y_start.values,
        z_start=_trans.z_start.values + 0.15,
        x_end=_trans.x_end.values,
        y_end=_trans.y_end.values,
        z_end=_trans.z_end.values + 0.15,
        name=name,
        machine_dimensions=_trans._machine_dims,
    )
    los_transform.set_flux_transform(plasma.flux_transform)
    _ = los_transform.convert_to_rho(t=data["smmh1"]["ne"].t)
    plasma.forward_models[name].set_los_transform(los_transform)
    bckc[name] = {}
    los_integral, _ = plasma.forward_models[name].integrate_on_los(plasma.el_dens)
    for quantity in opt["el_dens"]["quantities"]:
        bckc[name][quantity] = los_integral[quantity]

    # Test line_of_sight vs. lines_of_sight transforms
    start = [
        los_transform.x_start.values,
        los_transform.y_start.values,
        los_transform.z_start.values,
    ]
    finish = [
        los_transform.x_end.values,
        los_transform.y_end.values,
        los_transform.z_end.values,
    ]
    origin = np.array(start).flatten()
    direction = (np.array(finish) - np.array(start)).flatten()
    los_transform_jw = LineOfSightTransform(
        origin_x=origin[0],
        origin_y=origin[1],
        origin_z=origin[2],
        direction_x=direction[0],
        direction_y=direction[1],
        direction_z=direction[2],
        name="smmh2_jw",
        dl=0.006,
        machine_dimensions=los_transform._machine_dims,
    )

    los_transform_jw.set_flux_transform(plasma.flux_transform)
    _ = los_transform_jw.convert_to_rho(t=data["smmh1"]["ne"].t)
    plasma.forward_models["smmh2_jw"] = Interferometer(name="smmh2_jw")
    plasma.forward_models["smmh2_jw"].set_los_transform(los_transform_jw)
    bckc["smmh2_jw"] = {}
    los_integral, _ = plasma.forward_models["smmh2_jw"].integrate_on_los(plasma.el_dens)
    for quantity in opt["el_dens"]["quantities"]:
        bckc["smmh2_jw"][quantity] = los_integral[quantity]


def plot_results(plasma: Plasma, data: dict, bckc: dict, raw_data: dict):

    simulate_new_interferometer(plasma, data, bckc, name="smmh2")
    if not hasattr(plasma, "ion_dens_2d"):
        calc_centrifugal_asymmetry(plasma)

    opt = plasma.optimisation

    # Plot comparison of raw data, binned data and back-calculated values
    plt.figure()
    colors = {"nirh1": "blue", "smmh1": "purple"}
    for diag in INTERFEROMETERS:
        for quantity in opt["el_dens"]["quantities"]:
            raw_data[diag][quantity].plot(color=colors[diag], label=diag)
            data[diag][quantity].plot(color=colors[diag], marker="o")
            bckc[diag][quantity].plot(color=colors[diag], marker="x")

    for quantity in opt["el_dens"]["quantities"]:
        bckc["smmh2"][quantity].plot(color="red", marker="D", label="smmh2", alpha=0.5)
        bckc["smmh2_jw"][quantity].plot(
            color="green", marker="*", label="smmh2", alpha=0.5, linestyle="dashed"
        )
    plt.legend()

    # Plot resulting electron density profiles
    plt.figure()
    plt.plot(plasma.el_dens.sel(t=slice(0.02, 0.12)).transpose())

    # Plot resulting electron temperature profiles
    plt.figure()
    plt.plot(plasma.el_temp.sel(t=slice(0.02, 0.12)).transpose())

    # Plot resulting ion temperature profiles
    plt.figure()
    plt.plot(plasma.ion_temp.sel(element="h").sel(t=slice(0.02, 0.12)).transpose())

    # Plot lines of sights and equilibrium on (R, z) plane
    plt.figure()
    t = plasma.t[4]
    levels = [0.1, 0.3, 0.5, 0.7, 0.95]
    plasma.equilibrium.rho.sel(t=t, method="nearest").plot.contour(levels=levels)
    for diag in INTERFEROMETERS:
        plt.plot(
            plasma.forward_models[diag].los_transform.R,
            plasma.forward_models[diag].los_transform.z,
            color=colors[diag],
            label=diag,
        )

    plt.plot(
        plasma.forward_models["smmh2"].los_transform.R,
        plasma.forward_models["smmh2"].los_transform.z,
        color="red",
        label="smmh2",
        alpha=0.5,
    )
    plt.plot(
        plasma.forward_models["smmh2_jw"].los_transform.R,
        plasma.forward_models["smmh2_jw"].los_transform.z,
        color="green",
        label="smmh2_jw",
        alpha=0.5,
        linestyle="dashed",
    )
    plt.axis("scaled")
    plt.xlim(0.1, 0.8)
    plt.ylim(-0.6, 0.6)
    plt.legend()

    # Plot rho along line of sight
    plt.figure()
    for diag in INTERFEROMETERS:
        plt.plot(
            plasma.forward_models[diag].los_transform.rho.transpose(),
            color=colors[diag],
            label=diag,
        )

    plt.plot(
        plasma.forward_models["smmh2"].los_transform.rho.transpose(),
        color="red",
        label="smmh2",
        alpha=0.5,
    )

    plt.plot(
        plasma.forward_models["smmh2_jw"].los_transform.rho.transpose(),
        color="green",
        label="smmh2_jw",
        alpha=0.5,
        linestyle="dashed",
    )
    plt.legend()

    # 2D maps of density and radiation
    lz_tot = plasma.lz_tot
    lz_sxr = plasma.lz_sxr

    rho_2d = plasma.equilibrium.rho.sel(t=t, method="nearest")

    el_dens_2d = plasma.el_dens.sel(t=t).interp(rho_poloidal=rho_2d)
    neutral_dens_2d = plasma.neutral_dens.sel(t=t).interp(rho_poloidal=rho_2d)
    el_temp_2d = plasma.el_temp.sel(t=t).interp(rho_poloidal=rho_2d)
    ion_temp_2d = plasma.ion_temp.sel(t=t, element="h").interp(rho_poloidal=rho_2d)
    vtor_2d = plasma.vtor.sel(t=t, element="h").interp(rho_poloidal=rho_2d)

    results = {}
    for elem in plasma.impurities:
        imp_dens_2d = plasma.ion_dens_2d.sel(t=t).sel(element=elem)
        results[f"N{elem}"] = imp_dens_2d

        lz_tot_2d = lz_tot[elem].sel(t=t).interp(rho_poloidal=rho_2d).sum("ion_charges")
        results[f"lz_tot_{elem}"] = lz_tot_2d
        tot_rad_2d = lz_tot_2d * imp_dens_2d * el_dens_2d
        results[f"tot_rad_{elem}"] = tot_rad_2d

        lz_sxr_2d = lz_sxr[elem].sel(t=t).interp(rho_poloidal=rho_2d).sum("ion_charges")
        results[f"lz_sxr_{elem}"] = lz_sxr_2d
        sxr_rad_2d = lz_sxr_2d * imp_dens_2d * el_dens_2d
        results[f"sxr_rad_{elem}"] = sxr_rad_2d

        plt.figure()
        imp_dens_2d.plot()
        rho_2d.plot.contour(levels=levels, alpha=0.5, colors=["white"] * len(levels))
        plt.axis("scaled")
        plt.xlim(0.1, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.title(f"{elem} density at {int(t*1.e3)} ms")

        plt.figure()
        tot_rad_2d.plot()
        rho_2d.plot.contour(levels=levels, alpha=0.5, colors=["white"] * len(levels))
        plt.axis("scaled")
        plt.xlim(0.1, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.title(f"{elem} total radiation at {int(t*1.e3)} ms")

        plt.figure()
        sxr_rad_2d.plot()
        rho_2d.plot.contour(levels=levels, alpha=0.5, colors=["white"] * len(levels))
        plt.axis("scaled")
        plt.xlim(0.1, 0.8)
        plt.ylim(-0.6, 0.6)
        plt.title(f"{elem} SXR radiation at {int(t * 1.e3)} ms")

    plt.ion()
    plt.show()

    results = {
        "pulse": plasma.pulse,
        "t": t,
        "rho": rho_2d,
        "Ne": el_dens_2d,
        "Nh": neutral_dens_2d,
        "Te": el_temp_2d,
        "Ti": ion_temp_2d,
        "Vtor": vtor_2d,
    }

    # pickle.dump(
    #     results, open("/home/marco.sertoli/data/poloidal_asymmetries.pkl", "wb")
    # )


def save_to_pickle(pl, raw_data, data, bckc, pulse=0, name="", force=False):
    picklefile = f"/home/marco.sertoli/data/Indica/{pulse}_{name}_HDA.pkl"

    if (
        os.path.isfile(picklefile)
        and not (os.access(picklefile, os.W_OK))
        and force is True
    ):
        os.chmod(picklefile, 0o744)

    if os.access(picklefile, os.W_OK) or not os.path.isfile(picklefile):
        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))
        os.chmod(picklefile, 0o444)


def load_pickle(pulse, name):
    picklefile = f"/home/marco.sertoli/data/Indica/{pulse}_{name}_HDA.pkl"
    return pickle.load(open(picklefile, "rb"))
