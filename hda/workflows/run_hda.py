from hda.models.xray_crystal_spectrometer import XRCSpectrometer
from hda.models.interferometer import Interferometer
from hda.manage_data import bin_data_in_time, map_on_equilibrium, initialize_bckc
from hda.models.plasma import Plasma
import hda.profiles as profiles
from hda.read_st40 import ST40data
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates, LinesOfSightTransform
from indica.converters.line_of_sight import LineOfSightTransform
from indica.provenance import get_prov_attribute

from hda.optimizations.interferometer import match_interferometer_los_int
from hda.optimizations.xray_crystal_spectrometer import (
    match_line_ratios,
    match_ion_temperature,
)

from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np

plt.ion()

"""
Workflow designed to replicate the results of the pre-refactored HDA as implemented
in hdatests.test_hda
"""


def run_hda(
    pulse: int = 9780,
    impurities: tuple = ("c", "ar", "he"),
    imp_conc: tuple = (0.03, 0.001, 0.01),
):
    tstart = 0.025
    tend = 0.14
    dt = 0.015
    diagnostic_ne = "smmh1"
    diagnostic_te = "xrcs"
    diagnostic_ti = "xrcs"
    quantities_ne = ["ne"]
    quantities_te = ["int_k/int_w"]  # ["int_k/int_w", "int_n3/int_w", "int_n3/int_tot"]
    quantities_ti = ["ti_w"]
    lines_ti = ["w"]
    quantities_ar = ["int_w"]
    main_ion = "h"
    equilibrium_diagnostic = "efit"
    extrapolate = "constant"
    marchuk = True
    fast = True
    plot = True
    forward_models = {}

    # Initialize plasma class and assign equilibrium related objects
    pl = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        impurities=impurities,
        imp_conc=imp_conc,
        pulse=pulse,
    )
    raw_data = None

    # Read raw data
    if pulse is not None:
        raw = ST40data(pulse, tstart - dt / 2, tend + dt / 2)
        raw_data = raw.get_all()

        # Initialize equilibrium and flux transform objects
        equilibrium_data = raw_data[equilibrium_diagnostic]
        equilibrium = Equilibrium(equilibrium_data)
        flux_transform = FluxSurfaceCoordinates("poloidal")
        flux_transform.set_equilibrium(equilibrium)

        pl.set_equilibrium(equilibrium)
        pl.set_flux_transform(flux_transform)
        pl.calculate_geometry()

    # Assign default profile values and objects to plasma class
    profs = profiles.profile_scans(rho=pl.rho)
    pl.Ne_prof = profs["Ne"]["peaked"]
    pl.Te_prof = profs["Te"]["peaked"]
    pl.Ti_prof = profs["Ti"]["peaked"]
    pl.Nimp_prof = profs["Nimp"]["peaked"]
    pl.Vrot_prof = profs["Vrot"]["peaked"]
    pl.set_neutral_density(y1=1.0e16, y0=1.0e13, decay=15)

    # Calculate ionisation balance for standard profiles for interpolation
    pl.Te_prof.y0 = 10.0e3
    pl.Te_prof.build_profile()
    pl.build_atomic_data(
        pl.ADF11,
        full_run=False,
        Te=pl.Te_prof.yspl,
        Ne=pl.Ne_prof.yspl,
        Nh=pl.Nh_prof.yspl,
    )
    return pl

    # Document the provenance of the equilibrium
    # TODO: add the diagnostic and revision info to the Equilibrium class so it can then be read directly
    revision = get_prov_attribute(
        equilibrium_data[list(equilibrium_data)[0]].provenance, "revision"
    )
    pl.optimisation["equil"] = f"{equilibrium_diagnostic}:{revision}"

    # Set Fractional abundance objects to interpolate instead of calculate
    Te = deepcopy(pl.Te_prof.yspl) * 3.0
    Ne = deepcopy(pl.Ne_prof.yspl)
    Nh = deepcopy(pl.Nh_prof.yspl)
    for elem in pl.elements:
        pl.fract_abu[elem](Te=Te, Ne=Ne, Nh=Nh, full_run=True)

    return pl

    # Bin data as required to match plasma class, assign equlibrium objects to
    data = {}
    for kinstr in raw_data.keys():
        data[kinstr] = bin_data_in_time(
            raw_data[kinstr],
            pl.tstart,
            pl.tend,
            pl.dt,
        )
        map_on_equilibrium(data[kinstr], flux_transform=pl.flux_transform)

    # Initialize back-calculated (bckc) diactionary and forward model objects
    bckc = initialize_bckc(data)

    interferometers = ["smmh1", "nirh1"]
    for diag in interferometers:
        forward_models[diag] = Interferometer(name=diag)
        forward_models[diag].set_los_transform(data[diag]["ne"].attrs["transform"])

    forward_models["xrcs"] = XRCSpectrometer(
        marchuk=marchuk, extrapolate=extrapolate, fract_abu=pl.fract_abu, fast=fast
    )
    forward_models["xrcs"].set_los_transform(
        data["xrcs"][list(data["xrcs"])[0]].attrs["transform"]
    )

    # Optimize electron density to match interferometer and temperatures using XRCS
    revision = get_prov_attribute(
        data[diagnostic_ne][list(data[diagnostic_ne])[0]].provenance, "revision"
    )
    pl.optimisation["el_dens"] = f"{diagnostic_ne}:{revision}"
    revision = get_prov_attribute(
        data[diagnostic_te][list(data[diagnostic_te])[0]].provenance, "revision"
    )
    pl.optimisation["el_temp"] = f"{diagnostic_te}:{revision}"

    # Start optimisation
    te0 = 1.0e3
    xrcs = forward_models["xrcs"]
    for i, t in enumerate(pl.t):
        print(float(t))
        # Match chosen interferometer
        _, Ne_prof = match_interferometer_los_int(
            forward_models[diagnostic_ne],
            pl.Ne_prof,
            data[diagnostic_ne],
            t,
            quantities=quantities_ne,
        )
        pl.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
        # Back-calculate the LOS-integral of all the interferometers for consistency checks
        for diagnostic in interferometers:
            bckc_tmp, _ = forward_models[diagnostic].integrate_on_los(
                pl.el_dens.sel(t=t),
                t=t,
            )
            for quantity in quantities_ne:
                bckc[diagnostic][quantity].loc[dict(t=t)] = bckc_tmp[quantity].values

        # Optimize electron temperature for XRCS line ratios
        # Approach optimisation always from the below (te0 < previous time-point)
        pl.calc_imp_dens(t=t)
        if t > pl.t.min():
            te0 = pl.el_temp.sel(t=pl.t[i - 1], rho_poloidal=0).values / 2.0

        Ne = pl.el_dens.sel(t=t)
        Nimp = pl.imp_dens.sel(t=t)
        Nh = pl.neutral_dens.sel(t=t)
        tau = pl.tau.sel(t=t)
        if not np.any(tau > 0):
            tau = None

        bckc_tmp, Te_prof = match_line_ratios(
            xrcs,
            pl.Te_prof,
            data[diagnostic_te],
            t,
            Ne,
            Nimp=Nimp,
            Nh=Nh,
            tau=tau,
            quantities=quantities_te,
            te0=te0,
            bckc=bckc[diagnostic_te],
        )
        pl.el_temp.loc[dict(t=t)] = Te_prof.yspl.values

        # Calculate moment analysis of the electron temperature
        te_kw = xrcs.moment_analysis(pl.el_temp.sel(t=t), t, line="kw")
        bckc["xrcs"]["te_kw"].loc[dict(t=t)] = te_kw
        pos_kw, err_in_kw, err_out_kw = xrcs.calculate_emission_position(t, line="kw")
        te_n3w = xrcs.moment_analysis(pl.el_temp.sel(t=t), t, line="n3w")
        bckc["xrcs"]["te_n3w"].loc[dict(t=t)] = te_n3w
        pos_n3w, err_in_n3w, err_out_n3w = xrcs.calculate_emission_position(
            t, line="n3w"
        )

        bckc_tmp, Ti_prof = match_ion_temperature(
            xrcs,
            pl.Ti_prof,
            data[diagnostic_ti],
            t,
            quantities=quantities_ti,
            lines=lines_ti,
            bckc=bckc[diagnostic_ti],
        )

        for elem in pl.elements:
            pl.ion_temp.loc[dict(t=t, element=elem)] = Ti_prof.yspl.values

    return data, bckc, pl

    # Add invented interferometer with different LOS 20 cm above the SMMH1
    forward_models["smmh2"] = Interferometer(name="smmh2")
    _trans = data["smmh1"]["ne"].attrs["transform"]
    los_transform = LinesOfSightTransform(
        x_start=_trans.x_start.values,
        y_start=_trans.y_start.values,
        z_start=_trans.z_start.values + 0.15,
        x_end=_trans.x_end.values,
        y_end=_trans.y_end.values,
        z_end=_trans.z_end.values + 0.15,
        name="smmh2",
        machine_dimensions=_trans._machine_dims,
    )
    los_transform.set_flux_transform(flux_transform)
    _ = los_transform.convert_to_rho(t=data["smmh1"]["ne"].t)
    forward_models["smmh2"].set_los_transform(los_transform)
    bckc["smmh2"] = {}
    los_integral, _ = forward_models["smmh2"].integrate_on_los(pl.el_dens)
    bckc["smmh2"][quant_ne] = los_integral

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

    los_transform_jw.set_flux_transform(flux_transform)
    _ = los_transform_jw.convert_to_rho(t=data["smmh1"]["ne"].t)
    forward_models["smmh2_jw"] = Interferometer(name="smmh2_jw")
    forward_models["smmh2_jw"].set_los_transform(los_transform_jw)
    bckc["smmh2_jw"] = {}
    los_integral, _ = forward_models["smmh2_jw"].integrate_on_los(pl.el_dens)
    bckc["smmh2_jw"][quant_ne] = los_integral

    # if plot:
    # Plot comparison of raw data, binned data and back-calculated values
    plt.figure()
    colors = {"nirh1": "blue", "smmh1": "purple"}
    for diag in interferometers:
        raw_data[diag][quant_ne].plot(color=colors[diag], label=diag)
        data[diag][quant_ne].plot(color=colors[diag], marker="o")
        bckc[diag][quant_ne].plot(color=colors[diag], marker="x")

    bckc["smmh2"][quant_ne].plot(color="red", marker="D", label="smmh2", alpha=0.5)
    bckc["smmh2_jw"][quant_ne].plot(
        color="green", marker="*", label="smmh2", alpha=0.5, linestyle="dashed"
    )
    plt.legend()

    # Plot resulting density profiles
    plt.figure()
    plt.plot(pl.el_dens.sel(t=slice(0.02, 0.12)).transpose())

    # Plot lines of sights and equilibrium on (R, z) plane
    plt.figure()
    t = 0.05
    levels = [0.1, 0.3, 0.5, 0.7, 0.95]
    equilibrium.rho.sel(t=t, method="nearest").plot.contour(levels=levels)
    for diag in interferometers:
        plt.plot(
            forward_models[diag].los_transform.R,
            forward_models[diag].los_transform.z,
            color=colors[diag],
            label=diag,
        )

    plt.plot(
        forward_models["smmh2"].los_transform.R,
        forward_models["smmh2"].los_transform.z,
        color="red",
        label="smmh2",
        alpha=0.5,
    )
    plt.plot(
        forward_models["smmh2_jw"].los_transform.R,
        forward_models["smmh2_jw"].los_transform.z,
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
    for diag in interferometers:
        plt.plot(
            forward_models[diag].los_transform.rho.transpose(),
            color=colors[diag],
            label=diag,
        )

    plt.plot(
        forward_models["smmh2"].los_transform.rho.transpose(),
        color="red",
        label="smmh2",
        alpha=0.5,
    )

    plt.plot(
        forward_models["smmh2_jw"].los_transform.rho.transpose(),
        color="green",
        label="smmh2_jw",
        alpha=0.5,
        linestyle="dashed",
    )
    plt.legend()
