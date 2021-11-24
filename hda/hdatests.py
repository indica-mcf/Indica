from copy import deepcopy

from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import pickle
import xarray as xr

from hda.hdaplot import HDAplot
from hda.hdaworkflow import HDArun
from hda.spline_profiles import Plasma_profs
from hda.read_st40 import ST40data
from hda.plasma import Plasma
import hda.plots as plots
import hda.hda_tree as hda_tree
import hda.physics as ph
from indica.readers import ST40Reader

from hda.diagnostics.spectrometer import XRCSpectrometer
from hda.diagnostics.PISpectrometer import PISpectrometer

plt.ion()

"""
self = pl
forward_model = self.forward_models["xrcs"]
dl = data["xrcs"]["ti_w"].attrs["dl"]
ratio_kw_data = data[diagnostic]["int_k"] / data[diagnostic]["int_w"]
ratio_n3w_data = data[diagnostic]["int_n3"] / data[diagnostic]["int_tot"]
ratio_kw_bckc, ratio_n3w_bckc = [], []
for t in pl.time:
Ne = self.el_dens.sel(t=t)
Nimp = {"ar": self.ion_dens.sel(element="ar").sel(t=t)}
Nh = self.neutral_dens.sel(t=t)
Te = self.el_temp.sel(t=t)*1.5
Ti = self.ion_temp.sel(element="ar").sel(t=t)
_bckc = forward_model(Te, Ne, Nimp=Nimp, Nh=Nh, rho_los=rho_los, dl=dl)
ratio_kw_bckc.append((forward_model.intensity["k"] / forward_model.intensity["w"]).values)
int_tot = (
forward_model.intensity["n3"]
+ forward_model.intensity["n4"]
+ forward_model.intensity["n5"]
+ forward_model.intensity["w"]
)
ratio_n3w_bckc.append((forward_model.intensity["n3"] / int_tot).values)

ratio_kw_bckc = xr.DataArray(ratio_kw_bckc, coords=[("t", self.t)])
ratio_n3w_bckc = xr.DataArray(ratio_n3w_bckc, coords=[("t", self.t)])

ratio_n3w_data.plot(marker="o", color="red", label="n3w")
ratio_kw_data.plot(marker="x", linestyle="dashed", color="red", label="kw")
ratio_n3w_bckc.plot(marker="o", color="black", label="n3w")
ratio_kw_bckc.plot(marker="x", linestyle="dashed", color="black", label="kw")

"""


def plasma_workflow(
    pulse=9229,
    tstart=0.025,
    tend=0.14,
    dt=0.015,
    diagn_ne="smmh1",
    diagn_te="xrcs",
    quant_ne="ne",
    quant_te="te_kw",
    quant_ti="ti_w",
    quant_ar="int_w",
    main_ion="h",
    impurities=("c", "ar", "he"),
    imp_conc=(0.03, 0.001, 0.01),
    cal_ar=1.0e13,
    write=False,
    save_pickle=False,
    savefig=False,
    modelling=True,
    recover_dens=False,
    ne_peaking=None,
    marchuk=True,
    run_name="RUN40",
    descr=f"New profile shapes and ionisation balance",  # descr = "Experimental evolution of the Ar concentration"
    name="",
    xrcs_time=True,
    use_ratios=False,
    use_satellites=False,
    calc_error=False,
):
    """
    New framework for running HDA

    Pulses analysed up to now
    Jari's beam tests
        res = tests.plasma_workflow(pulse=9188, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...
        res = tests.plasma_workflow(pulse=9189, tstart=0.02, tend=0.17, dt=0.010, diagn_ne="nirh1_bin") #problem with Te
        res = tests.plasma_workflow(pulse=9190, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...
        res = tests.plasma_workflow(pulse=9191, tstart=0.02, tend=0.17, dt=0.007, diagn_ne="nirh1_bin") #not good...

    NBI [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599, 9184, 9219, 9221, 9229]
        res = tests.plasma_workflow(pulse=8599, tstart=0.02, tend=0.17, dt=0.01, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9219, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9221, tstart=0.02, tend=0.1, dt=0.007, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9229, tstart=0.02, tend=0.12, dt=0.007, diagn_ne="smmh1", quant_te="te_n3w")

    Ohmic [8385, 8386, 8387, 8390, 8405, 8458, 8909, 9184]
        res = tests.plasma_workflow(pulse=8387, tstart=0.035, tend=0.17, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)
        res = tests.plasma_workflow(pulse=8458, tstart=0.025, tend=0.12, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)
        res = tests.plasma_workflow(pulse=8909, tstart=0.025, tend=0.13, dt=0.01, diagn_ne="nirh1_bin")
        res = tests.plasma_workflow(pulse=9184, tstart=0.025, tend=0.2, dt=0.015, diagn_ne="nirh1_bin", recover_dens=True)

    Bremsstrahlung & Ar concentration calculation
        res = tests.plasma_workflow(pulse=9408, tstart=0.02, tend=0.11, dt=0.007, diagn_ne="smmh1", ne_peaking=1, quantity_te="te_n3w")

    """

    # Read raw data
    raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
    raw.get_all()  # smmh1_rev=2)
    raw_data = raw.data
    if "xrcs" in raw_data.keys() and xrcs_time:
        time = raw_data["xrcs"]["ti_w"].t.values
        tind = np.argwhere((time >= tstart) * (time <= tend)).flatten()
        tstart = time[tind[0]]
        tend = time[tind[-1]]
        dt = time[1] - time[0]

    # Plasma class
    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
    data = pl.build_data(raw_data, pulse=pulse)
    data = pl.apply_limits(data, "xrcs", err_lim=(np.nan, np.nan))

    if ne_peaking is not None:
        pl.Ne_prof.peaking = ne_peaking
        pl.Ne_prof.build_profile()

    # Impose impurity concentration and calculate dilution
    for i, elem in enumerate(impurities):
        if elem in pl.ion_conc.element:
            pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]

    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.build_atomic_data()
    pl.calculate_geometry()
    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk)
    if "princeton" in raw_data:
        pl.forward_models["princeton"] = PISpectrometer()

    # Rescale density to match interferometer
    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne
    )

    # Default impurity concentrations
    pl.calc_imp_dens()
    #
    # return pl, raw_data, data, bckc

    # Build temperature profiles to match XRCS
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        use_ratios=use_ratios,
        use_satellites=use_satellites,
        calc_error=calc_error,
    )

    # return pl, raw_data, data, bckc

    # Average charge known Te
    pl.calc_meanz()

    # Default impurity concentrations
    pl.calc_imp_dens()

    # Ar density from intensity of w line
    bckc = pl.match_xrcs_intensity(
        data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, cal=cal_ar,
    )

    # Quasineutrality
    pl.calc_main_ion_dens()
    pl.calc_zeff()
    bckc = pl.calc_pressure(data=data, bckc=bckc)

    # Recover density to match stored energy
    # if recover_dens:
    #     pl.recover_density(data)
    #     pl.calc_meanz()
    #     pl.calc_imp_dens()
    #     bckc = pl.match_xrcs_intensity(
    #         data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, mult=mult_ar,
    #     )
    #     pl.calc_main_ion_dens()
    #     pl.calc_zeff()
    #     bckc = pl.calc_pressure(data=data, bckc=bckc)

    pl.calc_rad_power()

    # pl.build_current_density()
    # pl.calc_magnetic_field()
    # pl.calc_beta_poloidal()
    # pl.calc_vloop()

    bckc = pl.interferometer(data, bckc=bckc)
    bckc = pl.bremsstrahlung(data, bckc=bckc)

    # return pl, raw_data, data, bckc

    # Compare diagnostic data with back-calculated data
    plots.compare_data_bckc(
        data, bckc, raw_data=raw_data, pulse=pl.pulse, savefig=savefig, name=name,
    )
    plots.profiles(pl, bckc=bckc, savefig=savefig, name=name)
    plots.time_evol(pl, data, bckc=bckc, savefig=savefig, name=name)

    if write:
        if modelling:
            pulse = pl.pulse + 25000000
        else:
            pulse = pl.pulse
        hda_tree.write(
            pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
        )

    if save_pickle:
        picklefile = f"/home/marco.sertoli/data/Indica/{pl.pulse}_HDA.pkl"
        pickle.dump([pl, raw_data, data, bckc], open(picklefile, "wb"))

    return pl, raw_data, data, bckc


def xrcs_sensitivity():
    """
    Test optimisation methods and use of different Te measurements but identical profile shapes
    on final results (Te and Ti + Wth)

    Test for 9539 saved to 25009539 runs 40, 41, 42, 43
    """

    def propagate(pl, data, bckc, quant_ar="int_w", cal_ar=1.0e13):
        pl.calc_meanz()
        pl.calc_imp_dens()
        bckc = pl.match_xrcs_intensity(
            data, bckc=bckc, diagnostic="xrcs", quantity=quant_ar, cal=cal_ar,
        )
        pl.calc_main_ion_dens()
        pl.calc_zeff()
        bckc = pl.calc_pressure(data=data, bckc=bckc)
        pl.calc_rad_power()
        bckc = pl.interferometer(data, bckc=bckc)
        bckc = pl.bremsstrahlung(data, bckc=bckc)
        return pl, bckc

    # W-line emission moments, n3w PPAC result
    res = plasma_workflow(
        pulse=9539,
        tstart=0.02,
        tend=0.12,
        diagn_ne="smmh1",
        quant_te="te_n3w",
        imp_conc=(0.03, 0.001, 0.01),
        marchuk=True,
        xrcs_time=True,
        use_ratios=False,
        use_satellites=False,
        calc_error=False,
    )
    pl, raw_data, data, bckc = res

    pulse = pl.pulse + 25000000

    run_name = "RUN40"
    descr = f"Moment analysis w-line, Te(n3w)"
    hda_tree.write(
        pl, pulse, "HDA", data=data, bckc=bckc, descr=descr, run_name=run_name
    )

    bckc_mom_n3 = deepcopy(bckc)
    pl_mom_n3 = deepcopy(pl)

    # W-line emission moments, kw PPAC result
    bckc_mom_k = deepcopy(bckc)
    pl_mom_k = deepcopy(pl)
    bckc_mom_k = pl_mom_k.match_xrcs_temperatures(
        data,
        bckc=bckc_mom_k,
        quantity_te="te_kw",
        use_ratios=False,
        use_satellites=False,
        calc_error=False,
    )
    pl_mom_k, bckc_mom_k = propagate(
        pl_mom_k, data, bckc_mom_k, quant_ar="int_w", cal_ar=1.0e13
    )
    run_name = "RUN41"
    descr = f"Moment analysis w-line, Te(kw)"
    hda_tree.write(
        pl_mom_k,
        pulse,
        "HDA",
        data=data,
        bckc=bckc_mom_k,
        descr=descr,
        run_name=run_name,
    )

    # W-line & satellite convoluted
    bckc_mom_k_satellites = deepcopy(bckc)
    pl_mom_k_satellites = deepcopy(pl)
    bckc_mom_k_satellites = pl_mom_k_satellites.match_xrcs_temperatures(
        data,
        bckc=bckc_mom_k_satellites,
        quantity_te="te_kw",
        use_ratios=False,
        use_satellites=True,
        calc_error=False,
    )
    pl_mom_k_satellites, bckc_mom_k_satellites = propagate(
        pl_mom_k_satellites,
        data,
        bckc_mom_k_satellites,
        quant_ar="int_w",
        cal_ar=1.0e13,
    )
    run_name = "RUN42"
    descr = f"Moment analysis w-line * satellite, Te(kw)"
    hda_tree.write(
        pl_mom_k_satellites,
        pulse,
        "HDA",
        data=data,
        bckc=bckc_mom_k_satellites,
        descr=descr,
        run_name=run_name,
    )

    # k/w ratio
    bckc_ratio_kw = deepcopy(bckc)
    pl_ratio_kw = deepcopy(pl)
    bckc_ratio_kw = pl_ratio_kw.match_xrcs_temperatures(
        data,
        bckc=bckc_ratio_kw,
        quantity_te="te_kw",
        use_ratios=True,
        use_satellites=False,
        calc_error=False,
    )
    pl_ratio_kw, bckc_ratio_kw = propagate(
        pl_ratio_kw, data, bckc_ratio_kw, quant_ar="int_w", cal_ar=1.0e13
    )
    run_name = "RUN43"
    descr = f"Line ratios, Te(kw)"
    hda_tree.write(
        pl_ratio_kw,
        pulse,
        "HDA",
        data=data,
        bckc=bckc_ratio_kw,
        descr=descr,
        run_name=run_name,
    )

    # n3/w ratio
    bckc_ratio_n3w = deepcopy(bckc)
    pl_ratio_n3w = deepcopy(pl)
    bckc_ratio_n3w = pl_ratio_n3w.match_xrcs_temperatures(
        data,
        bckc=bckc_ratio_n3w,
        quantity_te="te_n3w",
        use_ratios=True,
        use_satellites=False,
        calc_error=False,
    )
    pl_ratio_n3w, bckc_ratio_n3w = propagate(
        pl_ratio_n3w, data, bckc_ratio_n3w, quant_ar="int_w", cal_ar=1.0e13
    )
    run_name = "RUN44"
    descr = f"Line ratios, Te(n3w)"
    hda_tree.write(
        pl_ratio_n3w,
        pulse,
        "HDA",
        data=data,
        bckc=bckc_ratio_n3w,
        descr=descr,
        run_name=run_name,
    )

    plt.figure()
    pl_mom_n3.el_temp.sel(rho_poloidal=0).plot(
        color="red", label="moments (w) n3w measurement", alpha=0.5
    )
    pl_mom_k.el_temp.sel(rho_poloidal=0).plot(
        color="orange", label="moments (w) kw measurement", alpha=0.5
    )
    pl_mom_k_satellites.el_temp.sel(rho_poloidal=0).plot(
        color="cyan", label="moments (w * satellite) kw measurement", alpha=0.5
    )
    pl_ratio_kw.el_temp.sel(rho_poloidal=0).plot(
        color="blue", label="line ratios (k/w)", alpha=0.5
    )
    pl_ratio_n3w.el_temp.sel(rho_poloidal=0).plot(
        color="black", label="line ratios (n3/(n3+n4+n5+w))", alpha=0.5
    )
    plt.legend()


def sawtoothing(
    pulse=9229,
    tstart=0.072,
    tend=0.078,
    dt=0.0005,
    t_pre=0.0735,
    t_post=0.0745,
    r_inv=0.1,
    diagn_ne="smmh1",
    diagn_te="xrcs",
    quant_ne="ne",
    quant_te="te_kw",
    quant_ti="ti_w",
    quant_ar="int_w",
    main_ion="h",
    impurities=("c", "ar", "he"),
    imp_conc=(0.03, 0.001, 0.01),
    cal_ar=1.0e13,
    marchuk=True,
    use_ratios=False,
    calc_error=False,
):
    """
    tests.sawtoothing()
    tests.sawtoothing(t_pre=0.081, t_post=0.0825, tstart=0.078, tend=0.086)

    pulse=9229
    tstart=0.072
    tend=0.078
    dt=0.0005
    t_pre=0.0735
    t_post=0.0745
    r_inv=0.1
    diagn_ne="smmh1"
    diagn_te="xrcs"
    quant_ne="ne"
    quant_te="te_kw"
    quant_ti="ti_w"
    quant_ar="int_w"
    main_ion="h"
    impurities=("c", "ar", "he")
    imp_conc=(0.03, 0.001, 0.01)
    marchuk = True
    cal_ar=1.e13
    calc_error=False
    """

    raw = ST40data(pulse, tstart - 0.01, tend + 0.01)
    raw.get_all()
    raw_data = raw.data

    bckc = {}
    elements = list(main_ion)
    elements.extend(list(impurities))

    pl = Plasma(tstart=tstart, tend=tend, dt=dt, elements=elements)
    data = pl.build_data(raw_data, pulse=pulse)

    for i, elem in enumerate(impurities):
        if elem in pl.ion_conc.element:
            pl.ion_conc.loc[dict(element=elem)] = imp_conc[i]

    # Find Ar density sawtooth crash, assuming Te stays constant
    if "xrcs" in raw_data:
        pl.forward_models["xrcs"] = XRCSpectrometer(marchuk=marchuk)
    if "princeton" in raw_data:
        pl.forward_models["princeton"] = PISpectrometer()

    pl.set_neutral_density(y1=1.0e15, y0=1.0e9)
    pl.build_atomic_data()
    pl.calculate_geometry()

    # Reference times before and after the crash, inversion radius in rho
    t_pre = pl.time.values[np.argmin(np.abs(pl.time - t_pre).values)]
    t_post = pl.time.values[np.argmin(np.abs(pl.time - t_post).values)]
    R_inv = pl.equilibrium.rmag.sel(t=t_post, method="nearest")
    z_inv = pl.equilibrium.zmag.sel(t=t_post, method="nearest") + r_inv
    rho_inv, _, _ = pl.equilibrium.flux_coords(R_inv, z_inv, t=t_post)

    # Find electron density sawtooth crash
    ne_pre_data = data["smmh1"]["ne"].sel(t=t_pre).values
    ne_post_data = data["smmh1"]["ne"].sel(t=t_post).values

    # Test different peaking factors to match both pre and post crash profiles
    pl.Ne_prof.y1 = 0.5e19
    pl.Ne_prof.wcenter = rho_inv.values / 2.0
    pl.Ne_prof.wped = 5
    pl.Ne_prof.peaking = 1.0
    pl.Ne_prof.build_profile()
    bckc = pl.match_interferometer(
        data, bckc=bckc, diagnostic=diagn_ne, quantity=quant_ne, time=[t_pre, t_post]
    )
    volume = pl.volume.sel(t=t_post)

    scan = np.linspace(1.0, 2.5, 21)
    Ne_pre, Ne_post = [], []
    ne_pre_bckc, ne_post_bckc = [], []
    for s in scan:
        pre = deepcopy(pl.Ne_prof)
        pre.peaking = s
        pre.build_profile()
        pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
        ne_pre_tmp = pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre)
        pre.y0 *= (ne_pre_data / ne_pre_tmp).values
        pre.build_profile()
        Ne_pre.append(deepcopy(pre))
        pl.el_dens.loc[dict(t=t_pre)] = pre.yspl.values
        ne_pre_bckc.append(pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_pre).values)
        post = deepcopy(pre)
        post.sawtooth_crash(rho_inv.values, volume)
        Ne_post.append(deepcopy(post))
        pl.el_dens.loc[dict(t=t_post)] = post.yspl.values
        ne_post_bckc.append(
            pl.calc_ne_los_int(data[diagn_ne][quant_ne], t=t_post).values
        )

    ne_post_bckc = np.array(ne_post_bckc)

    plt.figure()
    raw_data["smmh1"]["ne"].plot()
    data["smmh1"]["ne"].plot(linewidth=3)
    ylim = plt.ylim()
    plt.vlines(t_post, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.plot(t_pre, ne_pre_data, marker="o", color="black")
    plt.plot(t_post, ne_post_data, marker="o", color="red")

    colors = cm.rainbow(np.linspace(0, 1, len(scan)))
    for i, s, in enumerate(scan):
        plt.plot(t_post, ne_post_bckc[i], "x", color=colors[i])

    ind = np.argmin(np.abs(ne_post_data - ne_post_bckc))
    plt.figure()
    for i, s, in enumerate(scan):
        Ne_pre[i].yspl.plot(color=colors[i], alpha=0.5)
        Ne_post[i].yspl.plot(color=colors[i], linestyle="dashed", alpha=0.5)

    Ne_pre[ind].yspl.plot(color="black", marker="o")
    Ne_post[ind].yspl.plot(color=colors[ind], linestyle="dashed", marker="o")

    # Fix electron density crash to best matching
    Ne_pre = Ne_pre[ind]
    Ne_post = Ne_post[ind]

    # Build tempertaure profiles to match XRCS using standard shapes
    bckc = pl.match_xrcs_temperatures(
        data,
        bckc=bckc,
        diagnostic=diagn_te,
        quantity_te=quant_te,
        quantity_ti=quant_ti,
        time=[t_pre, t_post],
        use_ratios=use_ratios,
        use_satellites=use_satellites,
        calc_error=calc_error,
    )

    # Scan impurity profile shape to match observed crash in XRCS w-line intensity
    t_xrcs = raw_data["xrcs"]["int_w"].t
    t_pre_xrcs = np.nanmax(xr.where(t_xrcs < t_pre, t_xrcs, np.nan).values)
    t_post_xrcs = np.nanmin(xr.where(t_xrcs > t_post, t_xrcs, np.nan).values)
    t_pre_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_pre_xrcs).values)]
    t_post_xrcs = pl.time.values[np.argmin(np.abs(pl.time - t_post_xrcs).values)]
    pl.el_temp.loc[dict(t=t_pre_xrcs)] = pl.el_temp.sel(t=t_pre)
    pl.el_temp.loc[dict(t=t_post_xrcs)] = pl.el_temp.sel(t=t_post)
    pl.el_dens.loc[dict(t=t_pre_xrcs)] = pl.el_dens.sel(t=t_pre)
    pl.el_dens.loc[dict(t=t_post_xrcs)] = pl.el_dens.sel(t=t_post)

    int_pre_data = data["xrcs"]["int_w"].sel(t=t_pre_xrcs, method="nearest").values
    int_post_data = data["xrcs"]["int_w"].sel(t=t_post_xrcs, method="nearest").values

    pl.Nimp_prof.peaking = Ne_pre.peaking
    pl.Nimp_prof.wcenter = rho_inv.values / 2.0
    pl.Nimp_prof.wped = 5
    pl.Nimp_prof.build_profile()
    volume = pl.volume.sel(t=t_post)

    pl.calc_meanz()
    pl.calc_imp_dens()

    scan = np.linspace(1.0, 2.5, 21)
    Nimp_pre, Nimp_post = [], []
    int_pre_bckc, int_post_bckc = [], []
    elem = "ar"

    # TODO: bypassing the renormalisation of impurity densituy setting niter=1
    for s in scan:
        pre = deepcopy(pl.Nimp_prof)
        pre.peaking = s
        pre.build_profile()
        Nimp_pre.append(deepcopy(pre))
        pl.Nimp_prof = deepcopy(pre)
        pl.calc_imp_dens()
        bckc = pl.match_xrcs_intensity(
            data,
            bckc=bckc,
            diagnostic="xrcs",
            quantity=quant_ar,
            cal=cal_ar,
            time=[t_pre_xrcs, t_post_xrcs],
        )
        int_pre_bckc.append(bckc["xrcs"][quant_ar].sel(t=t_pre_xrcs).values)
        Nimp = pl.ion_dens.sel(element=elem).sel(t=t_pre_xrcs)
        pre.y0 = Nimp.sel(rho_poloidal=0).values
        pre.y1 = Nimp.sel(rho_poloidal=1).values
        pre.yend = Nimp.sel(rho_poloidal=Nimp.rho_poloidal.max()).values
        pre.build_profile()
        post = deepcopy(pre)
        post.sawtooth_crash(rho_inv.values, volume)
        Nimp_post.append(deepcopy(post))

        pl.ion_dens.loc[dict(element=elem, t=t_post_xrcs)] = post.yspl.values
        bckc = pl.match_xrcs_intensity(
            data,
            bckc=bckc,
            diagnostic="xrcs",
            quantity=quant_ar,
            cal=cal_ar,
            time=[t_pre_xrcs, t_post_xrcs],
            niter=1,
        )
        int_post_bckc.append(bckc["xrcs"][quant_ar].sel(t=t_post_xrcs).values)

    int_pre_bckc = np.array(int_pre_bckc)
    int_post_bckc = np.array(int_post_bckc)

    plt.figure()
    raw_data["xrcs"]["int_w"].plot()
    data["xrcs"]["int_w"].plot(linewidth=3)
    ylim = plt.ylim()
    plt.vlines(t_post_xrcs, ylim[0], ylim[1], color="black", linestyle="dashed")
    plt.plot(t_pre_xrcs, int_pre_data, marker="o", color="black")
    plt.plot(t_post_xrcs, int_post_data, marker="o", color="red")

    colors = cm.rainbow(np.linspace(0, 1, len(scan)))
    for i, s, in enumerate(scan):
        plt.plot(t_post_xrcs, int_post_bckc[i], "x", color=colors[i])

    ind = np.argmin(np.abs(int_post_data - int_post_bckc))
    plt.figure()
    for i, s, in enumerate(scan):
        Nimp_pre[i].yspl.plot(color=colors[i], alpha=0.5)
        Nimp_post[i].yspl.plot(color=colors[i], linestyle="dashed", alpha=0.5)

    Nimp_pre[ind].yspl.plot(color="black", marker="o")
    Nimp_post[ind].yspl.plot(color=colors[ind], linestyle="dashed", marker="o")


def compare_astra(pulse=8574, tstart=0.02, tend=0.14, revision=105, interf="nirh1"):
    pulse = 8574
    tstart = 0.02
    tend = 0.14
    revision = 105
    reader = ST40Reader(int(pulse + 25.0e6), tstart, tend, tree="astra")
    astra = reader.get("", "astra", revision)
    astra["ne"] *= 1.0e19
    astra["ni"] *= 1.0e19
    astra["te"] *= 1.0e3
    astra["ti"] *= 1.0e3
    # rho_poloidal = astra["p"].rho_poloidal
    # rho_toroidal = astra["ne"].rho_toroidal

    time = astra["te"].t
    tstart, tend = time[0], time[-1]
    dt = time[1] - time[0]
    pl, raw_data, data, bckc = plasma_workflow(
        pulse=pulse, tstart=tstart, tend=tend, dt=dt
    )


def best_astra(
    pulse=8383, tstart=0.02, tend=0.12, hdarun=None, write=False, force=False
):
    """
    Best profile shapes from ASTRA runs of 8383 applied to database
    """
    ohmic_pulses = [8385, 8386, 8387, 8390, 8405, 8458]  # 8401
    nbi_pulses = [8338, 8373, 8374, 8574, 8575, 8582, 8583, 8597, 8598, 8599]  #

    pulses = np.sort(np.concatenate((np.array(ohmic_pulses), np.array(nbi_pulses))))
    if pulse is not None:
        pulses = [pulse]
    for pulse in pulses:
        interf = "nirh1"
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=tstart, tend=tend)

        # Rebuild temperature profiles
        hdarun.profiles_nbi()
        profs_spl = Plasma_profs(hdarun.data.time)

        # Rescale to match XRCS measurements
        hdarun.data.match_xrcs(profs_spl=profs_spl)

        # Recalculate average charge, dilution, Zeff, total pressure
        hdarun.data.calc_meanz()
        hdarun.data.calc_main_ion_dens(fast_dens=False)
        hdarun.data.impose_flat_zeff()
        hdarun.data.calc_main_ion_dens(fast_dens=False)
        hdarun.data.calc_zeff()
        hdarun.data.calc_pressure()

        descr = f"Best profile shapes from ASTRA {pulse}, c_C=3%"
        run_name = "RUN30"
        plt.close("all")
        hdarun.plot()
        if write:
            hdarun.write(hdarun.data, descr=descr, run_name=run_name, force=force)
        else:
            return hdarun


def scan_profile_shape(pulse=8383, hdarun=None, write=False):
    """
    Fix edge plasma parameters (rho > 0.8) and scan profile shapes
    """

    interf = "nirh1"
    if hdarun is None:
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)

    # Temperature profile shape scan, flat density
    hdarun.profiles_ohmic()

    te_flat = deepcopy(hdarun)
    te_peak1 = deepcopy(hdarun)
    te_peak2 = deepcopy(hdarun)

    profs_spl = Plasma_profs(te_flat.data.time)
    te_flat.data.match_xrcs(profs_spl=profs_spl)
    te_flat.data.calc_pressure()
    descr = "Flat density, flat temperature, c_C=3%"
    run_name = "RUN10"
    if write == True:
        te_flat.write(te_flat.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
    te_peak1.data.match_xrcs(profs_spl=profs_spl)
    te_peak1.data.calc_pressure()
    descr = "Flat density, peaked temperature, c_C=3%"
    run_name = "RUN11"
    if write == True:
        te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
    profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
    profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
    te_peak2.data.match_xrcs(profs_spl=profs_spl)
    te_peak2.data.calc_pressure()
    descr = "Flat density, very peaked temperature, c_C=3%"
    run_name = "RUN12"
    if write == True:
        te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)

    flat_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}

    # Peaked density
    hdarun.profiles_nbi()

    te_flat = deepcopy(hdarun)
    te_peak1 = deepcopy(hdarun)
    te_peak2 = deepcopy(hdarun)

    profs_spl = Plasma_profs(te_flat.data.time)
    te_flat.data.match_xrcs(profs_spl=profs_spl)
    te_flat.data.calc_pressure()
    descr = "Peaked density, flat temperature, c_C=3%"
    run_name = "RUN20"
    if write == True:
        te_flat.write(te_flat.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(2.0, dim_lim=(0, 0))
    te_peak1.data.match_xrcs(profs_spl=profs_spl)
    te_peak1.data.calc_pressure()
    descr = "Peaked density, peaked temperature, c_C=3%"
    run_name = "RUN21"
    if write == True:
        te_peak1.write(te_peak1.data, descr=descr, run_name=run_name)

    profs_spl.el_temp.scale(0.5, dim_lim=(0.7, 0.98))
    profs_spl.ion_temp.scale(2.0, dim_lim=(0, 0))
    profs_spl.ion_temp.scale(0.5, dim_lim=(0.7, 0.98))
    te_peak2.data.match_xrcs(profs_spl=profs_spl)
    te_peak2.data.calc_pressure()
    descr = "Peaked density, very peaked temperature, c_C=3%"
    run_name = "RUN22"
    if write == True:
        te_peak2.write(te_peak2.data, descr=descr, run_name=run_name)

    peaked_dens = {"te_flat": te_flat, "te_peak1": te_peak1, "te_peak2": te_peak2}

    if not write:
        return flat_dens, peaked_dens


def ohmic_pulses(write=False, interf="smmh1", match_kinetic=False):
    # pulses = [8385, 8386, 8387, 8390, 8401, 8405, 8458]
    for pulse in pulses:
        hdarun = HDArun(pulse=pulse, interf=interf, tstart=0.02, tend=0.1)
        hdarun.profiles_ohmic()
        if match_kinetic:
            hdarun.data.calc_pressure()
            descr = "New profile shapes, match kinetic profiles only, c_C=3%"
            run_name = "RUN01"
        else:
            hdarun.match_energy()
            descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
            run_name = "RUN05"
        if write == True:
            hdarun.write(hdarun.bckc, descr=descr, run_name=run_name)
        else:
            hdarun.plot()

    return hdarun


def NBI_pulses(write=False, interf="smmh1", match_kinetic=False):
    pulses = [8338, 8574, 8575, 8582, 8583, 8597, 8598, 8599]
    interf = ["nirh1"] * len(pulses)
    for i, pulse in enumerate(pulses):
        plt.close("all")
        hdarun = HDArun(
            pulse=pulse, interf=interf[i], tstart=0.015, tend=0.14, dt=0.015
        )
        hdarun.profiles_nbi()
        if match_kinetic:
            hdarun.data.calc_pressure()
            descr = "New profile shapes, match kinetic measurements only, c_C=3%"
            run_name = "RUN01"
        else:
            hdarun.match_energy()
            descr = "New profile shapes, adapt Ne to match Wmhd, c_C=3%"
            run_name = "RUN05"
        if write == True:
            hdarun.write(hdarun.data, descr=descr, run_name=run_name)
        else:
            hdarun.plot()
            _ = input("press...")

    return hdarun


def test_low_edge_temperature(hdarun, zeff=False):

    # low temperature edge
    hdarun.initialize_bckc()
    te_0 = 1.0e3
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 15.0,
        x_ped=0.9,
        w_core=0.2,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()
    elements = hdarun.bckc.elements
    main_ion = hdarun.bckc.main_ion
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values
        ti_0 = (
            hdarun.bckc.ion_temp.sel(element=main_ion)
            .sel(t=t)
            .sel(rho_poloidal=0)
            .values
        )
        for elem in elements:
            hdarun.bckc.ion_temp.loc[dict(t=t, element=elem)] = (
                hdarun.bckc.profs.te * ti_0
            ).values

    hdarun.bckc.match_xrcs()
    hdarun.bckc.simulate_spectrometers()

    # hdarun.recover_zeff(optimize="density")

    hdarun.bckc.propagate_parameters()
    # hdarun.recover_density()

    hdarun.plot()


def rabbit_ears(hdarun: HDArun):

    hdarun.initialize_bckc()
    ne_0 = hdarun.bckc.profs.ne.sel(rho_poloidal=0).values
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_density(
        x_0=0.7,
        y_0=ne_0,
        y_ped=ne_0 / 4.0,
        x_ped=0.95,
        w_core=0.1,
        datatype=("density", "electron"),
    )

    for t in hdarun.bckc.time:
        hdarun.bckc.el_dens.loc[dict(t=t)] = hdarun.bckc.profs.ne.values
    hdarun.bckc.match_interferometer(interf)

    # hdarun.recover_density()

    hdarun.plot()


def test_peaked_profiles(hdarun, zeff=False):
    hdarun.initialize_bckc()
    hdarun.recover_density()
    if zeff:
        hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    broad = hdarun.bckc

    # Peaked profiles
    hdarun.initialize_bckc()
    te_0 = 1.0e3
    hdarun.bckc.profs.te = hdarun.bckc.profs.build_temperature(
        y_0=te_0,
        y_ped=te_0 / 15.0,
        x_ped=0.9,
        w_core=0.3,
        datatype=("temperature", "electron"),
    )
    hdarun.bckc.profs.te /= hdarun.bckc.profs.te.max()

    ne_0 = 5.0e19
    hdarun.bckc.profs.ne = hdarun.bckc.profs.build_temperature(
        y_0=ne_0,
        y_ped=ne_0 / 15.0,
        x_ped=0.9,
        w_core=0.3,
        datatype=("density", "electron"),
    )
    hdarun.bckc.profs.ne /= hdarun.bckc.profs.ne.max()
    for t in hdarun.bckc.time:
        te_0 = hdarun.bckc.el_temp.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_temp.loc[dict(t=t)] = (hdarun.bckc.profs.te * te_0).values

        ne_0 = hdarun.bckc.el_dens.sel(t=t).sel(rho_poloidal=0).values
        hdarun.bckc.el_dens.loc[dict(t=t)] = (hdarun.bckc.profs.ne * ne_0).values

    hdarun.bckc.build_current_density()
    hdarun.recover_density()
    if zeff:
        hdarun.recover_zeff(optimize="density")
    hdarun.bckc.simulate_spectrometers()
    peaked = hdarun.bckc

    HDAplot(broad, peaked)


def test_current_density(hdarun):
    """Trust all measurements, find shape to explain data"""

    # L-mode profiles

    # Broad current density
    hdarun.initialize_bckc()
    hdarun.bckc.build_current_density(sigm=0.8)
    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    broad = deepcopy(hdarun.bckc)

    # Peaked current density
    hdarun.initialize_bckc()
    hdarun.bckc.build_current_density(sigm=0.2)
    hdarun.recover_density()
    hdarun.recover_zeff(optimize="density")
    peaked = deepcopy(hdarun.bckc)

    HDAplot(broad, peaked)
