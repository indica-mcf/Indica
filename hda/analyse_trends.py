"""Read and plot time evolution of various quantities
at identical times in a discharge over a defined pulse range

Example call:

    import hda.analyse_trends as trends
    corr = trends.correlations(8400, 8534, t=[0.03, 0.08])

"""

import getpass
import numpy as np
from indica.readers import ST40Reader
import matplotlib.pylab as plt
from copy import deepcopy

# First pulse after Boronisation
BORONISATION = [8440.5, 8536.5]
GDC = [8544.5, 8546.5, 8547.5, 8548.5, 8549.5]

plt.ion()


class correlations:
    def __init__(self, pulse_start, pulse_end, t=[0.03, 0.08], dt=0.01):
        if type(t) is not list:
            t = [t]
        self.t = np.array(t)
        self.dt = dt

        self.tstart = self.t - self.dt
        self.tend = self.t + self.dt
        self.results = self.read_data(pulse_start, pulse_end)
        # self.plot_trends()

    def init_avantes(self):

        if len(self.t) > 1:
            lines_labels = [
                "h_i_656",
                "he_ii_469",
                "b_v_494",
                "o_iv_306",
                "ar_ii_443",
            ]
        else:
            lines_labels = [
                "h_i_656",
                "he_i_588",
                "he_ii_469",
                "b_ii_207",
                "b_iv_449",
                "b_v_494",
                "o_iii_305",
                "o_iii_327",
                "o_iv_306",
                "o_v_278",
                "ar_ii_488",
                "ar_ii_481",
                "ar_ii_474",
                "ar_ii_443",
                "ar_iii_331",
                "ar_iii_330",
                "ar_iii_329",
            ]

        self.lines_labels = lines_labels

    def init_results_dict(self):
        results = {
            "pulses": [],
            "puff_prefill": [],
            "puff_total": [],
            "ipla": {"avrg": [], "stdev": []},
            "wp": {"avrg": [], "stdev": []},
            "nirh1": {"avrg": [], "stdev": []},
            "smmh1": {"avrg": [], "stdev": []},
            "brems_pi": {"avrg": [], "stdev": []},
            "brems_mp": {"avrg": [], "stdev": []},
            "xrcs_te": {"avrg": [], "stdev": []},
            "xrcs_ti": {"avrg": [], "stdev": []},
            "nbi": {"avrg": [], "stdev": []},
            "lines": {"avrg": [], "stdev": []},
        }
        return results

    def __call__(self, *args, **kwargs):
        """
        Analyse last 150 pulses

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """

    def read_data(self, pulse_start, pulse_end):
        """
        Read data in time-range of interest
        """

        results = self.init_results_dict()
        self.init_avantes()

        for pulse in np.arange(pulse_start, pulse_end + 1):
            print(pulse)
            reader = ST40Reader(
                pulse,
                np.min(self.tstart - self.dt * 2),
                np.max(self.tend + self.dt * 2),
            )
            magnetics = self.get_efit(reader)
            if magnetics["time"][0] == None:
                magnetics = self.get_pfit(reader)
                if magnetics["time"][0] == None:
                    continue
            nirh1 = self.get_nirh1(reader)
            smmh1 = self.get_smmh1(reader)
            gas = self.get_gas(reader)
            brems_pi, brems_mp = self.get_brems(reader)
            xrcs = self.get_xrcs(reader)
            nbi = self.get_nbi(reader)
            # mc = self.get_mc(reader)

            results["pulses"].append(pulse)

            tmp = self.init_results_dict()
            for tind in range(np.size(self.tstart)):
                avrg, std = calc_mean_std(
                    magnetics["time"],
                    magnetics["ipla"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=2.0e6,
                )
                tmp["ipla"]["avrg"].append(avrg)
                tmp["ipla"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    magnetics["time"],
                    magnetics["wp"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=1.0e6,
                )
                tmp["wp"]["avrg"].append(avrg)
                tmp["wp"]["stdev"].append(std)

                puff_total = np.nan
                puff_prefill = np.nan
                if gas["time"][0] is not None:
                    it = np.where(gas["time"] < self.t[tind])[0]
                    puff_total = np.sum(gas["puff"][it]) * (
                        gas["time"][1] - gas["time"][0]
                    )

                    it = np.where(gas["time"] <= 0)[0]
                    puff_prefill = np.sum(gas["puff"][it]) * (
                        gas["time"][1] - gas["time"][0]
                    )
                tmp["puff_total"].append(puff_total)
                tmp["puff_prefill"].append(puff_prefill)

                avrg, std = calc_mean_std(
                    nirh1["time"],
                    nirh1["ne_los_int"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=1.0e21,
                )
                tmp["nirh1"]["avrg"].append(avrg)
                tmp["nirh1"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    smmh1["time"],
                    smmh1["ne_los_int"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=1.0e21,
                )
                tmp["smmh1"]["avrg"].append(avrg)
                tmp["smmh1"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    brems_pi["time"],
                    brems_pi["brems"],
                    self.tstart[tind],
                    self.tend[tind],
                )
                tmp["brems_pi"]["avrg"].append(avrg)
                tmp["brems_pi"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    brems_pi["time"],
                    brems_pi["brems"],
                    self.tstart[tind],
                    self.tend[tind],
                )
                tmp["brems_mp"]["avrg"].append(avrg)
                tmp["brems_mp"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    xrcs["time"], xrcs["te"], self.tstart[tind], self.tend[tind]
                )
                tmp["xrcs_te"]["avrg"].append(avrg)
                tmp["xrcs_te"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    xrcs["time"], xrcs["ti"], self.tstart[tind], self.tend[tind]
                )
                tmp["xrcs_ti"]["avrg"].append(avrg)
                tmp["xrcs_ti"]["stdev"].append(std)

                avrg = np.nan
                std = np.nan
                if nbi["time"][0] is not None:
                    avrg, std = calc_mean_std(
                        nbi["time"],
                        nbi["power"],
                        self.tstart[tind],
                        self.tend[tind],
                        toffset=-0.5,
                    )
                tmp["nbi"]["avrg"].append(avrg)
                tmp["nbi"]["stdev"].append(std)

                lines_avrg = []
                lines_stdev = []
                line_time, _ = reader._get_signal(
                    "spectrom", "avantes.line_mon", ":time", 0
                )
                for line in self.lines_labels:
                    line_data, _ = reader._get_signal(
                        "spectrom",
                        "avantes.line_mon",
                        f".line_evol.{line}:intensity",
                        0,
                    )
                    avrg, std = calc_mean_std(
                        line_time, line_data, self.tstart[tind], self.tend[tind]
                    )
                    lines_avrg.append(avrg)
                    lines_stdev.append(std)

                tmp["lines"]["avrg"].append(lines_avrg)
                tmp["lines"]["stdev"].append(lines_stdev)

            for k1, res in tmp.items():
                if k1 == "pulses":
                    continue

                if type(res) != dict:
                    results[k1].append(res)
                    continue

                for k2, res2 in res.items():
                    results[k1][k2].append(res2)

        return results

    def add_data(self, pulse_end):
        """
        Add data from newer pulses to results dictionary

        Parameters
        ----------
        pulse_end
            Last pulse to include in the analysis
        """
        pulse_start = np.array(self.results["pulses"]).max() + 1
        if pulse_end < pulse_start:
            print("Newer pulses only (for the time being...)")
            return
        new = self.read_data(pulse_start, pulse_end)

        for i, pulse in enumerate(new["pulses"]):
            for k1, res in new.items():
                if k1 == "pulses":
                    continue
                if type(res) != dict:
                    self.results[k1].append(res[i])
                    continue

                for k2, res2 in res.items():
                    self.results[k1][k2].append(res2[i])

    def get_efit(self, reader):
        time, _ = reader._get_signal("", "efit", ":time", 0)
        if np.array_equal(time, "FAILED"):
            print("no Ip from EFIT")
            return {"time": [None]}
        if np.min(time) > np.max(self.tend) or np.max(time) < np.max(self.tstart):
            print("no Ip from EFIT in time range")
            return {"time": [None]}

        ipla, _ = reader._get_signal("", "efit", ".constraints.ip:cvalue", 0)
        wp, _ = reader._get_signal("", "efit", ".virial:wp", 0)
        return {"time": time, "ipla": ipla, "wp": wp}

    def get_pfit(self, reader):
        print("Reading PFIT instead of EFIT")
        time, _ = reader._get_signal("", "pfit", ".post_best.results:time", -1)
        if np.array_equal(time, "FAILED"):
            print("no Ip from PFIT")
            return {"time": [None]}
        if np.min(time) > np.max(self.tend) or np.max(time) < np.max(self.tstart):
            print("no Ip from PFIT in time range")
            return {"time": [None]}

        ipla, _ = reader._get_signal("", "pfit", ".post_best.results.global:ip", -1)
        wp, _ = reader._get_signal("", "pfit", ".post_best.results.global:wmhd", -1)
        return {"time": time, "ipla": ipla, "wp": wp}

    def get_nirh1(self, reader):
        nirh1 = {"time": [None]}
        ne_los_int, _path = reader._get_signal("interferom", "nirh1", ".line_int:ne", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED"):
            nirh1 = {"time": time[0], "ne_los_int": ne_los_int}
        return nirh1

    def get_smmh1(self, reader):
        smmh1 = {"time": [None]}
        ne_los_int, _path = reader._get_signal("interferom", "smmh1", ".line_int:ne", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED"):
            smmh1 = {"time": time[0], "ne_los_int": ne_los_int}
        return smmh1

    def get_gas(self, reader):
        gas = {"time": [None]}
        puff, _path = reader._get_signal("", "gas", ".puff_valve:gas_total", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time[0], "FAILED"):
            gas = {"time": time[0], "puff": puff}
        return gas

    def get_mc(self, reader):
        mc = {"time": [None]}
        imc, _path = reader._get_signal("", "psu", "mc:i", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED"):
            mc = {"time": time[0], "imc": imc}
        return mc

    def get_brems(self, reader):
        brems_pi = {"time": [None]}
        brems_mp = {"time": [None]}

        brems, _path = reader._get_signal(
            "spectrom", "princeton.passive", ".dc:brem_mp", 0
        )
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED"):
            brems_pi = {"time": time[0], "brems": brems}

        brems, _path = reader._get_signal(
            "spectrom", "lines", ".brem_mp1:intensity", -1
        )
        time, _ = reader._get_signal_dims(_path, 1)
        if not np.array_equal(time, "FAILED"):
            brems_mp = {"time": time[0], "brems": brems}

        return brems_pi, brems_mp

    def get_xrcs(self, reader):
        ti, _path = reader._get_signal("sxr", "xrcs", ".te_kw:ti", 0)
        te, path = reader._get_signal("sxr", "xrcs", ".te_kw:te", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        return {"time": time[0], "te": te, "ti": ti}

    def get_nbi(self, reader):
        i_hnbi, _path = reader._get_signal("raw_nbi", "hnbi1", ".hv_ps:i_jema", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        v_hnbi = 1.0

        power = i_hnbi * v_hnbi
        return {"time": time[0], "power": power}

    def plot_trends(self, savefig=False, xlim=(), heat="all"):
        """

        Parameters
        ----------
        results
            Output from read_data method
        savefig
            Save figure to file
        xlim
            Set xlim to values different from extremes in database
        heating
            Restrict analysis to specific heating only
            "all", "ohmic", "nbi"

        Returns
        -------

        """

        if len(xlim) == 0:
            xlim = (
                np.array(self.results["pulses"]).min() - 2,
                np.array(self.results["pulses"]).max() + 2,
            )
        pulse_range = f"{xlim[0]}-{xlim[1]}"

        time_tit = []
        puff_int_tit = []
        puff_prefill_tit = []
        for it, t in enumerate(self.t):
            time_range = f"{self.tstart[it]:1.3f}-{self.tend[it]:1.3f}"
            time_tit.append(f"t=[{time_range}]")
            puff_int_tit.append(f"t<{self.t[it]:1.3f}")
            puff_prefill_tit.append("t<0")
        name = f"ST40_trends_{pulse_range}"

        add_tit = ""
        lab = time_tit
        if len(self.t) == 1:
            lab = [""]
            add_tit = f" {time_tit[0]}"
        xlab = "Pulse #"
        ylab, tit = ("$(MA)$", "Plasma current" + add_tit)
        self.plot_evol("ipla", 1.0e-6, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_ipla")

        ylab, tit = ("$(kJ)$", "Plasma energy" + add_tit)
        self.plot_evol("wp", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_wp")

        ylab, tit = ("$(10^{19} m^{-3})$", "NIRH1 LOS-int density" + add_tit)
        self.plot_evol("nirh1", 1.0e-19, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_nirh1")

        ylab, tit = ("$(10^{19} m^{-3})$", "SMMH1 LOS-int density" + add_tit)
        self.plot_evol("smmh1", 1.0e-19, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_smmh1")

        self.results["brems_pi_nirh1"] = deepcopy(self.results["brems_pi"])
        self.results["brems_pi_nirh1"]["avrg"] /= (
            np.array(self.results["nirh1"]["avrg"]) * 1.0e9
        ) ** 2
        self.results["brems_pi_nirh1"]["stdev"] = (
            np.array(self.results["brems_pi_nirh1"]["stdev"]) * 0.0
        )
        ylab, tit = ("$(a.u.)$", "PI Bremsstrahlung/NIRH1^2" + add_tit)
        self.plot_evol("brems_pi_nirh1", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_brems_pi_norm")

        self.results["brems_mp_nirh1"] = deepcopy(self.results["brems_mp"])
        self.results["brems_mp_nirh1"]["avrg"] /= (
            np.array(self.results["nirh1"]["avrg"]) * 1.0e9
        ) ** 2
        self.results["brems_pi_nirh1"]["stdev"] = (
            np.array(self.results["brems_pi_nirh1"]["stdev"]) * 0.0
        )
        ylab, tit = ("$(a.u.)$", "MP filter Bremsstrahlung/NIRH1^2" + add_tit)
        self.plot_evol("brems_mp_nirh1", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_brems_mp_norm")

        ylab, tit = ("$(a.u.)$", "PI Bremsstrahlung" + add_tit)
        self.plot_evol("brems_pi", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_brems_pi")

        ylab, tit = ("$(a.u.)$", "MP filter Bremsstrahlung" + add_tit)
        self.plot_evol("brems_mp", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_brems_mp")

        ylab, tit = ("$(keV)$", "XRCS Te" + add_tit)
        self.plot_evol("xrcs_te", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_xrcs_te")

        ylab, tit = ("$(keV)$", "XRCS Ti" + add_tit)
        self.plot_evol("xrcs_ti", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_xrcs_ti")

        ylab, tit = ("$(a.u.)$", "NBI Power" + add_tit)
        self.plot_evol("nbi", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_nbi_power")

        lab = puff_prefill_tit
        ylab, tit = ("$(V * s)$", "Total gas prefill" + add_tit)
        self.plot_evol("puff_prefill", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_puff_prefill")

        lab = puff_int_tit
        ylab, tit = ("$(V * s)$", "Total gas" + add_tit)
        self.plot_evol("puff_total", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        add_vlines(GDC, color="r")
        if savefig:
            save_figure(fig_name=name + "_puff_total")

        lines = self.lines_labels
        elements = np.unique([l.split("_")[0] for l in lines])
        ylab = "$(a.u.)$"
        for elem in elements:
            print(elem)
            elem_name = elem.upper()
            if len(elem) > 1:
                elem_name = elem_name[0] + elem_name[1].lower()
            fig = True
            for il, l in enumerate(lines):
                lsplit = l.split("_")
                if elem != lsplit[0]:
                    continue

                tit = f"{elem_name} lines" + add_tit
                lab = f"{elem_name}{lsplit[1].upper()} {lsplit[2]} nm "
                if len(self.t) > 1:
                    tit = lab
                    lab = time_tit

                self.plot_evol("lines", 1.0, lab, xlab, ylab, tit, iline=il, fig=fig)
                fig = False
            plt.ylim(0,)
            add_vlines(BORONISATION)
            add_vlines(GDC, color="r")
            if savefig:
                save_figure(fig_name=name + f"_{elem_name}_lines")

    def select_data(self, heat):
        temporary = deepcopy(results)
        npulses = np.size(params["pulses"])
        nan_array = np.array([np.nan] * npulses)

        hind = {"all": [], "nbi": [], "ohmic": []}
        for it, t in enumerate(self.t):
            hind["all"] = np.array([True] * len(params["pulses"]))
            nbi_ind = results["nbi"]["avrg"][:, it] > 0
            hind["ohmic"] = nbi_ind == False
            hind["nbi"] = nbi_ind == True

            for k1 in temporary.keys():
                if type(temporary[k1]) == dict:
                    for k2 in temporary[k1].keys():
                        if len(temporary[k1][k2].shape) == 2:
                            tmp = temporary[k1][k2][:, it]
                            temporary[k1][k2][:, it] = np.where(
                                hind[heat], tmp, nan_array
                            )
                        else:
                            for il in range(len(temporary[k1][k2][0, it, :])):
                                tmp = temporary[k1][k2][:, it, il]
                                temporary[k1][k2][:, it, il] = np.where(
                                    hind[heat], tmp, nan_array
                                )
                else:
                    tmp = temporary[k1][:, it]
                    temporary[k1][:, it] = np.where(hind[heat], tmp, [np.nan] * npulses)

        return temporary

    def plot_evol(
        self, key, const, label, xlabel, ylabel, title, fig=True, iline=None, xlim=None
    ):
        if fig:
            plt.figure()
        for it, t in enumerate(self.t):
            if iline is not None:
                plt.errorbar(
                    np.array(self.results["pulses"]),
                    np.array(self.results[key]["avrg"])[:, it, iline] * const,
                    yerr=np.array(self.results[key]["stdev"])[:, it, iline] * const,
                    fmt="o",
                    label=label[it],
                )
            else:
                if type(self.results[key]) == dict:
                    plt.errorbar(
                        np.array(self.results["pulses"]),
                        np.array(self.results[key]["avrg"])[:, it] * const,
                        yerr=np.array(self.results[key]["stdev"])[:, it] * const,
                        fmt="o",
                        label=label[it],
                    )
                else:
                    plt.scatter(
                        np.array(self.results["pulses"]),
                        np.array(self.results[key])[:, it] * const,
                        marker="o",
                        label=label[it],
                    )
        if xlim is None:
            xlim = (
                np.array(self.results["pulses"]).min(),
                np.array(self.results["pulses"]).max(),
            )
        plt.xlim(xlim[0], xlim[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()


def add_vlines(xvalues, color="k"):
    ylim = plt.ylim()
    for b in xvalues:
        plt.vlines(b, ylim[0], ylim[1], linestyles="dashed", colors=color, alpha=0.5)


def save_figure(fig_name="", orientation="landscape", ext=".jpg"):
    plt.savefig(
        f"/home/{getpass.getuser()}/python/figures/data_trends/" + fig_name + ext,
        orientation=orientation,
        dpi=600,
        pil_kwargs={"quality": 95},
    )


def calc_mean_std(time, data, tstart, tend, lower=0.0, upper=None, toffset=None):
    avrg = np.nan
    std = np.nan
    offset = 0
    if (
        not np.array_equal(data, "FAILED")
        and (data.size == time.size)
        and data.size > 1
    ):
        it = (time >= tstart) * (time <= tend)
        if lower is not None:
            it *= data > lower
        if upper is not None:
            it *= data < upper

        it = np.where(it)[0]

        if toffset is not None:
            it_offset = np.where(time <= toffset)[0]
            if len(it_offset) > 1:
                offset = np.mean(data[it_offset])

        avrg = np.mean(data[it] + offset)
        if len(it) >= 2:
            std = np.std(data[it] + offset)

    return avrg, std
