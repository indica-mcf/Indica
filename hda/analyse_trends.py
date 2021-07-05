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
BORONISATION = [8441]

plt.ion()


class correlations:
    def __init__(self, pulse_start, pulse_end, t=[0.03, 0.08], dt=0.01):
        self.pulse_start = pulse_start
        self.pulse_end = pulse_end
        if type(t) is not list:
            t = [t]
        self.t = np.array(t)
        self.dt = dt

        self.tstart = self.t - self.dt
        self.tend = self.t + self.dt
        self.pulses = np.arange(self.pulse_start, self.pulse_end + 1)
        self.read_data()
        self.plot_trends()

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

    def read_data(self):
        """
        Read data in time-range of interest
        """

        self.results = self.init_results_dict()
        self.init_avantes()

        for pulse in self.pulses:
            print(pulse)
            reader = ST40Reader(
                pulse,
                np.min(self.tstart - self.dt * 2),
                np.max(self.tend + self.dt * 2),
            )
            efit = self.get_efit(reader)
            if efit["time"][0] == None:
                continue
            nirh1 = self.get_nirh1(reader)
            gas = self.get_gas(reader)
            brems_pi, brems_mp = self.get_brems(reader)
            xrcs = self.get_xrcs(reader)
            nbi = self.get_nbi(reader)

            self.results["pulses"].append(pulse)

            tmp = self.init_results_dict()
            for tind in range(np.size(self.tstart)):
                avrg, std = calc_mean_std(
                    efit["time"],
                    efit["ipla"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=2.0e6,
                )
                tmp["ipla"]["avrg"].append(avrg)
                tmp["ipla"]["stdev"].append(std)

                avrg, std = calc_mean_std(
                    efit["time"],
                    efit["wp"],
                    self.tstart[tind],
                    self.tend[tind],
                    upper=1.0e6,
                )
                tmp["wp"]["avrg"].append(avrg)
                tmp["wp"]["stdev"].append(std)

                it = np.where(gas["time"] < self.t[tind])[0]
                tmp["puff_total"].append(
                    np.sum(gas["puff"][it]) * (gas["time"][1] - gas["time"][0])
                )

                it = np.where(gas["time"] <= 0)[0]
                tmp["puff_prefill"].append(
                    np.sum(gas["puff"][it]) * (gas["time"][1] - gas["time"][0])
                )

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
                    self.results[k1].append(res)
                    continue

                for k2, res2 in res.items():
                    self.results[k1][k2].append(res2)

        for k1 in self.results.keys():
            if type(self.results[k1]) != dict:
                self.results[k1] = np.array(self.results[k1])
                continue

            for k2 in self.results[k1].keys():
                self.results[k1][k2] = np.array(self.results[k1][k2])

    def get_efit(self, reader):
        time, _ = reader._get_signal("", "efit", ":time", 0)
        if np.array_equal(time, "FAILED"):
            print("no Ip")
            return {"time": [None]}
        if np.min(time) > np.max(self.tend) or np.max(time) < np.max(self.tstart):
            print("no Ip in time range")
            return {"time": [None]}

        ipla, _ = reader._get_signal("", "efit", ".constraints.ip:cvalue", 0)
        wp, _ = reader._get_signal("", "efit", ".virial:wp", 0)
        return {"time": time, "ipla": ipla, "wp": wp}

    def get_nirh1(self, reader):
        ne_los_int, _path = reader._get_signal("interferom", "nirh1", ".line_int:ne", 0)
        time, _ = reader._get_signal_dims(_path, 1)
        return {"time": time[0], "ne_los_int": ne_los_int}

    def get_gas(self, reader):
        puff, _path = reader._get_signal("", "gas", ".puff_valve:gas_total", -1)
        time, _ = reader._get_signal_dims(_path, 1)
        return {"time": time[0], "puff": puff}

    def get_brems(self, reader):
        brems, _path = reader._get_signal(
            "spectrom", "princeton.passive", ".dc:brem_mp", 0
        )
        time, _ = reader._get_signal_dims(_path, 1)
        brems_pi = {"time": time[0], "brems": brems}

        brems, _path = reader._get_signal(
            "spectrom", "lines", ".brem_mp1:intensity", -1
        )
        time, _ = reader._get_signal_dims(_path, 1)
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
            xlim = (self.pulse_start - 2, self.pulse_end + 2)
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
        if savefig:
            save_figure(fig_name=name + "_ipla")

        ylab, tit = ("$(kJ)$", "Plasma energy" + add_tit)
        self.plot_evol("wp", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_wp")

        ylab, tit = ("$(10^{19} m^{-3})$", "NIRH1 LOS-int density" + add_tit)
        self.plot_evol("nirh1", 1.0e-19, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_nirh1")

        self.results["brems_pi_nirh1"] = deepcopy(self.results["brems_pi"])
        self.results["brems_pi_nirh1"]["avrg"] /= (
            self.results["nirh1"]["avrg"] * 1.0e9
        ) ** 2
        self.results["brems_pi_nirh1"]["stdev"] *= 0.0
        ylab, tit = ("$(a.u.)$", "PI Bremsstrahlung/NIRH1^2" + add_tit)
        self.plot_evol("brems_pi_nirh1", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_brems_pi_norm")

        self.results["brems_mp_nirh1"] = deepcopy(self.results["brems_mp"])
        self.results["brems_mp_nirh1"]["avrg"] /= (
            self.results["nirh1"]["avrg"] * 1.0e9
        ) ** 2
        self.results["brems_mp_nirh1"]["stdev"] *= 0.0
        ylab, tit = ("$(a.u.)$", "MP filter Bremsstrahlung/NIRH1^2" + add_tit)
        self.plot_evol("brems_mp_nirh1", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_brems_mp_norm")

        ylab, tit = ("$(a.u.)$", "PI Bremsstrahlung" + add_tit)
        self.plot_evol("brems_pi", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_brems_pi")

        ylab, tit = ("$(a.u.)$", "MP filter Bremsstrahlung" + add_tit)
        self.plot_evol("brems_mp", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_brems_mp")

        ylab, tit = ("$(keV)$", "XRCS Te" + add_tit)
        self.plot_evol("xrcs_te", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_xrcs_te")

        ylab, tit = ("$(keV)$", "XRCS Ti" + add_tit)
        self.plot_evol("xrcs_ti", 1.0e-3, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_xrcs_ti")

        ylab, tit = ("$(a.u.)$", "NBI Power" + add_tit)
        self.plot_evol("nbi", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_nbi_power")

        lab = puff_prefill_tit
        ylab, tit = ("$(V * s)$", "Total gas prefill" + add_tit)
        self.plot_evol("puff_prefill", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
        if savefig:
            save_figure(fig_name=name + "_puff_prefill")

        lab = puff_int_tit
        ylab, tit = ("$(V * s)$", "Total gas" + add_tit)
        self.plot_evol("puff_total", 1.0, lab, xlab, ylab, tit)
        plt.ylim(0,)
        add_vlines(BORONISATION)
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
                    self.results["pulses"],
                    self.results[key]["avrg"][:, it, iline] * const,
                    yerr=self.results[key]["stdev"][:, it, iline] * const,
                    fmt="o",
                    label=label[it],
                )
            else:
                if type(self.results[key]) == dict:
                    plt.errorbar(
                        self.results["pulses"],
                        self.results[key]["avrg"][:, it] * const,
                        yerr=self.results[key]["stdev"][:, it] * const,
                        fmt="o",
                        label=label[it],
                    )
                else:
                    plt.scatter(
                        self.results["pulses"],
                        self.results[key][:, it] * const,
                        marker="o",
                        label=label[it],
                    )
        if xlim is None:
            xlim = (self.pulse_start, self.pulse_end)
        plt.xlim(xlim[0], xlim[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()


def add_vlines(xvalues):
    ylim = plt.ylim()
    for b in xvalues:
        plt.vlines(b, ylim[0], ylim[1], linestyles="dashed", colors="black")


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
