import matplotlib as mpl

# mpl.rcParams['keymap.back'].remove('left')
# mpl.rcParams['keymap.forward'].remove('right')

import numpy as np
import matplotlib.pylab as plt
from IPython import embed
from scipy.linalg import eigh, solve_banded, inv
from scipy.interpolate import interp1d, RectBivariateSpline
import pickle
from scipy.signal import argrelextrema
from scipy.ndimage import convolve
from matplotlib.widgets import Slider, MultiCursor, Button
import time as tt


def update_fill_between(fill, x, y_low, y_up, min, max):
    (paths,) = fill.get_paths()
    nx = len(x)

    y_low = np.maximum(y_low, min)
    y_low[y_low == max] = min
    y_up = np.minimum(y_up, max)

    vertices = paths.vertices.T
    vertices[:, 1 : nx + 1] = x, y_up
    vertices[:, nx + 1] = x[-1], y_up[-1]
    vertices[:, nx + 2 : -1] = x[::-1], y_low[::-1]
    vertices[:, 0] = x[0], y_up[0]
    vertices[:, -1] = x[0], y_up[0]


def update_errorbar(err_plot, x, y, yerr):

    plotline, caplines, barlinecols = err_plot

    # Replot the data first
    plotline.set_data(x, y)

    # Find the ending points of the errorbars
    error_positions = (x, y - yerr), (x, y + yerr)

    # Update the caplines
    if len(caplines) > 0:
        for j, pos in enumerate(error_positions):
            caplines[j].set_data(pos)

    # Update the error bars
    barlinecols[0].set_segments(
        list(zip(list(zip(x, y - yerr)), list(zip(x, y + yerr))))
    )


def triband_transpose(A):
    # transpose the tridiagonal band matrix A
    AT = np.copy(A)
    AT[0, 1:], AT[2, :-1] = A[2, :-1], A[0, 1:]
    return AT


def triband_diag_multi(A, D):
    # multiply tridiagonal band matrix a diagonal matrix

    AD = np.copy(A)
    AD[0, 1:] *= D[:-1]
    AD[1] *= D
    AD[2, :-1] *= D[1:]
    return AD


def fast_svd(H):
    # fast method to calculate U,S,V = svd(H.T) of thin rectangular matrix
    LL = np.dot(H.T, H)
    S2, U = eigh(LL, overwrite_a=True, check_finite=True, lower=True)
    S = (
        np.maximum(S2, S2[-1] * 1e-20) ** 0.5
    )  # singular values S can be negative due to numerical uncertainty
    return U, S


class SXR_tomography:
    def __init__(
        self,
        input_dict: dict,
        reg_level_guess: float = 0.5,
        reg_level_min: float = 0.4,
        reg_level_max=0.4,
    ):
        """
        1D tomography code assuming the emissivity is flux-surface symmetric

        Parameters
        ----------
        input_dict
            input_dict = dict(
                brightness:array(ntime, nchannel) = measured experimental brightness,
                dl:float = radial precision of the lines of sight,
                t:array(ntime) = time array of the experimental data,
                R:array(nchannel, npoints) = R along the LOS (npoints = number of points along LOS),
                z:array(nchannel, npoints) = z along the LOS (npoints = number of points along LOS),
                rho_equil=dict(
                    R:array(nR_eq) = R grid of the equilibrium,
                    z:array(nz_eq) = z grid of the equilibrium,
                    t:array(nt_eq) = time of the equilibrium,
                    rho:array(nt_eq, nR_eq, nz_eq) = sqrt(normalised poloidal flux) from equilibrium,
                ),
                debug=debug,
                has_data:array(nchannel) = boolean array to discard non-active channels,
            )
        reg_level_guess:float = regularisation parameter (larger value --> stiffer profiles)
        """

        self.reg_level_guess = reg_level_guess
        self.reg_level_min = reg_level_min
        self.reg_level_max = reg_level_max
        self.eq = input_dict["rho_equil"]

        ##### Parameters
        los_shape = np.shape(input_dict["R"])
        self.nlos = los_shape[0]
        self.los_len = los_shape[1]

        # number of equalibrium timeslices
        self.nt = self.eq["t"].size

        # number of radial grid binds
        self.nr = 100

        # number of virtual LOS accounting for finite divergence of LOS
        self.nvirt = 11  # odd number

        # radial grid
        self.rho_grid_edges = np.linspace(0, 1, self.nr + 1)
        self.rho_grid_centers = (self.rho_grid_edges[1:] + self.rho_grid_edges[:-1]) / 2

        # step along LOS
        self.dL = input_dict["dl"]

        self.R = input_dict["R"]
        self.z = input_dict["z"]
        # self.rho = input_dict['rho']  # TODO: channel list

        # IMPACT PARAMETERS
        # self.impact_parameters = input_dict["impact_parameters"]

        self.debug = input_dict["debug"]

        # load data
        self.data = input_dict["brightness"]  # W/m^2 ??

        # guess!! of uncertainty
        self.err = self.data * 0.05 + np.nanmax(self.data) * 0.01  # assume 5% error

        self.tvec = input_dict["t"]

        self.valid = input_dict["has_data"]

        if "emissivity" in input_dict.keys():
            self.expected_emissivity = input_dict["emissivity"]

    def geom_matrix(self):
        # create geometry matrix witn contribution of all grid intervals to the measured signals

        # create vitual LOS
        self.x = np.arange(self.nlos)
        self.x_inter = (
            self.x[:, None] + np.linspace(-1, 1, self.nvirt) * (1 - 1 / self.nvirt) / 2
        )

        R = interp1d(self.x, self.R.T, fill_value="extrapolate")(self.x_inter)
        z = interp1d(self.x, self.z.T, fill_value="extrapolate")(self.x_inter)

        # locate index x_tg for R_tg
        i_tg = np.argmin(R, axis=0)[None]

        # weighted sum all dL values which fits in between bin edges
        # dLmat2 is calculate by summing all dL contributiions to each grid bin, used for benchmarking
        nt = self.eq["t"].size
        # calculate exactly length of chord in each grid bin
        dLmat = np.zeros((nt, self.nlos, self.nvirt, self.nr))
        rho_tg = np.zeros((nt, self.nlos, self.nvirt))

        #################  prepare L matrix #####################
        for it in range(nt):
            LOS_rho = RectBivariateSpline(
                self.eq["R"], self.eq["z"], self.eq["rho"][it].T
            ).ev(R, z)

            rho_tg[it] = np.take_along_axis(LOS_rho, i_tg, axis=0)[0]

            # iterate over chords and calculate contribution to each grid bin
            for ilos in range(self.nlos):
                # weight is given by dL value and is is splitted equally between all nvirt virtual LOSs

                for iv in range(self.nvirt):
                    # get first and last point just outside of lcfs
                    ind = np.where(np.abs(LOS_rho[:, ilos, iv]) < 1)[0]
                    if len(ind) == 0:  # no cross section with grid
                        continue
                    ilcfs_in = ind[0] - 1
                    ilcfs_out = ind[-1] + 2
                    rho_cut = LOS_rho[ilcfs_in:ilcfs_out, ilos, iv]

                    # find local minima and maxima, rho is monotonous in between
                    imin = argrelextrema(rho_cut, np.less_equal)[0]
                    imax = argrelextrema(rho_cut, np.greater_equal)[0]
                    i_extrema = np.unique(np.r_[imin, imax])

                    L = np.arange(ilcfs_out - ilcfs_in) * self.dL
                    L_turn = 0

                    for i in range(len(i_extrema) - 1):
                        # split LOS in regions with monotonously changing rho to make the inversion
                        monotone_ind = slice(i_extrema[i], i_extrema[i + 1] + 1)

                        if rho_cut[monotone_ind][0] == rho_cut[monotone_ind][-1]:
                            # special case, whole monotone_ind is constant
                            continue

                        # input for interpolation needs to be monotonous
                        Ledge = interp1d(
                            rho_cut[monotone_ind],
                            L[monotone_ind],
                            bounds_error=False,
                            fill_value=0,
                        )(self.rho_grid_edges)

                        if not np.any(Ledge > 0):
                            # special case, whole monotone_ind is within single grid cell
                            continue

                        # index of turning points
                        i_tg_in, i_tg_out = np.where(Ledge > 0)[0][[0, -1]]

                        dLmat[it, ilos, iv, i_tg_in:i_tg_out] += np.abs(
                            Ledge[i_tg_in:i_tg_out] - Ledge[i_tg_in + 1 : i_tg_out + 1]
                        )

                        # add the turning point singularity
                        if (
                            rho_cut[monotone_ind][-1] - rho_cut[monotone_ind][0] > 0
                        ):  # if L increases with rho
                            if i_tg_in > 0:
                                dLmat[it, ilos, iv, i_tg_in - 1] += (
                                    Ledge[i_tg_in] - L_turn
                                )
                            if (
                                i_tg_out < self.nr
                            ):  # this should happen just in the last step
                                L_turn = Ledge[i_tg_out]
                                assert i < len(i_extrema) - 1
                        else:  # if L decreases with rho
                            if (
                                i_tg_out < self.nr
                            ):  # this should happen just in the first step
                                dLmat[it, ilos, iv, i_tg_out] += (
                                    Ledge[i_tg_out] - L_turn
                                )
                            L_turn = Ledge[i_tg_in]

        # average over all virtual LOSs, assume equal weight
        self.dLmat = dLmat.mean(2)
        self.rho_tg = rho_tg.mean(2)

        dLmat_interleaved = dLmat.reshape(nt, self.nvirt * self.nlos, self.nr)
        self.dLmat_interleaved = convolve(
            dLmat_interleaved, np.ones((1, self.nvirt, 1)) / self.nvirt, mode="nearest"
        )

        rho_tg_interleaved = rho_tg.reshape(nt, -1)
        self.rho_tg_interleaved = convolve(
            rho_tg_interleaved, np.ones((1, self.nvirt)) / self.nvirt, mode="nearest"
        )

    def regul_matrix(self, bias_axis=True, bias_edge=False):
        # regularization band matrix, 2. order derivative, bias left or right side to zero
        bias = 0.1
        D = np.ones((3, self.nr))
        D[1, :] *= -2
        D[0, 1] = 0
        D[2, -2] = 0

        D[1, [0, -1]] = 1e-5  # just to make D invertible

        if bias_axis:
            # bias to zero gradient on axis
            D[1, 0] = 1
            D[0, 1] = -1

        if bias_edge:
            # bias right edge value towards zero
            D[1, -1] = bias

        return D

    def PRESS(self, g, prod, S, U):
        # predictive sum of squares
        w = 1.0 / (1.0 + np.exp(g) / S ** 2)
        ndets = len(prod)
        return (
            np.sum(
                (np.dot(U, (1 - w) * prod) / np.einsum("ij,ij,j->i", U, U, 1 - w)) ** 2
            )
            / ndets
        )

    def GCV(self, g, prod, S, U):
        # generalized crossvalidation
        w = 1.0 / (1.0 + np.exp(g) / S ** 2)
        ndets = len(prod)
        return (np.sum((((w - 1) * prod)) ** 2) + 1) / ndets / (1 - np.mean(w)) ** 2

    def FindMin(self, F, x0, dx0, prod, S, U, tol=0.01):
        # stupid but robust minimum searching algorithm.

        fg = F(x0, prod, S, U)
        while abs(dx0) > tol:
            fg2 = F(x0 + dx0, prod, S, U)

            if fg2 < fg:
                fg = fg2
                x0 += dx0
                continue
            else:
                dx0 /= -2.0

        return x0, np.log(fg2)

    def calc_tomo(self, reg_level=0.8, nfisher=3, eps=1e-2, optim_regul=False):
        # calculate tomography using optimised minimum Fisher regularisation
        # Odstrcil, T., et al. "Optimized tomography methods for plasma
        # emissivity reconstruction at the ASDEX  Upgrade tokamak.
        # Review of Scientific Instruments 87.12 (2016): 123505.

        # prepare regularisation operator - contains all prior information
        # biased_edges - assume zero value at the boundaries of the grid.
        D = self.regul_matrix()

        nt = len(self.tvec)

        self.chi2 = np.zeros(nt)
        self.gamma = np.zeros(nt)

        self.backprojection_int = np.zeros((nt, self.nlos * self.nvirt))
        self.backprojection = np.ones((nt, self.nlos)) * np.nan
        self.emiss = np.zeros((nt, self.nr))
        self.emiss_err = np.zeros((nt, self.nr))

        teq_ind = np.int_(
            interp1d(
                self.eq["t"],
                np.arange(len(self.eq["t"])),
                kind="nearest",
                fill_value="extrapolate",
            )(self.tvec)
        )

        # split reconstruction in blocks nearest to equilibrium data
        for it, teq in enumerate(self.eq["t"]):
            t_ind = teq_ind == it
            if not np.any(t_ind):
                continue

            valid = np.all(
                (np.isfinite(self.data[t_ind, :])) & (self.data[t_ind, :] > 0), axis=0
            )
            # valid = self.valid
            # weight the contribution matrix and data by the uncertainty
            # print(it,teq,t_ind)#,(self.err[t_ind,:]/self.data[t_ind,:]).shape,valid)
            err = np.atleast_2d(self.err[t_ind, valid]).mean(0)
            T = self.dLmat[it, valid] / err[:, None]
            mean_d = (
                np.atleast_2d(self.data[t_ind, valid]).mean(0) / err
            )  # NOTE take a ratio of means of mean of rations?
            d = np.atleast_2d(self.data[t_ind, valid] / self.err[t_ind, valid])

            # flat initial estimate of the weight matrix W
            W = np.ones(self.nr)

            q = np.linspace(0, 1, len(mean_d))

            # iterative calculation of minimum Fisher regularisation
            for ifisher in range(nfisher):

                #####    solve Tikhonov regularization (optimised for speed)

                # multiply tridiagonal regularisation operator by a diagonal weight matrix W

                WD = triband_diag_multi(D, W ** 0.5)

                # transpose the band matrix WD
                DTW = triband_transpose(WD)

                # calculate (D.TW)^-1*T.T
                H = solve_banded(
                    (1, 1), DTW, T.T, overwrite_ab=False, check_finite=False
                )

                invalid = H.sum(0) == 0
                if np.any(invalid):
                    print("Warning - some LOS are not linearly independent")

                # fast method to calculate U,S,V = svd(H.T) of rectangular matrix
                U, S = fast_svd(H)

                # projection of the data on the U base
                mean_p = np.dot(mean_d, U)

                # calculate regularisation parameter
                g0 = np.interp(reg_level, q, 2 * np.log(S))

                # filtering factors attenuating high frequency eigenvectors
                w = 1.0 / (1.0 + np.exp(g0) / S ** 2)

                # mean solution
                y = np.dot(H, np.dot(U / S ** 2, w * mean_p))
                # final inversion of  solution, reconstruction
                y = solve_banded(
                    (1, 1),
                    WD,
                    y,
                    overwrite_ab=False,
                    overwrite_b=True,
                    check_finite=False,
                )

                W = 1 / np.maximum(y, eps)

            # evaluate the basis in the reconstruction space
            V = np.dot(H, U / S)
            V = solve_banded(
                (1, 1), WD, V, overwrite_ab=False, overwrite_b=True, check_finite=False
            )

            # estimate optimal regularisation level for initial guess
            if optim_regul:
                g0, log_fg2 = self.FindMin(self.PRESS, g0, 1, mean_p, S, U)
                self.gamma[t_ind] = np.interp(g0, np.log(S) * 2, q)
                self.gamma[t_ind] = max(self.reg_level_min, self.gamma[t_ind]) ** 2
                w = 1.0 / (1.0 + np.exp(g0) / S ** 2)
            else:
                self.gamma[t_ind] = reg_level

            # find solution for each timeslice
            p = np.dot(d, U)
            y = np.dot((w / S) * p, V.T)
            fit = np.dot(p * w, U.T)

            self.backprojection[t_ind, valid] = fit * err
            # self.backprojection[t_ind,valid] = np.dot(y, self.dLmat[it].T)[:,valid]

            self.backprojection_int[t_ind] = np.dot(y, self.dLmat_interleaved[it].T)

            self.chi2[t_ind] = np.sum((d - fit) ** 2, 1) / np.size(
                fit, 1
            )  # TODO jen ty funkcni ch

            self.emiss[t_ind] = y
            self.emiss_err[t_ind] = np.sqrt(
                np.dot(V ** 2, (w / S) ** 2) * np.maximum(1, self.chi2[t_ind, None])
            )

    def show_reconstruction(self):
        # reg_level_min = .4
        # reg_level_max = .4
        f, ax = plt.subplots(1, 3, sharex="col", figsize=(12, 5))
        ax_reg = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor="y")
        slide_reg = Slider(
            ax_reg,
            "Regularisation:",
            0,
            1,
            valinit=self.reg_level_guess,
            valstep=0.001,
            valfmt="%1.3f",
        )

        ax_time = plt.axes([0.2, 0.07, 0.65, 0.03], facecolor="y")
        slide_time = Slider(
            ax_time,
            "Time [s]:",
            self.tvec[0],
            self.tvec[-1],
            valinit=self.tvec[0],
            valstep=0.001,
            valfmt="%1.3f",
        )

        r = self.rho_grid_centers
        f.subplots_adjust(bottom=0.2)

        tomo_var = ax[0].fill_between(
            r, r * 0, r * 0, alpha=0.5, facecolor="b", edgecolor="None"
        )
        (expected_emissivity, ) = ax[0].plot(
            [], [], alpha=0.5, color="r", linestyle="dashed", lw=2
        )
        (tomo_mean,) = ax[0].plot([], [], lw=2)

        errorbar = ax[1].errorbar(0, np.nan, 0, capsize=4, c="g", marker="o", ls="none")
        (retro_inter,) = ax[1].plot([], [], "b-")
        (retro,) = ax[1].plot([], [], "bx")

        ax[0].axhline(0, c="k")
        ax[1].axhline(0, c="k")

        ax[0].set_xlim(0, 1)
        ax[1].set_xlim(self.x_inter.min(), self.x_inter.max())

        ax[1].set_ylim(0, np.nanmax(self.data) * 1.2 / 1e3)
        ymax = np.nanmax(self.emiss)
        if hasattr(self, "expected_emissivity"):
            ymax = np.nanmax([ymax, self.expected_emissivity.max()])
        ax[0].set_ylim(0, ymax * 1.2 / 1e3)
        print(ymax)

        ax[0].set_xlabel(r"$\rho$")
        ax[1].set_xlabel(r"index")
        ax[1].set_ylabel("Brightness [kW/m$^2$]")
        ax[0].set_ylabel("Emissivity [kW/m$^3$]")
        global cont
        cvals = np.linspace(0, 1, 20)

        cont = ax[2].contour(
            self.eq["R"], self.eq["z"], self.eq["rho"][0], cvals, colors="k"
        )

        ax[2].set_ylim(self.eq["R"][[0, -1]])
        ax[2].set_xlim(self.eq["z"][[0, -1]])
        ax[2].axis("equal")
        ax[2].plot(self.R.T, self.z.T, "b", zorder=99)
        R, z = np.meshgrid(self.eq["R"], self.eq["z"])
        ax[2].plot(R, z, "k,", zorder=99)
        ax[2].axis(
            [0.15, 0.7, -0.4, 0.4,]
        )
        ax[2].set_ylabel("z [m]")
        ax[2].set_xlabel("R [m]")

        f.subplots_adjust(wspace=0.3)

        title = f.suptitle("")

        def update(reg=None, time=None):
            global cont
            if reg is not None:
                self.calc_tomo(reg_level=reg ** 0.5, nfisher=3, eps=1e-5)

            if time is None:
                time = slide_time.val

            it = np.argmin(np.abs(self.tvec - time))

            update_fill_between(
                tomo_var,
                r,
                self.emiss[it] / 1e3 - self.emiss_err[it] / 1e3,
                self.emiss[it] / 1e3 + self.emiss_err[it] / 1e3,
                -np.inf,
                np.inf,
            )
            if hasattr(self, "expected_emissivity"):
                expected_emissivity.set_data(
                    self.expected_emissivity.rho_poloidal,
                    self.expected_emissivity.sel(t=time, method="nearest") / 1.0e3,
                )
            tomo_mean.set_data(r, self.emiss[it] / 1e3)

            retro_inter.set_data(self.x_inter, self.backprojection_int[it] / 1e3)
            retro.set_data(self.x, self.backprojection[it] / 1e3)

            update_errorbar(errorbar, self.x, self.data[it] / 1e3, self.err[it] / 1e3)

            for c in cont.collections:
                c.remove()  # removes only the contours, leaves the rest intact

            it_eq = np.argmin(np.abs(self.eq["t"] - time))

            cont = ax[2].contour(
                self.eq["R"], self.eq["z"], self.eq["rho"][it_eq], cvals, colors="k"
            )

            title.set_text(
                "  $\chi^2/nDoF$ = %.1f  $\gamma$ = %.2f"
                % (self.chi2[it], self.gamma[it])
            )
            f.canvas.draw_idle()

        def update_time(time):
            update(time=time)

        def update_reg(reg):
            update(reg=reg)

        def on_key(event):
            dt = (self.tvec[-1] - self.tvec[0]) / (len(self.tvec) - 1)
            tnew = slide_time.val

            if hasattr(event, "step"):
                # scroll_event
                tnew += event.step * dt

            elif "left" == event.key:
                # key_press_event
                tnew -= dt

            elif "right" == event.key:
                tnew += dt

            tnew = min(max(tnew, self.tvec[0]), self.tvec[-1])
            slide_time.set_val(tnew)
            update(time=tnew)

        slide_reg.on_changed(update_reg)
        slide_time.on_changed(update_time)

        update(slide_reg.valinit, slide_time.valinit)

        self.cid = f.canvas.mpl_connect("key_press_event", on_key)
        self.cid_scroll = f.canvas.mpl_connect("scroll_event", on_key)

        axbutton = plt.axes([0.85, 0.1, 0.1, 0.05])
        self.save_button = Button(axbutton, "Save")
        self.save_button.on_clicked(self.save)
        plt.show()

    def save(self, args):
        np.savez_compressed(
            "Reconstruction",
            backprojection=self.backprojection,
            brightness=self.data,
            brightness_err=self.err,
            time=self.tvec,
            grid=self.rho_grid_centers,
            gamma=self.gamma,
            chi2=self.chi2,
            emiss=self.emiss.astype("single"),
            emiss_err=self.emiss_err.astype("single"),
        )

        print("Saved to Reconstruction.npz file")

    # FUNCTION TO GET THE TOMOGRAPHY AND RETURN THE DATA
    def __call__(self):
        if self.debug:
            debug_data = {"invert_class": {}}
            start_time = tt.time()
            st = start_time
        # CREATING GEOMETRY MATRIX
        self.geom_matrix()
        # DEBUG TIME
        if self.debug:
            step = "Creating geometry matrix"
            step_time = np.round(tt.time() - st, 2)
            debug_data["invert_class"][step] = step_time
            print(step + ". It took " + str(step_time) + " seconds")
            st = tt.time()
        # CALCULATING TOMOGRAPHY
        self.calc_tomo(optim_regul=False)
        # DEBUG TIME
        if self.debug:
            step = "Calculating tomography"
            step_time = np.round(tt.time() - st, 2)
            debug_data["invert_class"][step] = step_time
            print(step + ". It took " + str(step_time) + " seconds")
            st = tt.time()
        # FUNCTION TO GET 2D EMISSIVITY DATA
        def get_emissivity_2D(data, eq_data):
            # R AND z VALUES
            R = eq_data["R"]
            z = eq_data["z"]
            rho = eq_data["rho"]
            # EMISSIVITY 1D PROFILE VALUE
            emiss_1D = data["profile"]["sym_emissivity"]
            rho_1D = data["profile"]["rho_poloidal"]
            # DATA DECLARATION
            emiss = np.nan * np.ones((len(data["t"]), len(R), len(z)))
            # SWEEP OF TIME
            for it, t in enumerate(data["t"]):
                # SELECTED EQUILIBRIUM INDEX
                sel_ind = np.where(
                    np.abs(eq_data["t"] - t) == np.nanmin(np.abs(eq_data["t"] - t))
                )[0][0]
                # SELECTING RHO
                sel_rho = rho[sel_ind, :, :].T
                # 2D VALUE OF EMISSIVITY
                emiss[it, :, :] = interp1d(
                    rho_1D[it, :], emiss_1D[it, :], bounds_error=False, fill_value=0
                )(sel_rho)
            # RETURN DATA
            return_data = dict(R=R, z=z, data=emiss.T,)
            # RETURNING THE DATA
            return return_data

        # GATHERING THE DATA
        return_data = dict(
            t=self.tvec,
            channels_considered=self.valid,  # np.ones(np.size(self.data,1),dtype=bool),
            # BACK INTEGRAL DATA
            back_integral=dict(
                # p_impact=self.impact_parameters,
                data_experiment=self.data,
                data_theory=self.backprojection,
                channel_no=np.arange(1, np.size(self.data, 1) + 1),
            ),
            # PROJECTION DATA
            projection=dict(R=self.R, z=self.z,),
            # PROFILES
            profile=dict(
                sym_emissivity=self.emiss,
                asym_parameter=np.zeros(self.emiss.shape),
                rho_poloidal=np.repeat(
                    np.array([self.rho_grid_centers]), len(self.tvec), axis=0
                ),
            ),
        )
        # EMISSIVITY 2D
        return_data["emissivity_2D"] = get_emissivity_2D(return_data, self.eq)
        # ESTIMATING THE CHI2
        data_exp = return_data["back_integral"]["data_experiment"][
            :, return_data["channels_considered"]
        ]
        data_the = return_data["back_integral"]["data_theory"][
            :, return_data["channels_considered"]
        ]
        return_data["back_integral"]["chi2"] = np.sqrt(
            np.nansum(((data_exp - data_the) ** 2) / (data_exp ** 2), axis=1)
        )
        # APPENDING DEBUG DATA
        if self.debug:
            return_data["debug_data"] = debug_data
        # RETURNING THE DATA
        return return_data


def main():

    import time

    tomo = SXR_tomography()
    t = time.time()
    tomo.geom_matrix()
    print("Geometry matrix calculated in %.2fs" % (time.time() - t))

    t = time.time()
    tomo.calc_tomo(optim_regul=False)
    print("Tomographic reconstruction calculates in %.2fs" % (time.time() - t))

    tomo.show_reconstruction()


if __name__ == "__main__":
    main()
