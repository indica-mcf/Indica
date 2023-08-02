# import getpass
#
# import matplotlib.pyplot as plt
# import numpy as np
# import xarray as xr
#
# from indica.numpy_typing import ArrayLike
# from indica.operators.gpr_fit import plot_gpr_fit
# from indica.operators.gpr_fit import run_gpr_fit
# from indica.readers.read_st40 import ReadST40
# from indica.utilities import save_figure
# from indica.utilities import set_plot_colors
# from indica.utilities import set_plot_rcparams
#
# FIG_PATH = f"/home/{getpass.getuser()}/figures/Indica/profiles/"
# CMAP, COLORS = set_plot_colors()
# set_plot_rcparams("profiles")
#
#
# def plot_ts_cxrs_profiles(
#     pulse: int = 10605,
#     instruments: list = ["ts", "cxff_pi", "cxff_tws_c"],
#     tplot: ArrayLike = None,
#     save_fig=False,
#     max_norm_err: float = 0.05,
#     plot_rho: bool = True,
#     plot_spectra: bool = True,
# ):
#     tstart = 0.02
#     tend = 0.1
#
#     st40 = ReadST40(pulse, tstart, tend)
#     st40(instruments=instruments)
#
#     _data = st40.raw_data["ts"]
#     ts_te = _data["te"].swap_dims({"channel": "R"})
#     ts_rho, _ = _data["te"].transform.convert_to_rho_theta(t=_data["te"].t)
#     ts_rho = ts_rho.assign_coords(R=("channel", _data["te"].R)).swap_dims(
#         {"channel": "R"}
#     )
#     x_bounds = _data["te"].transform._machine_dims[0]
#     ts_fit, ts_fit_err = run_gpr_fit(
#         ts_te,
#         x_bounds=x_bounds,
#         y_bounds=(1, 1),
#         err_bounds=(0, 0),
#         xdim="R",
#     )
#
#     if "cxff_pi" in st40.raw_data.keys():
#         chan_slice = slice(3, 5)
#         _data = st40.raw_data["cxff_pi"]
#         pi_rho, _ = _data["ti"].transform.convert_to_rho_theta(t=_data["ti"].t)
#         pi_rho = pi_rho.sel(channel=chan_slice)
#         pi_ti = _data["ti"].sel(channel=chan_slice)
#         pi_ti_err = _data["ti"].error.sel(channel=chan_slice)
#         pi_spectra = _data["spectra"].sel(channel=chan_slice)
#         pi_fit = _data["fit"].sel(channel=chan_slice)
#
#         swap_dims = {"channel": "R"}
#         ti_cond = (pi_ti > 0) * (pi_ti_err / pi_ti <= max_norm_err)
#         spectra_cond = pi_ti.expand_dims({"wavelength": pi_spectra.wavelength}) > 0
#         pi_ti = xr.where(ti_cond, pi_ti, np.nan).swap_dims(swap_dims)
#         pi_ti_err = xr.where(ti_cond, pi_ti_err, np.nan).swap_dims(swap_dims)
#         pi_spectra = xr.where(spectra_cond, pi_spectra, np.nan).swap_dims(swap_dims)
#         pi_fit = xr.where(spectra_cond, pi_fit, np.nan).swap_dims(swap_dims)
#         if plot_rho:
#             pi_rho = pi_rho.assign_coords(R=("channel", pi_ti.R)).swap_dims(swap_dims)
#
#     if "cxff_tws_c" in st40.raw_data.keys():
#         chan_slice = slice(0, 2)
#         _data = st40.raw_data["cxff_tws_c"]
#         tws_rho, _ = _data["ti"].transform.convert_to_rho_theta(t=_data["ti"].t)
#         tws_rho = tws_rho.sel(channel=chan_slice)
#         tws_ti = _data["ti"].sel(channel=chan_slice)
#         tws_ti_err = _data["ti"].error.sel(channel=chan_slice)
#         tws_spectra = _data["spectra"].sel(channel=chan_slice)
#         tws_fit = _data["fit"].sel(channel=chan_slice)
#
#         swap_dims = {"channel": "R"}
#         ti_cond = (tws_ti > 0) * (tws_ti_err / tws_ti <= max_norm_err)
#         spectra_cond = tws_ti.expand_dims({"wavelength": tws_spectra.wavelength}) > 0
#         tws_ti = xr.where(ti_cond, tws_ti, np.nan).swap_dims(swap_dims)
#         tws_ti_err = xr.where(ti_cond, tws_ti_err, np.nan).swap_dims(swap_dims)
#         tws_spectra = xr.where(spectra_cond, tws_spectra, np.nan).swap_dims(swap_dims)
#         tws_fit = xr.where(spectra_cond, tws_fit, np.nan).swap_dims(swap_dims)
#         tws_rho = tws_rho.assign_coords(R=("channel", tws_ti.R)).swap_dims(swap_dims)
#
#     plt.ioff()
#     fig_name = f"{pulse}_TS_CXRS_profiles"
#     if tplot is None:
#         tplot = ts_te.t.values
#
#     t_pi = -10
#     t_tws = -10
#     tplot = np.array(tplot, ndmin=1)
#     for t in tplot:
#         t_ts = ts_te.t.sel(t=t, method="nearest").values
#         plot_gpr_fit(
#             ts_te,
#             ts_fit,
#             ts_fit_err,
#             t_ts,
#             ylabel="[eV]",
#             xlabel="R [m]",
#             title=str(st40.pulse),
#             label=f"El. (TS) {t_ts:.3f} s",
#             fig_name=f"{fig_name}_vs_R",
#             color=COLORS["electron"],
#         )
#         if "cxff_pi" in st40.raw_data.keys():
#             t_pi = pi_ti.t.sel(t=t, method="nearest").values
#         if "cxff_tws_c" in st40.raw_data.keys():
#             t_tws = tws_ti.t.sel(t=t, method="nearest").values
#         if np.abs(t_ts - t_pi) < 0.02:
#             plt.errorbar(
#                 pi_ti.R,
#                 pi_ti.sel(t=t_pi).values,
#                 pi_ti_err.sel(t=t_pi).values,
#                 color=COLORS["ion"],
#             )
#             plt.plot(
#                 pi_ti.R,
#                 pi_ti.sel(t=t_pi).values,
#                 marker="s",
#                 color=COLORS["ion"],
#                 label=f"Ion (PI) {t_pi:.3f} s",
#             )
#
#         if np.abs(t_ts - t_tws) < 0.02:
#             plt.errorbar(
#                 tws_ti.R,
#                 tws_ti.sel(t=t_tws).values,
#                 tws_ti_err.sel(t=t_tws).values,
#                 color=COLORS["ion"],
#             )
#             plt.plot(
#                 tws_ti.R,
#                 tws_ti.sel(t=t_tws).values,
#                 marker="^",
#                 color=COLORS["ion"],
#                 label=f"Ion (TWS) {t_tws:.3f} s",
#             )
#         plt.legend(loc="upper left")
#         if save_fig:
#             save_figure(
#                 FIG_PATH,
#                 f"{fig_name}_vs_R_{t:.3f}_s",
#                 save_fig=save_fig,
#             )
#
#         if plot_rho:
#             x_data = ts_rho
#             x_fit = x_data.interp(R=ts_fit.R)
#             plot_gpr_fit(
#                 ts_te,
#                 ts_fit,
#                 ts_fit_err,
#                 t_ts,
#                 x_data=x_data,
#                 x_fit=x_fit,
#                 ylabel="[eV]",
#                 xlabel="rho-poloidal",
#                 title=str(st40.pulse),
#                 label=f"El. (TS) {t_ts:.3f} s",
#                 color=COLORS["electron"],
#             )
#
#         if plot_rho and np.abs(t_ts - t_pi) < 0.02:
#             plt.errorbar(
#                 pi_rho.sel(t=t_pi).values,
#                 pi_ti.sel(t=t_pi).values,
#                 pi_ti_err.sel(t=t_pi).values,
#                 color=COLORS["ion"],
#             )
#             plt.plot(
#                 pi_rho.sel(t=t_pi).values,
#                 pi_ti.sel(t=t_pi).values,
#                 marker="s",
#                 color=COLORS["ion"],
#                 label=f"Ion (PI) {t_pi:.3f} s",
#             )
#
#         if plot_rho and np.abs(t_ts - t_tws) < 0.02:
#             plt.errorbar(
#                 tws_rho.sel(t=t_tws).values,
#                 tws_ti.sel(t=t_tws).values,
#                 tws_ti_err.sel(t=t_tws).values,
#                 color=COLORS["ion"],
#             )
#             plt.plot(
#                 tws_rho.sel(t=t_tws).values,
#                 tws_ti.sel(t=t_tws).values,
#                 marker="^",
#                 color=COLORS["ion"],
#                 label=f"Ion (TWS) {t_tws:.3f} s",
#             )
#         if plot_rho:
#             plt.legend(loc="upper left")
#             if save_fig:
#                 save_figure(
#                     FIG_PATH,
#                     f"{fig_name}_vs_rho_{t:.3f}_s",
#                     save_fig=save_fig,
#                 )
#
#         R0 = 0.48
#         if plot_spectra and np.abs(t_ts - t_pi) < 0.02:
#             plt.figure()
#             R = pi_spectra.R.sel(R=R0, method="nearest").values
#             pi_spectra.sel(t=t_pi, R=R).plot(label="Spectra", color="black")
#             pi_fit.sel(t=t_pi, R=R).plot(label="Fit", color="orange")
#             plt.legend(loc="upper left")
#             plt.title(f"PI spectra {t_tws:.3f} s, R={R:.2f}m")
#
#         if plot_spectra and np.abs(t_ts - t_tws) < 0.02:
#             plt.figure()
#             R = tws_spectra.R.sel(R=R0, method="nearest").values
#             tws_spectra.sel(t=t_tws, R=R).plot(label="Spectra", color="black")
#             tws_fit.sel(t=t_tws, R=R).plot(label="Fit", color="orange")
#             plt.legend(loc="upper left")
#             plt.title(f"TWS spectra {t_tws:.3f} s, R={R:.2f}m")
#
#         if save_fig:
#             plt.close("all")
#         else:
#             plt.show()
#
#
# if __name__ == "__main__":
#     plot_ts_cxrs_profiles()
