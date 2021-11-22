from copy import deepcopy

import pickle

from matplotlib import cm
import matplotlib.pylab as plt

import numpy as np

from scipy.optimize import least_squares
from scipy import constants

from hda.profiles import Profiles
import hda.physics as ph

from indica.readers import ADASReader
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.converters.time import bin_in_time

import xarray as xr
from xarray import DataArray, Dataset

plt.ion()

# TODO: add elongation and triangularity in all equations

ADF11 = {
    "h": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
    },
    "he": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
    },
    "c": {
        "scd": "96",
        "acd": "96",
        "ccd": "96",
        "plt": "96",
        "prb": "96",
        "prc": "96",
    },
    "ar": {
        "scd": "89",
        "acd": "89",
        "ccd": "89",
        "plt": "00",
        "prb": "00",
        "prc": "89",
    },
}


class Plasma:
    def __init__(
        self,
        tstart=0.01,
        tend=0.14,
        dt=0.01,
        ntheta=5,
        machine_dimensions=((0.15, 0.9), (-0.8, 0.8)),
        elements=("h", "c", "ar"),
    ):
        """

        Parameters
        ----------
        pulse

        """

        self.ADASReader = ADASReader()
        self.elements = elements
        self.tstart = tstart
        self.tend = tend
        self.dt = dt
        self.t = np.arange(tstart, tend, dt)
        self.theta = np.linspace(0, 2 * np.pi, ntheta + 1)[:-1]
        self.radial_coordinate = np.linspace(0, 1.0, 41)
        self.radial_coordinate_type = "rho_poloidal"
        self.machine_dimensions = machine_dimensions

        self.forward_models = {}

        self.initialize_variables()

    def build_data(self, data, pulse=None, equil="efit"):
        """
        Reorganise raw data on new time axis and generate geometry information

        Parameters
        ----------
        data
            Raw data dictionary
        equil
            Equilibrium code to use for equilibrium object

        Returns
        -------

        """
        print_like("Building data class")

        self.initialize_variables()

        self.pulse = pulse
        self.optimisation["equil"] = f"{equil}:{data[equil]['rmag'].revision}"

        t_ip = data["efit"]["ipla"].t
        if self.tstart < t_ip.min():
            print_like("Start time changed to stay inside Ip limit")
            self.tstart = t_ip.min().values
        if self.tend > t_ip.max():
            print_like("End time changed to stay inside Ip limit")
            self.tend = t_ip.max().values

        self.equil = equil

        if equil in data.keys():
            print_like("Initialise equilibrium object")
            self.equilibrium = Equilibrium(data[equil])
            self.flux_coords = FluxSurfaceCoordinates("poloidal")
            self.flux_coords.set_equilibrium(self.equilibrium)

        print_like("Assign equilibrium, bin data in time")
        binned_data = {}
        for kinstr in data.keys():
            instrument_data = {}

            print(kinstr)
            if type(data[kinstr]) != dict:
                value = deepcopy(data[kinstr])
                if np.size(value) > 1:
                    value = bin_in_time(
                        self.tstart, self.tend, self.freq, value
                    ).interp(t=self.time, method="linear")
                binned_data[kinstr] = value
                continue

            transform = None
            geom_attrs = None
            for kquant in data[kinstr].keys():
                # print("   " + kquant)
                value = bin_in_time(
                    self.tstart, self.tend, self.freq, data[kinstr][kquant]
                ).interp(t=self.time, method="linear")

                if "error" in value.attrs.keys():
                    error = bin_in_time(
                        self.tstart,
                        self.tend,
                        self.freq,
                        data[kinstr][kquant].attrs["error"],
                    ).interp(t=self.time, method="linear")
                    value.attrs["error"] = error

                if "transform" in value.attrs and transform is None:
                    transform = value.attrs["transform"]
                    transform.set_equilibrium(self.equilibrium, force=True)
                    if "LinesOfSightTransform" in str(transform):
                        geom_attrs = remap_diagnostic(value, self.flux_coords)

                if transform is not None:
                    value.attrs["transform"] = transform

                if geom_attrs is not None:
                    for kattrs in geom_attrs:
                        value.attrs[kattrs] = geom_attrs[kattrs]
                instrument_data[kquant] = value

            binned_data[kinstr] = instrument_data

        self.ipla.values = binned_data["efit"]["ipla"]
        self.cr0.values = (
            binned_data["efit"]["rmjo"] - binned_data["efit"]["rmji"]
        ).sel(rho_poloidal=1) / 2.0

        self.R_mag = binned_data["efit"]["rmag"]
        self.z_mag = binned_data["efit"]["zmag"]

        self.R_bt_0 = binned_data["R_bt_0"]
        self.bt_0 = binned_data["bt_0"]

        return binned_data

    def apply_limits(
        self,
        data,
        diagnostic: str,
        quantity=None,
        val_lim=(np.nan, np.nan),
        err_lim=(np.nan, np.nan),
    ):
        """
        Set to Nan all data whose value or relative error aren't within specified limits
        """

        if quantity is None:
            quantity = list(data[diagnostic])
        else:
            quantity = list(quantity)

        for q in quantity:
            error = None
            value = data[diagnostic][q]
            if "error" in value.attrs.keys():
                error = data[diagnostic][q].attrs["error"]

            if np.isfinite(val_lim[0]):
                print(val_lim[0])
                value = xr.where(value >= val_lim[0], value, np.nan)
            if np.isfinite(val_lim[1]):
                print(val_lim[1])
                value = xr.where(value <= val_lim[1], value, np.nan)

            if error is not None:
                if np.isfinite(err_lim[0]):
                    print(err_lim[0])
                    value = xr.where((error/value) >= err_lim[0], value, np.nan)
                if np.isfinite(err_lim[1]):
                    print(err_lim[1])
                    value = xr.where((error/value) <= err_lim[1], value, np.nan)

            data[diagnostic][q].values = value.values

        return data

    def calculate_geometry(self):
        if hasattr(self, "equilibrium"):
            print_like("Calculate geometric quantities")
            for ith, th in enumerate(self.theta):
                min_r_tmp, _ = self.equilibrium.minor_radius(
                    self.equilibrium.rmji.rho_poloidal, th
                )
                if ith == 0:
                    min_r = min_r_tmp
                else:
                    min_r += min_r_tmp

            min_r /= len(self.theta)
            min_r = min_r.interp(rho_poloidal=self.rho.values, method="cubic")
            min_r = bin_in_time(self.tstart, self.tend, self.freq, min_r,).interp(
                t=self.time, method="linear"
            )
            self.min_r = min_r

            volume, area, _ = self.equilibrium.enclosed_volume(self.rho)
            volume = bin_in_time(self.tstart, self.tend, self.freq, volume,).interp(
                t=self.time, method="linear"
            )
            area = bin_in_time(self.tstart, self.tend, self.freq, area,).interp(
                t=self.time, method="linear"
            )
            self.area.values = area.values
            self.volume.values = volume.values

            self.r_a.values = self.min_r.sel(rho_poloidal=1.0)
            self.r_b.values = self.r_a.values
            self.r_c.values = self.r_a.values
            self.r_d.values = self.r_a.values
            self.kappa.values = (self.r_b / self.r_a).values
            self.delta.values = ((self.r_c + self.r_d) / (2 * self.r_a)).values

            self.maj_r_lfs = bin_in_time(
                self.tstart,
                self.tend,
                self.freq,
                self.equilibrium.rmjo.interp(rho_poloidal=self.rho),
            ).interp(t=self.time, method="linear")
            self.maj_r_hfs = bin_in_time(
                self.tstart,
                self.tend,
                self.freq,
                self.equilibrium.rmji.interp(rho_poloidal=self.rho),
            ).interp(t=self.time, method="linear")

    def build_atomic_data(self, adf11: dict = None):
        print_like("Initialize fractional abundance objects")
        fract_abu, power_loss = {}, {}
        for elem in self.elements:
            if adf11 is None:
                adf11 = ADF11

            scd = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])
            fract_abu[elem] = FractionalAbundance(scd, acd, CCD=ccd)

            plt = self.ADASReader.get_adf11("plt", elem, adf11[elem]["plt"])
            prb = self.ADASReader.get_adf11("prb", elem, adf11[elem]["prb"])
            prc = self.ADASReader.get_adf11("prc", elem, adf11[elem]["prc"])
            power_loss[elem] = PowerLoss(plt, prb, PRC=prc)

        self.adf11 = adf11
        self.fract_abu = fract_abu
        self.power_loss = power_loss

    def set_neutral_density(self, y0=1.0e10, y1=1.0e15, decay=12):
        self.Nh_prof.y0 = y0
        self.Nh_prof.y1 = y1
        self.Nh_prof.yend = y1
        self.Nh_prof.wped = decay
        self.Nh_prof.build_profile()
        for t in self.t:
            self.neutral_dens.loc[dict(t=t)] = self.Nh_prof.yspl.values

    def interferometer(self, data, bckc={}, diagnostic=None, quantity=None):
        """
        Calculate expected diagnostic measurement given plasma profile
        """

        if diagnostic is None:
            diagnostic = ["nirh1", "nirh1_bin", "smmh1"]
        if quantity is None:
            quantity = ["ne"]
        diagnostic = list(diagnostic)
        quantity = list(quantity)

        for diag in diagnostic:
            if diag not in data.keys():
                continue

            for quant in quantity:
                if quant not in data[diag].keys():
                    continue

                bckc = initialize_bckc(diag, quant, data, bckc=bckc)

                bckc[diag][quant].values = self.calc_ne_los_int(
                    data[diag][quant]
                ).values

        return bckc

    def bremsstrahlung(
        self,
        data,
        bckc={},
        diagnostic="lines",
        quantity="brems",
        wavelength=532.0,
        cal=2.5e-5,
    ):
        """
        Estimate back-calculated Bremsstrahlung measurement from plasma quantities

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            name of diagnostic usef for bremsstrahlung measurement
        quantity
            Measurement to be used for the bremsstrahlung
        wavelength
            Wavelength of measurement
        cal
            Calibration factor for measurement
            Default value calculated to match Zeff before Ar puff from LINES.BREMS_MP for pulse 9408

        Returns
        -------
        bckc
            dictionary with back calculated value(s)

        """
        brems = ph.zeff_bremsstrahlung(
            self.el_temp, self.el_dens, wavelength, zeff=self.zeff.sum("element")
        )
        if diagnostic in data.keys():
            if quantity in data[diagnostic].keys():
                bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

                bckc[diagnostic][quantity].values = self.calc_los_int(
                    data[diagnostic][quantity], brems * cal
                ).values

        bckc[diagnostic][quantity].attrs["calibration"] = cal
        return bckc

    def match_xrcs_temperatures(
        self,
        data,
        bckc={},
        diagnostic: str = "xrcs",
        quantity_te="te_kw",
        quantity_ti="ti_w",
        y1=50,
        wped=4,
        wcenter=0.4,
        peaking=1.5,
        niter=3,
        leastsq=False,
        use_ratios=False,
        time = None,
    ):
        """
        Rescale temperature profiles to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_te
            Measurement to be used for the electron temperature optimisation
        quantity_ti
            Measurement to be used for the ion temperature optimisation
        y1,  wped, wcenter, peaking
            Parameters of the profiler object
        niter
            Number of iterations

        Returns
        -------

        """

        def residuals_te_ratio(te0):
            """
            Optimisation for line ratios
            """
            Te_prof.y0 = te0
            Te_prof.build_profile()
            el_temp = Te_prof.yspl
            forward_model.radiation_characteristics(el_temp, el_dens)
            forward_model.los_integral(rho_los, dl)

            int_n3 = forward_model.intensity
            if quantity_te == "te_kw":
                ratio_bckc = (forward_model.intensity["k"] / forward_model.intensity["w"])
            elif quantity_te == "te_n3w":
                ratio_bckc = forward_model.intensity["n3"] / (data[diagnostic]["int_tot"])

            te_bckc = _bckc[quantity_te]
            resid = te_data - te_bckc
            return resid

        def residuals_te(te0):
            Te_prof.y0 = te0
            Te_prof.build_profile()
            el_temp = Te_prof.yspl
            ion_temp = Ti_prof.yspl
            forward_model.radiation_characteristics(el_temp, el_dens)
            _bckc = forward_model.moment_analysis(ion_temp, rho_los=rho_los, dl=dl)
            te_bckc = _bckc[quantity_te]
            resid = te_data - te_bckc
            return resid

        def residuals_ti(ti0):
            Ti_prof.y0 = ti0
            Ti_prof.build_profile(y0_ref=Te_prof.y0)
            ion_temp = Ti_prof.yspl
            el_temp = Te_prof.yspl
            forward_model.radiation_characteristics(el_temp, el_dens)
            _bckc = forward_model.moment_analysis(ion_temp, rho_los=rho_los, dl=dl)
            ti_bckc = _bckc[quantity_ti]
            resid = ti_data - ti_bckc
            return resid

        def iterative(
            te_data, ti_data, el_dens, niter,
        ):

            Te_prof.y0 = deepcopy(te_data)
            Ti_prof.y0 = deepcopy(ti_data)

            const_te, const_ti = 1, 1
            for j in range(niter):
                Te_prof.y0 *= const_te
                Te_prof.build_profile()
                el_temp = Te_prof.yspl

                Ti_prof.y0 *= const_ti
                Ti_prof.build_profile(y0_ref=Te_prof.y0)
                ion_temp = Ti_prof.yspl

                forward_model.radiation_characteristics(el_temp, el_dens)
                _bckc = forward_model.moment_analysis(ion_temp)
                te_bckc = _bckc[quantity_te]
                ti_bckc = _bckc[quantity_ti]

                const_te = (
                    te_data
                    / el_temp.sel(rho_poloidal=te_bckc.rho_poloidal, method="nearest")
                ).values[0]
                const_ti = (
                    ti_data
                    / ion_temp.sel(rho_poloidal=ti_bckc.rho_poloidal, method="nearest")
                ).values[0]

                if (np.abs(1 - const_te) < 1.0e-4) and (np.abs(1 - const_ti) < 1.0e-4):
                    break

            return te_bckc, ti_bckc

        if use_ratios:
            if not leastsq:
                print_like("Ratio optimisation possible only with least-sq optimisation..")
                raise ValueError

        if diagnostic not in data.keys():
            print_like(f"No {diagnostic.upper()} data available")
            return

        print_like(
            f"Re-calculating temperature profiles to match {diagnostic.upper()} values"
        )

        if time is None:
            time = self.t

        if diagnostic not in bckc:
            bckc[diagnostic] = {}

        bckc = initialize_bckc(diagnostic, quantity_te, data, bckc=bckc)
        bckc = initialize_bckc(diagnostic, quantity_ti, data, bckc=bckc)

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        Te_prof = self.Te_prof
        Ti_prof = self.Ti_prof

        Te_prof.y1 = y1
        Te_prof.wped = wped
        Te_prof.wcenter = wcenter
        Te_prof.peaking = peaking

        Ti_prof.y1 = y1
        Ti_prof.wped = wped
        Ti_prof.wcenter = wcenter
        Ti_prof.peaking = peaking

        pos = xr.full_like(data[diagnostic][quantity_te].t, np.nan)
        err_in = deepcopy(pos)
        err_out = deepcopy(pos)
        te_pos = Dataset({"value": pos, "err_in": err_in, "err_out": err_out})
        ti_pos = deepcopy(te_pos)
        bounds = (100.0, 30.0e3)

        emiss = []
        fz = []
        forward_model.radiation_characteristics(self.Te_prof.yspl, self.Ne_prof.yspl)
        for t in self.t:
            emiss.append(forward_model.emiss["w"] * 0.0)
            fz.append(forward_model.fz["ar"] * 0.0)
        emiss = xr.concat(emiss, "t").assign_coords(t=self.t)
        fz = xr.concat(fz, "t").assign_coords(t=self.t)

        if quantity_te == "te_kw":
            ratio = (data[diagnostic]["int_k"]/data[diagnostic]["int_w"])
        elif quantity_te == "te_n3w":
            ratio = data[diagnostic]["int_n3"]/(data[diagnostic]["int_tot"])

        dl = data[diagnostic][quantity_ti].attrs["dl"]
        for t in time:
            print(t)
            te_data = data[diagnostic][quantity_te].sel(t=t)
            ti_data = data[diagnostic][quantity_ti].sel(t=t)
            rho_los = data[diagnostic][quantity_ti].attrs["rho"].sel(t=t)
            ratio_data = ratio.sel(t=t)

            if not ((te_data > 0) * (ti_data > 0)).values:
                self.el_temp.loc[dict(t=t)] = np.full_like(Te_prof.yspl.values, np.nan)
                for elem in self.elements:
                    self.ion_temp.loc[dict(t=t, element=elem)] = np.full_like(
                        Te_prof.yspl.values, np.nan
                    )

                continue

            el_dens = self.el_dens.sel(t=t)

            if leastsq:
                fit_te = least_squares(residuals_te, te_data, bounds=bounds)
                fit_ti = least_squares(residuals_ti, ti_data, bounds=bounds)

                el_temp = Te_prof.yspl
                ion_temp = Ti_prof.yspl
                forward_model.radiation_characteristics(el_temp, el_dens)
                _bckc = forward_model.moment_analysis(ion_temp, rho_los=rho_los, dl=dl)
                te_bckc = _bckc[quantity_te]
                ti_bckc = _bckc[quantity_ti]
            else:
                te_bckc, ti_bckc = iterative(
                    te_data.values, ti_data.values, el_dens, niter
                )

                el_temp = Te_prof.yspl
                ion_temp = Ti_prof.yspl

            te_pos.value.loc[dict(t=t)] = te_bckc.rho_poloidal.values[0]
            te_pos.err_in.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["in"]
            te_pos.err_out.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["out"]

            ti_pos.value.loc[dict(t=t)] = ti_bckc.rho_poloidal.values[0]
            ti_pos.err_in.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["in"]
            ti_pos.err_out.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["out"]

            emiss.loc[dict(t=t)] = forward_model.emiss["w"]
            fz.loc[dict(t=t)] = forward_model.fz["ar"]

            bckc[diagnostic][quantity_te].loc[dict(t=t)] = te_bckc.values[0]

            bckc[diagnostic][quantity_ti].loc[dict(t=t)] = ti_bckc.values[0]

            self.el_temp.loc[dict(t=t)] = el_temp.values
            for elem in self.elements:
                self.ion_temp.loc[dict(t=t, element=elem)] = ion_temp.values

        bckc[diagnostic][quantity_te].attrs["pos"] = te_pos
        bckc[diagnostic][quantity_te].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_te].attrs["fz"] = fz

        bckc[diagnostic][quantity_ti].attrs["pos"] = ti_pos
        bckc[diagnostic][quantity_ti].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_ti].attrs["fz"] = fz

        self.optimisation[
            "el_temp"
        ] = f"{diagnostic}.{quantity_te}:{data[diagnostic][quantity_te].revision}"
        self.optimisation[
            "ion_temp"
        ] = f"{diagnostic}.{quantity_ti}:{data[diagnostic][quantity_ti].revision}"

        return bckc

    def match_xrcs_intensity(
        self,
        data,
        bckc={},
        diagnostic: str = "xrcs",
        quantity: str = "int_w",
        elem="ar",
        cal=1.e13,
        niter=2,
        time = None,
    ):
        """
        Compute Ar density to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_int
            Measurement to be used for determining the impurity concentration from line intensity
        cal
            Calibration factor for measurement
            Default value calculated to match Zeff from LINES.BREMS_MP for pulse 9408
        elem
            Element responsible for measured spectra
        niter
            Number of iterations

        Returns
        -------

        """

        if diagnostic not in data.keys():
            print_like(f"No {diagnostic.upper()} data available")
            return

        print_like(
            f"Re-calculating Ar density profiles to match {diagnostic.upper()} values"
        )

        if time is None:
            time = self.t

        if diagnostic not in bckc:
            bckc[diagnostic] = {}

        bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        dl = data[diagnostic][quantity].attrs["dl"]
        for i, t in enumerate(time):
            print(t)

            int_data = data[diagnostic][quantity].sel(t=t)
            Te = self.el_temp.sel(t=t)
            Ne = self.el_dens.sel(t=t)
            tau = self.tau.sel(t=t)
            Nh = self.neutral_dens.sel(t=t)
            Nimp = {elem:self.ion_dens.sel(element=elem).sel(t=t)}
            if np.isnan(Te).any():
                continue
            rho_los = data[diagnostic][quantity].attrs["rho"].sel(t=t)

            const = 1.0
            for j in range(niter):
                Nimp[elem] *= const
                forward_model.radiation_characteristics(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau)
                int_bckc, _ = forward_model.los_integral(rho_los, dl)
                int_bckc = int_bckc["w"] * cal
                const = (int_data / int_bckc).values

                if np.abs(1 - const) < 1.0e-4:
                    break

            self.ion_dens.loc[dict(element=elem, t=t)] = Nimp[elem].values
            bckc[diagnostic][quantity].loc[dict(t=t)] = int_bckc.values

        bckc[diagnostic][quantity].attrs["calibration"] = cal
        self.optimisation[
            "imp_dens"
        ] = f"{diagnostic}.{quantity}:{data[diagnostic][quantity].revision}"

        return bckc

    def match_interferometer(
        self,
        data,
        bckc={},
        diagnostic: str = "nirh1",
        quantity: str = "ne_bin",
        error=False,
        niter=3,
        time = None,
    ):
        """
        Rescale density profiles to match the interferometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by self.build_data()
        bckc
            back calculated data dictionary
        interf
            Name of interferometer to be used

        Returns
        -------

        """
        print_like(
            f"Re-calculating density profiles to match {diagnostic}.{quantity} values"
        )

        if time is None:
            time = self.t

        # TODO: make more elegant optimisation

        bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)

        Ne_prof = self.Ne_prof
        for t in time:
            const = 1.0
            for j in range(niter):
                ne0 = Ne_prof.yspl.sel(rho_poloidal=0) * const
                ne0 = xr.where((ne0 <= 0) or (not np.isfinite(ne0)), 5.0e19, ne0)
                Ne_prof.y0 = ne0.values
                Ne_prof.build_profile()
                self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
                ne_bckc = self.calc_ne_los_int(data[diagnostic][quantity], t=t)
                const = (data[diagnostic][quantity].sel(t=t) / ne_bckc).values

            bckc[diagnostic][quantity].loc[dict(t=t)] = ne_bckc

        self.optimisation[
            "el_dens"
        ] = f"{diagnostic}.{quantity}:{data[diagnostic][quantity].revision}"
        self.optimisation["stored_en"] = ""

        return bckc

    def recover_density(
        self, data, diagnostic: str = "efit", quantity: str = "wp", niter=3,
    ):
        """
        Match stored energy by adapting electron density

        Parameters
        ----------
        data
        diagnostic
        quantity
        niter

        Returns
        -------

        """
        print("\n Re-calculating density to match plasma energy \n")

        Ne_prof = self.Ne_prof
        const = DataArray([1.0] * len(self.t), coords=[("t", self.t)])
        ne0 = self.el_dens.sel(rho_poloidal=0)
        data_tmp = data[diagnostic][quantity]
        for j in range(niter):
            for t in self.t:
                if np.isfinite(const.sel(t=t)):
                    Ne_prof.y0 = (ne0 * const).sel(t=t).values
                    Ne_prof.build_profile()
                    self.el_dens.loc[dict(t=t)] = Ne_prof.yspl.values
            self.calc_imp_dens()
            self.calc_main_ion_dens()
            self.calc_zeff()
            self.calc_pressure()
            bckc_tmp = self.wp.sel()
            const = 1 + (data_tmp - bckc_tmp) / bckc_tmp

        self.optimisation[
            "stored_en"
        ] = f"{diagnostic}.{quantity}:{data[diagnostic][quantity].revision}"

    def propagate_parameters(self):
        """
        Propagate all parameters to maintain parameter consistency
        """
        self.match_xrcs()
        self.build_current_density()
        self.calc_magnetic_field()
        self.calc_meanz()
        self.calc_main_ion_dens(fast_dens=False)
        self.impose_flat_zeff()
        self.calc_zeff()
        self.calc_rad_power()
        self.calc_pressure()
        self.calc_beta_poloidal()
        self.calc_vloop()

    def calc_ne_los_int(self, data, passes=2, t=None):
        """
        Calculate line of sight integral for a specified number of passes through the plasma

        Returns
        -------
        los_int
            Integral along the line of sight

        """
        dl = data.attrs["dl"]
        rho = data.attrs["rho"]
        transform = data.attrs["transform"]

        x2_name = transform.x2_name

        el_dens = self.el_dens.interp(rho_poloidal=rho)
        if t is not None:
            el_dens = el_dens.sel(t=t, method="nearest")
            rho = rho.sel(t=t, method="nearest")
        el_dens = xr.where(rho <= 1, el_dens, 0,)
        el_dens_int = passes * el_dens.sum(x2_name) * dl

        return el_dens_int

    def calc_los_int(self, data, profile, t=None):
        """
        Calculate line of sight integral of quantity saved as attr in class
        
        Parameters
        ----------
        data
            raw data with LOS information of specified diagnostic
        quantity
            Quantity to be integrated along the LOS
        t
            Time

        Returns
        -------
        Integral along the line of sight

        """
        dl = data.attrs["dl"]
        rho = data.attrs["rho"]
        transform = data.attrs["transform"]

        x2_name = transform.x2_name

        value = profile.interp(rho_poloidal=rho)
        if t is not None:
            value = value.sel(t=t)
            rho = rho.sel(t=t)
        value = xr.where(rho <= 1, value, 0,)
        los_int = value.sum(x2_name) * dl

        return los_int

    def calc_main_ion_dens(self, fast_dens=True):
        """
        Calculate main ion density from quasi-neutrality given electron and impurity densities

        Parameters
        ----------
        fast_dens
            Include fast ion density in calculation
        """

        ion_dens_meanz = self.ion_dens * self.meanz
        main_ion_dens = deepcopy(self.el_dens)
        for elem in self.impurities:
            main_ion_dens -= ion_dens_meanz.sel(element=elem)

        if fast_dens is True:
            main_ion_dens -= self.fast_dens

        self.ion_dens.loc[dict(element=self.main_ion)] = main_ion_dens

    def calc_imp_dens(self):
        """
        Calculate impurity density from concentration
        """

        imp_dens = self.Nimp_prof.yspl / self.Nimp_prof.yspl.sel(rho_poloidal=0)
        for elem in self.impurities:
            imp_dens_0 = self.el_dens.sel(rho_poloidal=0) * self.ion_conc.sel(
                element=elem
            )
            for t in self.t:
                self.ion_dens.loc[dict(t=t, element=elem)] = imp_dens * imp_dens_0.sel(
                    t=t
                )

    def calc_fz_lz(self, use_tau=False):
        """
        Calculate fractional abundance and cooling factors
        """
        tau = None
        fz = {}
        lz = {}
        rho = self.el_dens.rho_poloidal.values
        t = self.el_dens.t.values
        for elem in self.elements:
            ion_charges = np.arange(len(self.fract_abu[elem].SCD.ion_charges) + 1)
            fz[elem] = DataArray(
                np.full((len(t), len(rho), len(ion_charges)), np.nan),
                coords={"t": t, "rho_poloidal": rho, "ion_charges": ion_charges},
                dims=["t", "rho_poloidal", "ion_charges",],
            )
            lz[elem] = deepcopy(fz[elem])
        for elem in self.elements:
            for t in self.t:
                Ne = self.el_dens.sel(t=t)
                Nh = self.neutral_dens.sel(t=t)
                Te = self.el_temp.sel(t=t)

                if any(np.logical_not((Te > 0) * (Ne > 0))):
                    continue

                if use_tau:
                    tau = self.tau.sel(t=t)
                fz_tmp = self.fract_abu[elem](Ne, Te, Nh=Nh, tau=tau)
                fz[elem].loc[dict(t=t)] = fz_tmp.transpose()
                lz[elem].loc[dict(t=t)] = self.power_loss[elem](
                    Ne, Te, fz_tmp, Nh=Nh
                ).transpose()

        self.fz = fz
        self.lz = lz

    def calc_meanz(self):
        """
        Calculate mean charge
        """
        if not hasattr(self, "fz") or not hasattr(self, "lz"):
            self.calc_fz_lz()

        for elem in self.elements:
            fz = self.fz[elem]
            self.meanz.loc[dict(element=elem)] = (fz * fz.ion_charges).sum(
                "ion_charges"
            )

    def calc_rad_power(self):
        """
        Calculate total and SXR filtered radiated power
        """
        for elem in self.elements:
            self.tot_rad.loc[dict(element=elem)] = (
                self.lz[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )

            self.tot_rad.loc[dict(element=elem)] = xr.where(
                self.tot_rad.sel(element=elem) >= 0,
                self.tot_rad.sel(element=elem),
                0.0,
            )

            for t in self.t:
                self.prad.loc[dict(element=elem, t=t)] = np.trapz(
                    self.tot_rad.sel(element=elem, t=t), self.volume.sel(t=t)
                )

    def calc_pressure(self, data=None, bckc=None):
        """
        Calculate pressure profiles (thermal and total), MHD and diamagnetic energies
        """

        if data is not None:
            if hasattr(self, "equil"):
                bckc = initialize_bckc(self.equil, "wp", data, bckc=bckc)
            if "dialoop" in data.keys():
                bckc = initialize_bckc("wdia", "dialoop", data, bckc=bckc)

        p_el = ph.calc_pressure(self.el_dens.values, self.el_temp.values)

        p_ion = ph.calc_pressure(
            self.ion_dens.sel(element=self.main_ion).values,
            self.ion_temp.sel(element=self.main_ion).values,
        )
        for elem in self.impurities:
            p_ion += ph.calc_pressure(
                self.ion_dens.sel(element=elem).values,
                self.ion_temp.sel(element=elem).values,
            )
        p_fast = ph.calc_pressure(self.fast_dens.values, self.fast_temp.values)

        self.pressure_th.values = p_el + p_ion
        self.pressure_tot.values = p_el + p_ion + p_fast

        for t in self.t:
            self.pth.loc[dict(t=t)] = np.trapz(
                self.pressure_th.sel(t=t), self.volume.sel(t=t)
            )
            self.ptot.loc[dict(t=t)] = np.trapz(
                self.pressure_tot.sel(t=t), self.volume.sel(t=t)
            )
            # self.pf_par.loc[dict(t=t)] = np.trapz(
            #     self.fpressure_parallel.sel(t=t), self.volume.sel(t=t)
            # )
            # self.pf_perp.loc[dict(t=t)] = np.trapz(
            #     self.fpressure_perp.sel(t=t), self.volume.sel(t=t)
            # )
        # self.wfast = 1/2 *self.pf_par + self.pf_perp
        self.wth.values = 3 / 2 * self.pth.values
        self.wp.values = 3 / 2 * self.ptot.values

        if data is not None:
            if hasattr(self, "equil"):
                bckc[self.equil]["wp"].values = self.wp.values
            if "dialoop" in data.keys():
                bckc["dialoop"]["wdia"].values = self.wp.values
            return bckc

    def calc_zeff(self):
        """
        Calculate Zeff including all ion species
        """
        for elem in self.elements:
            self.zeff.loc[dict(element=elem)] = (
                self.ion_dens.sel(element=elem) * self.meanz.sel(element=elem) ** 2
            ) / self.el_dens

    def calc_vloop(self):
        """
        Given Zeff, Te and Ne: calculate resistivity and Vloop
        """

        self.conductivity = ph.conductivity_neo(
            self.el_dens,
            self.el_temp,
            self.zeff.sum("element"),
            self.min_r,
            self.r_a,
            self.R_mag,
            self.q_prof,
            approx="sauter",
        )
        for t in self.time:
            resistivity = 1.0 / self.conductivity.sel(t=t)
            ir = np.where(np.isfinite(resistivity))

            j_phi = self.j_phi.sel(t=t)
            area = self.area.sel(t=t)

            vloop = ph.vloop(resistivity[ir], j_phi[ir], area[ir])

            self.vloop.loc[dict(t=t)] = vloop

    def impose_flat_zeff(self):
        """
        Adapt impurity concentration to generate flat Zeff contribution
        """

        for elem in self.impurities:
            if np.count_nonzero(self.ion_dens.sel(element=elem)) != 0:
                zeff_tmp = (
                    self.ion_dens.sel(element=elem)
                    * self.meanz.sel(element=elem) ** 2
                    / self.el_dens
                )
                value = zeff_tmp.where(zeff_tmp.rho_poloidal < 0.2).mean("rho_poloidal")
                zeff_tmp = zeff_tmp / zeff_tmp * value
                ion_dens_tmp = zeff_tmp / (
                    self.meanz.sel(element=elem) ** 2 / self.el_dens
                )
                self.ion_dens.loc[dict(element=elem)] = ion_dens_tmp

        self.calc_zeff()

    def build_current_density(self):
        """
        Build current density profile (A/m**2) given the total plasma current,
        plasma geometry and a shape parameter
        """

        for t in self.time:
            rho = self.rho.values
            ipla = self.ipla.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            area = self.area.sel(t=t).values
            prof_shape = self.el_temp.sel(t=t) / self.el_temp.sel(t=t).max()

            j_phi = ph.current_density(ipla, rho, r_a, area, prof_shape)

            self.j_phi.loc[dict(t=t)] = j_phi * 10

    def calc_magnetic_field(self):
        """
        Calculate magnetic field profiles (poloidal & toroidal)
        """

        for t in self.time:
            R_bt_0 = self.R_bt_0.values
            R_mag = self.R_mag.sel(t=t).values
            ipla = self.ipla.sel(t=t).values
            bt_0 = self.bt_0.sel(t=t).values
            zmag = self.equilibrium.zmag.sel(t=t, method="nearest").values
            maj_r_lfs = self.maj_r_lfs.sel(t=t).values
            maj_r_hfs = self.maj_r_hfs.sel(t=t).values
            j_phi = self.j_phi.sel(t=t).values
            r_a = self.r_a.sel(t=t).values
            min_r = self.min_r.sel(t=t).values
            volume = self.volume.sel(t=t).values
            area = self.area.sel(t=t).values

            # self.b_tor_lfs.loc[dict(t=t)] = self.equilibrium.Btot(maj_r_lfs, np.full_like(maj_r_lfs, zmag), t)

            self.b_tor_lfs.loc[dict(t=t)] = ph.toroidal_field(bt_0, R_bt_0, maj_r_lfs)
            self.b_tor_hfs.loc[dict(t=t)] = ph.toroidal_field(bt_0, R_bt_0, maj_r_hfs)

            b_pol = ph.poloidal_field(j_phi, min_r, area)
            self.b_pol.loc[dict(t=t)] = b_pol
            self.l_i.loc[dict(t=t)] = ph.internal_inductance(
                b_pol, ipla, volume, approx=2, R_mag=R_mag
            )

            b_tor = ((self.b_tor_lfs.sel(t=t) + self.b_tor_hfs.sel(t=t)) / 2.0).values

            self.q_prof.loc[dict(t=t)] = ph.safety_factor(
                b_tor, b_pol, min_r, r_a, R_mag
            )

    def calc_beta_poloidal(self):
        """
        Calculate Beta poloidal

        ??? Use total or thermal pressure ???
        """

        for t in self.time:
            b_pol = self.b_pol.sel(t=t).values
            pressure = self.pressure_tot.sel(t=t).values
            volume = self.volume.sel(t=t).values

            self.beta_pol.loc[dict(t=t)] = ph.beta_poloidal(b_pol, pressure, volume)

    def propagate_ion_dens(self, fast_dens=False):
        """
        After having modified anything in the ion_density data, propagate the result to all
        other variables depending on it
        """
        self.calc_main_ion_dens(fast_dens=fast_dens)
        self.impose_flat_zeff()
        self.calc_pressure()
        self.calc_zeff()
        self.calc_vloop()
        self.calc_beta_poloidal()
        self.calc_rad_power()

    def add_transport(self):
        """
        Modify ionization distribution including transport
        """

        x_ped = 0.85
        diffusion = (
            xr.where(
                self.rho < x_ped,
                ph.gaussian(self.rho, 0.2, 0.02, x_ped, 0.3),
                ph.gaussian(self.rho, 0.2, 0.01, x_ped, 0.04),
            )
            * 2
        )

        for elem in self.elements:
            fz = (
                self.atomic_data[elem]["fz"]
                .interp(electron_temperature=self.el_temp, method="cubic")
                .drop_vars(["electron_temperature"])
            )
            fz_transp = deepcopy(fz)
            for t in self.time:
                fz_tmp = fz_transp.sel(t=t, drop=True)
                for i, rho in enumerate(self.rho):
                    gauss = (
                        ph.gaussian(self.rho, diffusion[i], 0, rho, diffusion[i] / 3)
                        * diffusion
                    )
                    gauss /= np.sum(gauss)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (fz_tmp * gauss).sum(
                        "rho_poloidal"
                    )
                for ir, rho in enumerate(self.rho):
                    norm = np.nansum(fz_tmp.sel(rho_poloidal=rho), axis=0)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = fz_tmp / norm
                    fz_transp.loc[dict(t=t)] = fz_tmp

                plt.figure()
                colors = cm.rainbow(np.linspace(0, 1, len(fz.ion_charges)))
                for i in fz.ion_charges:
                    plt.plot(
                        fz.rho_poloidal,
                        fz.sel(ion_charges=i).sel(t=t),
                        color=colors[i],
                    )
                    plt.plot(
                        fz_transp.rho_poloidal,
                        fz_transp.sel(ion_charges=i).sel(t=t),
                        "--",
                        color=colors[i],
                    )
                plt.title(f"Time = {t}")

    def initialize_variables(self):
        """
        Initialize all class attributes

        Assign elements, machine dimensions and coordinates used throughout the analysis
            rho
            time
            theta
        """

        # Assign attributes
        self.machine_R = np.linspace(
            self.machine_dimensions[0][0], self.machine_dimensions[0][1], 100
        )
        self.machine_z = np.linspace(
            self.machine_dimensions[1][0], self.machine_dimensions[1][1], 100
        )

        self.optimisation = {
            "equil": "",
            "el_dens": "",
            "el_temp": "",
            "ion_temp": "",
            "stored_en": "",
        }
        self.pulse = None

        nt = len(self.t)
        nr = len(self.radial_coordinate)
        nel = len(self.elements)
        nth = len(self.theta)

        coords_radius = (self.radial_coordinate_type, self.radial_coordinate)
        coords_theta = ("poloidal_angle", self.theta)
        coords_time = ("t", self.t)
        coords_elem = ("element", list(self.elements))

        data0d = DataArray(0.0)
        data1d_theta = DataArray(np.zeros(nth), coords=[coords_theta])
        data1d_time = DataArray(np.zeros(nt), coords=[coords_time])
        data1d_rho = DataArray(np.zeros(nr), coords=[coords_radius])
        data2d = DataArray(np.zeros((nt, nr)), coords=[coords_time, coords_radius])
        data2d_elem = DataArray(np.zeros((nel, nt)), coords=[coords_elem, coords_time])
        data3d = DataArray(
            np.zeros((nel, nt, nr)), coords=[coords_elem, coords_time, coords_radius]
        )

        self.time = deepcopy(data1d_time)
        self.time.values = self.t
        assign_datatype(self.time, ("t", "plasma"))
        self.freq = 1.0 / self.dt

        self.rho = deepcopy(data1d_rho)
        self.rho.values = self.radial_coordinate
        rho_type = self.radial_coordinate_type.split("_")
        if rho_type[1] != "poloidal":
            print_like("Only poloidal rho in input for the time being...")
            raise AssertionError
        assign_datatype(self.rho, (rho_type[0], rho_type[1]))

        self.Te_prof = Profiles(datatype=("temperature", "electron"), xspl=self.rho)
        self.Ti_prof = Profiles(datatype=("temperature", "ion"), xspl=self.rho)
        self.Ne_prof = Profiles(datatype=("density", "electron"), xspl=self.rho)
        self.Nimp_prof = Profiles(datatype=("density", "impurity"), xspl=self.rho)
        self.Nimp_prof.y1 = 3.0e19
        self.Nimp_prof.yend = 3.0e19
        self.Nimp_prof.build_profile()
        self.Nh_prof = Profiles(datatype=("neutral_density", "neutrals"), xspl=self.rho)
        self.Vrot_prof = Profiles(datatype=("rotation", "ion"), xspl=self.rho)

        # self.rhot = deepcopy(data2d)
        # rhot, _ = self.equilibrium.convert_flux_coords(self.rho)
        # self.rhot.values = bin_in_time(self.tstart, self.tend, self.freq, rhot).interp(
        #     t=self.time, method="linear"
        # )
        # assign_datatype(self.rhot, ("rho", "toroidal"))

        self.theta = deepcopy(data1d_theta)
        self.theta.values = self.theta
        assign_datatype(self.theta, ("angle", "poloidal"))

        self.ipla = deepcopy(data1d_time)
        assign_datatype(self.ipla, ("current", "plasma"))

        self.bt_0 = deepcopy(data1d_time)
        assign_datatype(self.bt_0, ("field", "toroidal"))

        self.R_bt_0 = deepcopy(data0d)
        self.R_bt_0.values = self.R_bt_0
        assign_datatype(self.R_bt_0, ("major_radius", "toroidal_field"))

        # Geometric major radius
        self.R_0 = deepcopy(data1d_time)
        assign_datatype(self.R_0, ("major_radius", "geometric"))

        self.R_mag = deepcopy(data1d_time)
        assign_datatype(self.R_mag, ("major_radius", "magnetic"))

        # Major radius array at midplane
        self.maj_r_lfs = deepcopy(data2d)
        assign_datatype(self.maj_r_lfs, ("radius", "major"))
        self.maj_r_hfs = deepcopy(data2d)
        assign_datatype(self.maj_r_hfs, ("radius", "major"))

        # Main plasma profiles
        self.ne_0 = deepcopy(data1d_time)
        assign_datatype(self.ne_0, ("density", "electron"))

        self.te_0 = deepcopy(data1d_time)
        assign_datatype(self.te_0, ("temperature", "electron"))

        self.ti_0 = deepcopy(data1d_time)
        assign_datatype(self.ti_0, ("temperature", "ion"))

        self.el_temp = deepcopy(data2d)
        assign_datatype(self.el_temp, ("temperature", "electron"))
        self.el_dens = deepcopy(data2d)
        assign_datatype(self.el_dens, ("density", "electron"))
        self.neutral_dens = deepcopy(data2d)
        assign_datatype(self.neutral_dens, ("density", "neutral"))

        self.tau = deepcopy(data2d)
        assign_datatype(self.tau, ("time", "residence"))

        # Other geometrical quantities
        self.min_r = deepcopy(data2d)
        assign_datatype(self.min_r, ("minor_radius", "plasma"))
        self.cr0 = deepcopy(data1d_time)
        assign_datatype(self.cr0, ("minor_radius", "separatrizx"))
        self.rmag = deepcopy(data1d_time)
        assign_datatype(self.rmag, ("major_radius", "magnetic_axis"))
        self.zmag = deepcopy(data1d_time)
        assign_datatype(self.rmag, ("z", "magnetic_axis"))
        self.volume = deepcopy(data2d)
        assign_datatype(self.volume, ("volume", "plasma"))
        self.area = deepcopy(data2d)
        assign_datatype(self.area, ("area", "plasma"))

        self.r_a = deepcopy(data1d_time)
        assign_datatype(self.r_a, ("minor_radius", "LFS"))
        self.r_b = deepcopy(data1d_time)
        assign_datatype(self.r_b, ("minor_radius", "top"))
        self.r_c = deepcopy(data1d_time)
        assign_datatype(self.r_c, ("minor_radius", "HFS"))
        self.r_d = deepcopy(data1d_time)
        assign_datatype(self.r_d, ("minor_radius", "bottom"))

        self.kappa = deepcopy(data1d_time)
        assign_datatype(self.kappa, ("elongation", "plasma"))
        self.delta = deepcopy(data1d_time)
        assign_datatype(self.delta, ("triangularity", "plasma"))

        # Fast particle density and temperature
        self.fast_temp = deepcopy(data2d)
        assign_datatype(self.fast_temp, ("temperature", "fast"))

        self.fast_dens = deepcopy(data2d)
        assign_datatype(self.fast_dens, ("density", "fast"))

        # Current density, poloidal field, li, resistivity, vloop, q-profile
        self.j_phi = deepcopy(data2d)
        assign_datatype(self.j_phi, ("current", "density"))
        self.b_pol = deepcopy(data2d)
        assign_datatype(self.b_pol, ("field", "poloidal"))
        self.b_tor_lfs = deepcopy(data2d)
        assign_datatype(self.b_tor_lfs, ("field", "toroidal"))
        self.b_tor_hfs = deepcopy(data2d)
        assign_datatype(self.b_tor_hfs, ("field", "toroidal"))
        self.q_prof = deepcopy(data2d)
        assign_datatype(self.q_prof, ("factor", "safety"))
        self.conductivity = deepcopy(data2d)
        assign_datatype(self.conductivity, ("conductivity", "plasma"))
        self.l_i = deepcopy(data1d_time)
        assign_datatype(self.l_i, ("inductance", "internal"))
        self.vloop = deepcopy(data1d_time)
        assign_datatype(self.vloop, ("voltage", "loop"))

        # Ion densities
        self.main_ion = self.elements[0]
        self.impurities = self.elements[1:]

        self.ion_conc = DataArray(np.zeros(len(self.elements)), coords=[coords_elem])
        assign_datatype(self.ion_conc, ("concentration", "ion"))

        self.ion_dens = deepcopy(data3d)
        assign_datatype(self.ion_dens, ("density", "ion"))
        self.ion_temp = deepcopy(data3d)
        assign_datatype(self.ion_temp, ("temperature", "ion"))
        self.vtor = deepcopy(data3d)
        assign_datatype(self.vtor, ("temperature", "ion"))
        self.meanz = deepcopy(data3d)
        assign_datatype(self.meanz, ("charge", "mean"))

        self.zeff = deepcopy(data3d)
        assign_datatype(self.zeff, ("charge", "effective"))

        self.tot_rad = deepcopy(data3d)
        assign_datatype(self.tot_rad, ("radiation_emission", "total"))
        self.sxr_rad = deepcopy(data3d)
        assign_datatype(self.sxr_rad, ("radiation_emission", "sxr"))

        self.prad = deepcopy(data2d_elem)
        assign_datatype(self.tot_rad, ("radiation", "total"))

        self.pressure_th = deepcopy(data2d)
        assign_datatype(self.pressure_th, ("pressure", "thermal"))
        self.pressure_tot = deepcopy(data2d)
        assign_datatype(self.pressure_tot, ("pressure", "total"))

        self.pth = deepcopy(data1d_time)
        assign_datatype(self.pth, ("pressure", "thermal"))
        self.ptot = deepcopy(data1d_time)
        assign_datatype(self.ptot, ("pressure", "total"))

        self.wth = deepcopy(data1d_time)
        assign_datatype(self.wth, ("stored_energy", "thermal"))
        self.wp = deepcopy(data1d_time)
        assign_datatype(self.wp, ("stored_energy", "total"))
        self.beta_pol = deepcopy(data1d_time)
        assign_datatype(self.beta_pol, ("beta", "poloidal"))

    def write_to_pickle(self):

        with open(f"data_{self.pulse}.pkl", "wb") as f:
            pickle.dump(
                self, f,
            )


def initialize_bckc(diagnostic, quantity, data, bckc={}):
    """
    Initialise back-calculated data with all info as original data, apart
    from provenance and revision attributes

    Parameters
    ----------
    data
        DataArray of original data to be "cloned"

    Returns
    -------

    """
    if diagnostic not in bckc:
        bckc[diagnostic] = {}

    data_tmp = data[diagnostic][quantity]
    bckc_tmp = xr.full_like(data_tmp, np.nan)
    attrs = bckc_tmp.attrs
    if type(bckc_tmp) == DataArray:
        if "error" in attrs.keys():
            attrs["error"] = xr.full_like(attrs["error"], np.nan)
        if "partial_provenance" in attrs.keys():
            attrs.pop("partial_provenance")
            attrs.pop("provenance")
        if "revision" in attrs.keys():
            attrs.pop("revision")
    bckc_tmp.attrs = attrs

    bckc[diagnostic][quantity] = bckc_tmp

    return bckc


def remap_diagnostic(diag_data, flux_transform, npts=300):
    """
    Calculate maping on equilibrium for speccified diagnostic

    Returns
    -------

    """
    new_attrs = {}
    trans = diag_data.attrs["transform"]
    x1 = diag_data.coords[trans.x1_name]
    x2_arr = np.linspace(0, 1, npts)
    x2 = DataArray(x2_arr, dims=trans.x2_name)
    dl = trans.distance(trans.x2_name, DataArray(0), x2[0:2], 0)[1]
    new_attrs["x2"] = x2
    new_attrs["dl"] = dl
    new_attrs["R"], new_attrs["z"] = trans.convert_to_Rz(x1, x2, 0)
    rho_equil, _ = flux_transform.convert_from_Rz(new_attrs["R"], new_attrs["z"])
    rho = rho_equil.interp(t=diag_data.t, method="linear")
    rho = xr.where(rho >= 0, rho, 0.0)
    new_attrs["rho"] = rho

    return new_attrs


def assign_datatype(data_array: DataArray, datatype: tuple):
    data_array.name = f"{datatype[1]}_{datatype[0]}"
    data_array.attrs["datatype"] = datatype


def print_like(string):
    print(f"\n {string}")
