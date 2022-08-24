from copy import deepcopy
import pickle

import hda.physics as ph
from hda.profiles import Profiles
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import least_squares
import xarray as xr
from xarray import DataArray
from xarray import Dataset

from indica.converters import FluxSurfaceCoordinates
from indica.converters.time import bin_in_time_dt
from indica.converters.time import get_tlabels_dt
from indica.datatypes import ELEMENTS
from indica.equilibrium import Equilibrium
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.provenance import get_prov_attribute
from indica.readers import ADASReader

plt.ion()
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

    def map_to_midplane(self):
        # TODO: streamline this to avoid continuously re-calculating quantities e.g. ion_dens..
        keys = [
            "el_dens",
            "ion_dens",
            "neutral_dens",
            "el_temp",
            "ion_temp",
            "pressure_th",
            "vtor",
            "zeff",
            "meanz",
            "volume",
        ]

        nchan = len(self.R_midplane)
        chan = np.arange(nchan)
        R = DataArray(self.R_midplane, coords=[("channel", chan)])
        z = DataArray(self.z_midplane, coords=[("channel", chan)])

        midplane_profs = {}
        for k in keys:
            k_hi = f"{k}_hi"
            k_lo = f"{k}_lo"

            midplane_profs[k] = []
            if hasattr(self, k_hi):
                midplane_profs[k_hi] = []
            if hasattr(self, k_lo):
                midplane_profs[k_lo] = []

        for k in midplane_profs.keys():
            for t in self.t:
                rho = (
                    self.equilibrium.rho.sel(t=t, method="nearest")
                    .interp(R=R, z=z)
                    .drop(["R", "z"])
                )
                midplane_profs[k].append(
                    getattr(self, k)
                    .sel(t=t, method="nearest")
                    .interp(rho_poloidal=rho)
                    .drop("rho_poloidal")
                )
            midplane_profs[k] = xr.concat(midplane_profs[k], "t").assign_coords(
                t=self.t
            )
            midplane_profs[k] = xr.where(
                np.isfinite(midplane_profs[k]), midplane_profs[k], 0.0
            )

        self.midplane_profs = midplane_profs

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
            diagnostic data as returned by build_data()
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
            Default value calculated to match Zeff before Ar puff from
            LINES.BREMS_MP for pulse 9408

        Returns
        -------
        bckc
            dictionary with back calculated value(s)

        """
        zeff = self.zeff
        brems = ph.zeff_bremsstrahlung(
            self.el_temp, self.el_dens, wavelength, zeff=zeff.sum("element")
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
        quantity_te="te_n3w",
        quantity_ti="ti_w",
        half_los=True,
        use_ratios=False,
        use_satellites=False,
        time=None,
        calc_error=False,
        use_ref=True,
        wcenter_exp=0.05,
        method="dogbox",
    ):
        """
        Rescale temperature profiles to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_te
            Measurement to be used for the electron temperature optimisation
        quantity_ti
            Measurement to be used for the ion temperature optimisation
        niter
            Number of iterations

        Returns
        -------

        """

        def line_ratios(forward_model, quantity_ratio):
            if quantity_ratio == "int_k/int_w":
                ratio_bckc = forward_model.intensity["k"] / forward_model.intensity["w"]
            elif quantity_ratio == "int_n3/int_w":
                ratio_bckc = (
                    forward_model.intensity["n3"] / forward_model.intensity["w"]
                )
            elif quantity_ratio == "int_n3/int_tot":
                int_tot = (
                    forward_model.intensity["n3"]
                    + forward_model.intensity["n345"]
                    + forward_model.intensity["w"]
                )
                ratio_bckc = forward_model.intensity["n3"] / int_tot

            return ratio_bckc

        def residuals_te_ratio(te0):
            """
            Optimisation for line ratios
            """
            Te_prof.y0 = te0
            Te_prof.build_profile()
            Te = Te_prof.yspl
            _ = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            ratio_bckc = line_ratios(forward_model, quantity_ratio)
            resid = ratio_data - ratio_bckc
            return resid

        def residuals_te(te0):
            print(te0)
            Te_prof.y0 = te0
            Te_prof.build_profile()
            Te = Te_prof.yspl
            Ti = Ti_prof.yspl
            _bckc = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                Ti=Ti,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            te_bckc = _bckc[quantity_te]
            resid = te_data - te_bckc
            return resid

        def residuals_ti(ti0):
            Ti_prof.y0 = ti0
            if use_ref:
                Ti_prof.build_profile(y0_ref=Te_prof.y0, wcenter_exp=wcenter_exp)
            else:
                Ti_prof.build_profile()
            Ti = Ti_prof.yspl

            _bckc = forward_model.moment_analysis(
                Ti,
                rho_los=rho_los,
                dl=dl,
                half_los=half_los,
                use_satellites=use_satellites,
            )
            ti_bckc = _bckc[quantity_ti]
            resid = ti_data - ti_bckc
            return resid

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

        if quantity_te == "te_kw":
            quantity_ratio = "int_k/int_w"
        elif quantity_te == "te_n3w":
            quantity_ratio = "int_n3/int_tot"
            # quantity_ratio = "int_n3/int_w"
        else:
            print_like(f"{quantity_te} not available for ratio calculation")
            raise ValueError

        bckc = initialize_bckc(diagnostic, quantity_te, data, bckc=bckc)
        bckc = initialize_bckc(diagnostic, quantity_ti, data, bckc=bckc)
        bckc = initialize_bckc(diagnostic, quantity_ratio, data, bckc=bckc)

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        Te_prof = self.Te_prof
        Ti_prof = self.Ti_prof

        pos = xr.full_like(data[diagnostic][quantity_te].t, np.nan)
        err_in = deepcopy(pos)
        err_out = deepcopy(pos)
        te_pos = Dataset({"value": pos, "err_in": err_in, "err_out": err_out})
        ti_pos = deepcopy(te_pos)
        bounds_te = (100.0, 10.0e3)
        bounds_ti = (100.0, 30.0e3)
        if method == "lm":
            bounds_te = (-np.inf, np.inf)
            bounds_ti = (-np.inf, np.inf)

        # Initialize variables
        emiss_tmp: list = []
        fz_tmp: list = []
        forward_model.radiation_characteristics(self.Te_prof.yspl, self.Ne_prof.yspl)
        for t in self.t:
            emiss_tmp.append(forward_model.emiss["w"] * 0.0)
            fz_tmp.append(forward_model.fz["ar"] * 0.0)
        emiss = xr.concat(emiss_tmp, "t").assign_coords(t=self.t)
        fz = xr.concat(fz_tmp, "t").assign_coords(t=self.t)

        dl = data[diagnostic][quantity_ti].attrs["dl"]

        if calc_error:
            self.el_temp_hi = deepcopy(self.el_temp)
            self.el_temp_lo = deepcopy(self.el_temp)
            self.ion_temp_hi = deepcopy(self.ion_temp)
            self.ion_temp_lo = deepcopy(self.ion_temp)

        for t in time:
            print(t)
            Ne = self.el_dens.sel(t=t)
            Nimp = {"ar": self.ion_dens.sel(element="ar").sel(t=t)}
            Nh = self.neutral_dens.sel(t=t)

            if use_ratios:
                ratio_data = data[diagnostic][quantity_ratio].sel(t=t)
            te_data = deepcopy(data[diagnostic][quantity_te].sel(t=t))
            ti_data = deepcopy(data[diagnostic][quantity_ti].sel(t=t))
            rho_los = data[diagnostic][quantity_ti].attrs["rho"].sel(t=t)

            if t == time[0]:
                te0 = te_data.values
                ti0 = ti_data.values
            else:
                te0 = Te_prof.y0
                ti0 = Ti_prof.y0

            if not ((te0 > 0) * (ti0 > 0)):
                self.el_temp.loc[dict(t=t)] = np.full_like(Te_prof.yspl.values, np.nan)
                for elem in self.elements:
                    self.ion_temp.loc[dict(t=t, element=elem)] = np.full_like(
                        Te_prof.yspl.values, np.nan
                    )
                continue

            if use_ratios:
                if calc_error:
                    ratio_tmp = deepcopy(ratio_data)

                    # Upper limit of the ratio
                    ratio_data = ratio_tmp + ratio_tmp.attrs["error"]
                    least_squares(
                        residuals_te_ratio, te0, bounds=bounds_te, method=method
                    )
                    least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)

                    Te_lo = deepcopy(Te_prof)
                    Ti_lo = deepcopy(Ti_prof)

                    # Lower limit of the ratio
                    ratio_data = ratio_tmp - ratio_tmp.attrs["error"]
                    least_squares(
                        residuals_te_ratio, te0, bounds=bounds_te, method=method
                    )
                    least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)
                    Te_hi = deepcopy(Te_prof)
                    Ti_hi = deepcopy(Ti_prof)

                    te0 = np.mean(Te_hi.y0 + Te_lo.y0) / 2.0
                    ti0 = np.mean(Ti_hi.y0 + Ti_lo.y0) / 2.0
                    ratio_data = ratio_tmp

                _ = least_squares(
                    residuals_te_ratio, te0, bounds=bounds_te, method=method
                )
                least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)
            else:
                if calc_error:
                    print_like(
                        """Error calculation currently available
                        only for ratio optimisation"""
                    )
                    raise ValueError

                least_squares(residuals_te, te0, bounds=bounds_te, method=method)
                least_squares(residuals_ti, ti0, bounds=bounds_ti, method=method)

            Te = Te_prof.yspl
            Ti = Ti_prof.yspl
            _bckc = forward_model(
                Te,
                Ne,
                Nimp=Nimp,
                Nh=Nh,
                Ti=Ti,
                rho_los=rho_los,
                dl=dl,
                use_satellites=use_satellites,
            )
            if use_ratios:
                ratio_bckc = line_ratios(forward_model, quantity_ratio)
            te_bckc = _bckc[quantity_te]
            ti_bckc = _bckc[quantity_ti]

            te_pos.value.loc[dict(t=t)] = te_bckc.rho_poloidal.values[0]
            te_pos.err_in.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["in"]
            te_pos.err_out.loc[dict(t=t)] = te_bckc.attrs["rho_poloidal_err"]["out"]

            ti_pos.value.loc[dict(t=t)] = ti_bckc.rho_poloidal.values[0]
            ti_pos.err_in.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["in"]
            ti_pos.err_out.loc[dict(t=t)] = ti_bckc.attrs["rho_poloidal_err"]["out"]

            emiss.loc[dict(t=t)] = forward_model.emiss["w"].values
            fz.loc[dict(t=t)] = forward_model.fz["ar"].values

            bckc[diagnostic][quantity_te].loc[dict(t=t)] = te_bckc.values[0]
            bckc[diagnostic][quantity_ti].loc[dict(t=t)] = ti_bckc.values[0]
            if use_ratios:
                bckc[diagnostic][quantity_ratio].loc[dict(t=t)] = ratio_bckc.values

            self.el_temp.loc[dict(t=t)] = Te_prof.yspl.values
            if calc_error:
                self.el_temp_hi.loc[dict(t=t)] = Te_hi.yspl.values
                self.el_temp_lo.loc[dict(t=t)] = Te_lo.yspl.values
            for elem in self.elements:
                self.ion_temp.loc[dict(t=t, element=elem)] = Ti_prof.yspl.values
                if calc_error:
                    self.ion_temp_hi.loc[dict(t=t, element=elem)] = Ti_hi.yspl.values
                    self.ion_temp_lo.loc[dict(t=t, element=elem)] = Ti_lo.yspl.values

        bckc[diagnostic][quantity_te].attrs["pos"] = te_pos
        bckc[diagnostic][quantity_te].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_te].attrs["fz"] = fz

        bckc[diagnostic][quantity_ti].attrs["pos"] = ti_pos
        bckc[diagnostic][quantity_ti].attrs["emiss"] = emiss
        bckc[diagnostic][quantity_ti].attrs["fz"] = fz

        revision = get_prov_attribute(
            data[diagnostic][quantity_te].provenance, "revision"
        )
        self.optimisation["el_temp"] = f"{diagnostic}.{quantity_te}:{revision}"

        revision = get_prov_attribute(
            data[diagnostic][quantity_ti].provenance, "revision"
        )
        self.optimisation["ion_temp"] = f"{diagnostic}.{quantity_ti}:{revision}"

        return bckc

    def match_xrcs_intensity(
        self,
        data,
        bckc={},
        diagnostic: str = "xrcs",
        quantity: str = "int_w",
        elem="ar",
        cal=2.0e3,
        niter=2,
        time=None,
        scale=True,
    ):
        """
        TODO: separate calculation of line intensity from optimisation
        TODO: tau currently not included in calculation
        Compute Ar density to match the XRCS spectrometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by build_data()
        bckc
            back-calculated data
        diagnostic
            diagnostic name corresponding to xrcs
        quantity_int
            Measurement to be used for determining the impurity concentration
            from line intensity
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
        if quantity not in bckc[diagnostic].keys():
            bckc = initialize_bckc(diagnostic, quantity, data, bckc=bckc)
        line = quantity.split("_")[1]

        # Initialize back calculated values of diagnostic quantities
        forward_model = self.forward_models[diagnostic]
        dl = data[diagnostic][quantity].attrs["dl"]
        for t in time:
            print(t)

            int_data = data[diagnostic][quantity].sel(t=t)
            Te = self.el_temp.sel(t=t)
            if np.isnan(Te).any():
                continue

            Ne = self.el_dens.sel(t=t)
            # tau = self.tau.sel(t=t)
            Nh = self.neutral_dens.sel(t=t)
            if np.isnan(Te).any():
                continue
            rho_los = data[diagnostic][quantity].attrs["rho"].sel(t=t)

            const = 1.0
            for j in range(niter):
                Nimp = {elem: self.imp_dens.sel(element=elem, t=t) * const}
                _ = forward_model(Te, Ne, Nimp=Nimp, Nh=Nh, rho_los=rho_los, dl=dl,)
                int_bckc = forward_model.intensity[line] * cal
                const = (int_data / int_bckc).values

                if (np.abs(1 - const) < 1.0e-4) or not (scale):
                    break

            self.imp_dens.loc[dict(element=elem, t=t)] = Nimp[elem].values
            bckc[diagnostic][quantity].loc[dict(t=t)] = int_bckc.values

        bckc[diagnostic][quantity].attrs["calibration"] = cal

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["imp_dens"] = f"{diagnostic}.{quantity}:{revision}"

        return bckc

    def match_interferometer(
        self,
        data,
        bckc={},
        diagnostic: str = "nirh1",
        quantity: str = "ne_bin",
        error=False,
        niter=3,
        time=None,
    ):
        """
        Rescale density profiles to match the interferometer measurements

        Parameters
        ----------
        data
            diagnostic data as returned by build_data()
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

            bckc[diagnostic][quantity].loc[dict(t=t)] = ne_bckc.values

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["el_dens"] = f"{diagnostic}.{quantity}:{revision}"
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
            bckc_tmp = self.wp.sel()
            const = 1 + (data_tmp - bckc_tmp) / bckc_tmp

        revision = get_prov_attribute(data[diagnostic][quantity].provenance, "revision")
        self.optimisation["stored_en"] = f"{diagnostic}.{quantity}:{revision}"

    def calc_ne_los_int(self, data, passes=2, t=None):
        """
        Calculate line of sight integral for a specified number of
        passes through the plasma

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

    def calc_centrifugal_asymmetry(self, time=None, test_vtor=None, plot=False):
        """
        Calculate (R, z) maps of the ion densities caused by centrifugal asymmetry
        """
        if time is None:
            time = self.t

        # TODO: make this attribute creation a property and standardize?
        if not hasattr(self, "ion_dens_2d"):
            self.rho_2d = self.equilibrium.rho.interp(t=self.t, method="nearest")
            tmp = deepcopy(self.rho_2d)
            ion_dens_2d = []
            for elem in self.elements:
                ion_dens_2d.append(tmp)

            self.ion_dens_2d = xr.concat(ion_dens_2d, "element").assign_coords(
                element=self.elements
            )
            assign_data(self.ion_dens_2d, ("density", "ion"))
            self.centrifugal_asymmetry = deepcopy(self.ion_dens)
            assign_data(self.centrifugal_asymmetry, ("asymmetry", "centrifugal"))
            self.asymmetry_multiplier = deepcopy(self.ion_dens_2d)
            assign_data(
                self.asymmetry_multiplier, ("asymmetry_multiplier", "centrifugal")
            )

        # If toroidal rotation != 0 calculate ion density on 2D poloidal plane
        if test_vtor is not None:
            vtor = deepcopy(self.ion_temp)
            assign_data(vtor, ("rotation", "toroidal"), "rad/s")
            vtor /= vtor.max("rho_poloidal")
            vtor *= test_vtor  # rad/s
            self.vtor = vtor

        if not np.any(self.vtor != 0):
            return

        zeff = self.zeff.sum("element")
        R_0 = self.maj_r_lfs.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
        for elem in self.elements:
            main_ion_mass = ELEMENTS[self.main_ion][1]
            mass = ELEMENTS[elem][1]
            asymm = ph.centrifugal_asymmetry(
                self.ion_temp.sel(element=elem).drop("element"),
                self.el_temp,
                mass,
                self.meanz.sel(element=elem).drop("element"),
                zeff,
                main_ion_mass,
                toroidal_rotation=self.vtor.sel(element=elem).drop("element"),
            )
            self.centrifugal_asymmetry.loc[dict(element=elem)] = asymm
            asymmetry_factor = asymm.interp(rho_poloidal=self.rho_2d)
            self.asymmetry_multiplier.loc[dict(element=elem)] = np.exp(
                asymmetry_factor * (self.rho_2d.R ** 2 - R_0 ** 2)
            )

        self.ion_dens_2d = (
            self.ion_dens.interp(rho_poloidal=self.rho_2d).drop("rho_poloidal")
            * self.asymmetry_multiplier
        )
        assign_data(self.ion_dens_2d, ("density", "ion"), "m^-3")

        if plot:
            t = self.t[6]
        for elem in self.elements:
            plt.figure()
            z = self.z_mag.sel(t=t)
            rho = self.rho_2d.sel(t=t).sel(z=z, method="nearest")
            plt.plot(
                rho, self.ion_dens_2d.sel(element=elem).sel(t=t, z=z, method="nearest"),
            )
            self.ion_dens.sel(element=elem).sel(t=t).plot(linestyle="dashed")
            plt.title(elem)

        elem = "ar"
        plt.figure()
        np.log(self.ion_dens_2d.sel(element=elem).sel(t=t, method="nearest")).plot()
        self.rho_2d.sel(t=t, method="nearest").plot.contour(levels=10, colors="white")
        plt.xlabel("R (m)")
        plt.ylabel("z (m)")
        plt.title(f"log({elem} density")
        plt.axis("scaled")
        plt.xlim(0, 0.8)
        plt.ylim(-0.6, 0.6)

    def calc_rad_power_2d(self):
        """
        Calculate total and SXR filtered radiated power on a 2D poloidal plane
        including effects from poloidal asymmetries
        """
        for elem in self.elements:
            tot_rad = (
                self.lz_tot[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            tot_rad = xr.where(tot_rad >= 0, tot_rad, 0.0,)
            self.tot_rad.loc[dict(element=elem)] = tot_rad.values

            sxr_rad = (
                self.lz_sxr[elem].sum("ion_charges")
                * self.el_dens
                * self.ion_dens.sel(element=elem)
            )
            sxr_rad = xr.where(sxr_rad >= 0, sxr_rad, 0.0,)
            self.sxr_rad.loc[dict(element=elem)] = sxr_rad.values

            if not hasattr(self, "prad_tot"):
                self.prad_tot = deepcopy(self.prad)
                self.prad_sxr = deepcopy(self.prad)
                assign_data(self.prad_sxr, ("radiation", "sxr"))

            prad_tot = self.prad_tot.sel(element=elem)
            prad_sxr = self.prad_sxr.sel(element=elem)
            for t in self.t:
                prad_tot.loc[dict(t=t)] = np.trapz(
                    tot_rad.sel(t=t), self.volume.sel(t=t)
                )
                prad_sxr.loc[dict(t=t)] = np.trapz(
                    sxr_rad.sel(t=t), self.volume.sel(t=t)
                )
            self.prad_tot.loc[dict(element=elem)] = prad_tot.values
            self.prad_sxr.loc[dict(element=elem)] = prad_sxr.values

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
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (
                        (fz_tmp * gauss).sum("rho_poloidal").values
                    )
                for ir, rho in enumerate(self.rho):
                    norm = np.nansum(fz_tmp.sel(rho_poloidal=rho), axis=0)
                    fz_tmp.loc[dict(rho_poloidal=rho)] = (fz_tmp / norm).values
                    fz_transp.loc[dict(t=t)] = fz_tmp.values

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
    bckc_tmp.attrs = attrs

    bckc[diagnostic][quantity] = bckc_tmp

    return bckc


def remap_diagnostic(diag_data, flux_transform, npts=100):
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

    dt_equil = flux_transform.equilibrium.rho.t[1] - flux_transform.equilibrium.rho.t[0]
    dt_data = diag_data.t[1] - diag_data.t[0]
    if dt_data > dt_equil:
        t = diag_data.t
    else:
        t = None
    rho_equil, _ = flux_transform.convert_from_Rz(new_attrs["R"], new_attrs["z"], t=t)
    rho = rho_equil.interp(t=diag_data.t, method="linear")
    rho = xr.where(rho >= 0, rho, 0.0)
    rho.coords[trans.x2_name] = x2
    new_attrs["rho"] = rho

    return new_attrs


def assign_data(data: DataArray, datatype: tuple, unit="", make_copy=True):
    if make_copy:
        new_data = deepcopy(data)
    else:
        new_data = data

    new_data.name = f"{datatype[1]}_{datatype[0]}"
    new_data.attrs["datatype"] = datatype
    if len(unit) > 0:
        new_data.attrs["unit"] = unit

    return new_data


def print_like(string):
    print(f"\n {string}")


def build_data(plasma: Plasma, data, equil="efit", instrument="", pulse=None):
    """
    Reorganise raw data dictionary on the desired time axis and generate
    geometry information from equilibrium reconstruction

    Parameters
    ----------
    plasma
        Plasma class
    data
        Raw data dictionary
    equil
        Equilibrium code to use for equilibrium object
    instrument
        Build data only for specified instrument

    Returns
    -------

    """
    print_like("Building data class")

    if len(instrument) == 0:
        plasma.initialize_variables()

        plasma.pulse = pulse
        revision = get_prov_attribute(data[equil]["rmag"].provenance, "revision")
        plasma.optimisation["equil"] = f"{equil}:{revision}"

        t_ip = data["efit"]["ipla"].t
        if plasma.tstart < t_ip.min():
            print_like("Start time changed to stay inside Ip limit")
            plasma.tstart = t_ip.min().values
        if plasma.tend > t_ip.max():
            print_like("End time changed to stay inside Ip limit")
            plasma.tend = t_ip.max().values

        plasma.equil = equil

        if equil in data.keys():
            print_like("Initialise equilibrium object")
            plasma.equilibrium = Equilibrium(data[equil])
            plasma.flux_coords = FluxSurfaceCoordinates("poloidal")
            plasma.flux_coords.set_equilibrium(plasma.equilibrium)

    print_like("Assign equilibrium, bin data in time")
    binned_data = {}

    for kinstr in data.keys():
        if (len(instrument) > 0) and (kinstr != instrument):
            continue
        instrument_data = {}

        if type(data[kinstr]) != dict:
            value = deepcopy(data[kinstr])
            if np.size(value) > 1:
                value = bin_in_time_dt(plasma.tstart, plasma.tend, plasma.dt, value)
            binned_data[kinstr] = value
            continue

        for kquant in data[kinstr].keys():
            value = data[kinstr][kquant]
            if "t" in value.coords:
                value = bin_in_time_dt(plasma.tstart, plasma.tend, plasma.dt, value)

            if "transform" in data[kinstr][kquant].attrs:
                value.attrs["transform"] = data[kinstr][kquant].transform
                value.transform.set_equilibrium(plasma.equilibrium, force=True)
                if "LinesOfSightTransform" in str(value.attrs["transform"]):
                    geom_attrs = remap_diagnostic(value, plasma.flux_coords)
                    for kattrs in geom_attrs:
                        value.attrs[kattrs] = geom_attrs[kattrs]

            if "provenance" in data[kinstr][kquant].attrs:
                value.attrs["provenance"] = data[kinstr][kquant].provenance

            instrument_data[kquant] = value

        binned_data[kinstr] = instrument_data
        if kinstr == instrument:
            break

    if (len(instrument) == 0) and ("efit" in binned_data.keys()):
        plasma.ipla.values = binned_data["efit"]["ipla"]
        plasma.cr0.values = (
            binned_data["efit"]["rmjo"] - binned_data["efit"]["rmji"]
        ).sel(rho_poloidal=1) / 2.0

        plasma.R_mag = binned_data["efit"]["rmag"]
        plasma.z_mag = binned_data["efit"]["zmag"]

        plasma.R_bt_0 = binned_data["R_bt_0"]
        plasma.bt_0 = binned_data["bt_0"]

    return binned_data


def average_runs(plasma_dict: dict):
    runs = list(plasma_dict)
    pl_avrg = deepcopy(plasma_dict[runs[0]])
    el_dens, imp_dens, neutral_dens, el_temp, ion_temp = (
        [],
        [],
        [],
        [],
        [],
    )
    runs = []
    for run_name, pl in plasma_dict.items():
        runs.append(run_name)
        el_dens.append(pl.el_dens)
        imp_dens.append(pl.imp_dens)
        neutral_dens.append(pl.neutral_dens)
        el_temp.append(pl.el_temp)
        ion_temp.append(pl.ion_temp)

    el_dens = xr.concat(el_dens, "run_name").assign_coords({"run_name": runs})
    stdev = el_dens.std("run_name")
    pl_avrg.el_dens = el_dens.mean("run_name")
    pl_avrg.el_dens_hi = pl_avrg.el_dens + stdev
    pl_avrg.el_dens_lo = pl_avrg.el_dens - stdev

    ion_dens = xr.concat(imp_dens, "run_name").assign_coords({"run_name": runs})
    stdev = ion_dens.std("run_name")
    pl_avrg.imp_dens = ion_dens.mean("run_name")
    pl_avrg.imp_dens_hi = pl_avrg.imp_dens + stdev
    pl_avrg.imp_dens_lo = pl_avrg.imp_dens - stdev

    neutral_dens = xr.concat(neutral_dens, "run_name").assign_coords({"run_name": runs})
    stdev = neutral_dens.std("run_name")
    pl_avrg.neutral_dens = neutral_dens.mean("run_name")
    pl_avrg.neutral_dens_hi = pl_avrg.neutral_dens + stdev
    pl_avrg.neutral_dens_lo = pl_avrg.neutral_dens - stdev

    el_temp = xr.concat(el_temp, "run_name").assign_coords({"run_name": runs})
    stdev = el_temp.std("run_name")
    pl_avrg.el_temp = el_temp.mean("run_name")
    pl_avrg.el_temp_hi = pl_avrg.el_temp + stdev
    pl_avrg.el_temp_lo = pl_avrg.el_temp - stdev

    ion_temp = xr.concat(ion_temp, "run_name").assign_coords({"run_name": runs})
    stdev = ion_temp.std("run_name")
    pl_avrg.ion_temp = ion_temp.mean("run_name")
    pl_avrg.ion_temp_hi = pl_avrg.ion_temp + stdev
    pl_avrg.ion_temp_lo = pl_avrg.ion_temp - stdev

    return pl


def apply_limits(
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
                value = xr.where((error / value) >= err_lim[0], value, np.nan)
            if np.isfinite(err_lim[1]):
                print(err_lim[1])
                value = xr.where((error / value) <= err_lim[1], value, np.nan)

        data[diagnostic][q].values = value.values

    return data
