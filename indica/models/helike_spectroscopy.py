from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.datatypes import ELEMENTS
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers import ADASReader
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.readers.marchuk import MARCHUKReader

# TODO: add Marchuk PECs to repo or to .indica/ (more easily available to others)

MARCHUK = "/home/marco.sertoli/python/Indica/hda/Marchuk_Argon_PEC.pkl"
ADF15 = {
    "w": {
        "element": "ar",
        "file": ("16", "llu", "transport"),
        "charge": 16,
        "transition": "(1)1(1.0)-(1)0(0.0)",
        "wavelength": 4.0,
    }
}


class Helike_spectroscopy(DiagnosticModel):
    """
    Data and methods to model XRCS spectrometer measurements

    TODO: calibration and Etendue to be correctly included
    """

    transform: LineOfSightTransform

    def __init__(
        self,
        name: str,
        instrument_method="get_helike_spectroscopy",
        etendue: float = 1.0,
        calibration: float = 1.0e-18,
        int_cal: float = 1.3e-27,
        marchuk: bool = True,
        adf15: dict = None,
        extrapolate: str = None,
        full_run: bool = False,
        element: str = "ar",
        window_len: int = 1030,
        window_lim: list = [0.394, 0.401],
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            String identifier for the spectrometer
        fract_abu
            dictionary of fractional abundance objects FractionalAbundance to calculate ionisation balance
        marchuk
            Use Marchuk PECs instead of ADAS adf15 files
        adf15
            ADAS PEC file identifier dictionary
        extrapolate
            Go beyond validity limit of Machuk's data

        Returns
        -------

        """
        window = np.linspace(window_lim[0], window_lim[1], window_len)
        self.window = DataArray(window, coords=[("wavelength", window)])
        self.name = name
        self.instrument_method = instrument_method
        self.marchuk = marchuk

        self.element = element
        self.ion_charge = ELEMENTS[element][0] - 2  # He-like
        self.ion_mass = ELEMENTS[element][1]

        self.etendue = etendue
        self.calibration = calibration
        self.int_cal = int_cal  # TODO: absolute calibration? use only this or above
        self.full_run = full_run

        if self.marchuk:
            marchuck_reader = MARCHUKReader()
            self.pec_database = marchuck_reader.pec_database
            self.pec = marchuck_reader.pec_lines
        else:
            self.pec = self._set_adas_pecs(adf15=adf15, extrapolate=extrapolate)

        self.emission = {}
        self.emission_los = {}
        self.los_integral_intensity = {}
        self.measured_intensity = {}
        self.measured_Te = {}
        self.measured_Ti = {}
        self.pos = {}
        self.err_in = {}
        self.err_out = {}

        self.Te = None
        self.Ne = None
        self.Nimp = None
        self.Fz = None
        self.Nh = None
        self.t = None

    def _set_adas_pecs(self, adf15: dict = None, extrapolate: str = None):
        """
        Read ADAS adf15 data

        Parameters
        ----------
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)
        extrapolate
            Go beyond validity limit of machuk's data
        """
        self.adasreader = ADASReader()
        if adf15 is None:
            self.adf15 = ADF15

        pec = deepcopy(adf15)
        for line in adf15.keys():
            element = adf15[line]["element"]
            transition = adf15[line]["transition"]
            wavelength = adf15[line]["wavelength"]

            charge, filetype, year = adf15[line]["file"]
            adf15_data = self.adasreader.get_adf15(element, charge, filetype, year=year)
            # TODO: add the element layer to the pec dictionary (as for fract_abu)
            pec[line]["emiss_coeff"] = select_transition(
                adf15_data, transition, wavelength
            )

        if not self.full_run:
            for line in pec:
                pec[line]["emiss_coeff"] = (
                    pec[line]["emiss_coeff"]
                    .sel(electron_density=4.0e19, method="nearest")
                    .drop("electron_density")
                )

        self.adf15 = adf15
        self.pec = pec

    def _set_marchuk_pecs(self, extrapolate: str = None):
        """
        Read marchuk PEC data

        Parameters
        ----------
        extrapolate
            Go beyond validity limit of machuk's data
        """

        adf15, adf15_data = get_marchuk(extrapolate=extrapolate)
        pec = deepcopy(adf15)
        for line in adf15.keys():
            # TODO: add the element layer to the pec dictionary (as for fract_abu)
            element = adf15[line]["element"]
            pec[line]["emiss_coeff"] = adf15_data[line]

        if not self.full_run:
            for line in pec:
                pec[line]["emiss_coeff"] = (
                    pec[line]["emiss_coeff"]
                    .sel(electron_density=4.0e19, method="nearest")
                    .drop("electron_density")
                )

        self.adf15 = adf15
        self.pec = pec

    def _calculate_emission(self):
        """
        Calculate emission of all spectral lines included in the model

        Parameters
        ----------
        Te
            Electron temperature
        Ne
            Electron density
        Nimp
            Total impurity densities as defined in plasma.py
        fractional_abundance
            Fractional abundance dictionary of DataArrays of each element to be included
        Nh
            Neutral (thermal) hydrogen density
        t
            Time (s) for remapping on equilibrium reconstruction
        Returns
        -------

        """

        emission = {}

        for line, pec in self.pec.items():
            elem, charge, wavelength = pec["element"], pec["charge"], pec["wavelength"]
            coords = pec["emiss_coeff"].coords

            # Sum contributions from all transition types
            _emission = []
            if "index" in coords or "type" in coords:
                for pec_type in coords["type"]:
                    _pec = interp_pec(
                        select_type(pec["emiss_coeff"], type=pec_type), self.Ne, self.Te
                    )
                    mult = transition_rules(
                        pec_type,
                        self.Fz[elem],
                        charge,
                        self.Ne,
                        self.Nh,
                        self.Nimp.sel(element=elem),
                    )
                    _emission.append(_pec * mult)
            else:
                _pec = interp_pec(pec["emiss_coeff"], self.Ne, self.Te)
                _emission.append(
                    _pec
                    * self.Fz[elem].sel(ion_charges=charge)
                    * self.Ne
                    * self.Nimp.sel(element=elem)
                )

            _emission = xr.concat(_emission, "type").sum("type")
            # TODO: convert all wavelengths when reading PECs to nm as per convention at TE!
            ev_wavelength = ph.nm_eV_conversion(nm=wavelength / 10.0)
            emission[line] = xr.where(_emission >= 0, _emission, 0) * ev_wavelength

        if "k" in emission.keys() and "w" in emission.keys():
            emission["kw"] = emission["k"] * emission["w"]
        if "n3" in emission.keys() and "w" in emission.keys():
            emission["n3w"] = emission["n3"] * emission["w"]
        if (
            "n3" in emission.keys()
            and "n345" in emission.keys()
            and "w" in emission.keys()
        ):
            emission["tot"] = emission["n3"] + emission["n345"] + emission["w"]
            emission["n3tot"] = emission["n3"] * emission["n345"] * emission["w"]

        self.emission = emission

        return emission

    def _calculate_los_integral(self, calc_rho=False):
        for line in self.emission.keys():
            self.measured_intensity[line] = self.transform.integrate_on_los(
                self.emission[line], t=self.emission[line].t, calc_rho=calc_rho,
            )
            self.emission_los[line] = self.transform.along_los
            (
                _,
                self.pos[line],
                self.err_in[line],
                self.err_out[line],
            ) = self._moment_analysis(line)

        if self.calc_spectra:
            self.measured_spectra = self.transform.integrate_on_los(
                self.spectra["total"], t=self.spectra["total"].t, calc_rho=calc_rho,
            )

        self.t = self.measured_intensity[line].t

    def _calculate_temperatures(self):
        x1 = self.transform.x1
        x1_name = self.transform.x1_name

        for quant in self.quantities:
            datatype = self.quantities[quant]
            if datatype == ("temperature", "ions"):
                line = str(quant.split("_")[1])
                (
                    Ti_tmp,
                    self.pos[line],
                    self.err_in[line],
                    self.err_out[line],
                ) = self._moment_analysis(line, profile_1d=self.Ti)
                self.measured_Ti[line] = xr.concat(Ti_tmp, x1_name).assign_coords(
                    {x1_name: x1}
                )
            elif datatype == ("temperature", "electrons"):
                line = str(quant.split("_")[1])
                (
                    Te_tmp,
                    self.pos[line],
                    self.err_in[line],
                    self.err_out[line],
                ) = self._moment_analysis(line, profile_1d=self.Te)
                self.measured_Te[line] = xr.concat(Te_tmp, x1_name).assign_coords(
                    {x1_name: x1}
                )

    def _moment_analysis(
        self, line: str, profile_1d: DataArray = None, half_los: bool = True,
    ):
        """
        Perform moment analysis using a specific line emission as distribution function
        and calculating the position of emissivity, and expected measured value if
        measured profile (profile_1d) is given

        Parameters
        -------
        line
            identifier of measured spectral line
        t
            time (s)
        profile_1d
            1D profile on which to perform the moment analysis
        half_los
            set to True if only half of the LOS to be used for the analysis
        """

        element = self.emission[line].element.values
        result: list = []
        pos: list = []
        err_in: list = []
        err_out: list = []
        if np.size(self.t) ==1:
            times = np.array([self.t])
        else:
            times = self.t

        for chan in self.transform.x1:
            _value = None
            _result = []
            _pos, _err_in, _err_out = [], [], []
            for t in times:
                if "t" in self.emission_los[line][chan].dims:
                    distribution_function = self.emission_los[line][chan].sel(t=t).values
                else:
                    distribution_function = self.emission_los[line][chan].values

                if "t" in self.transform.rho[chan].dims:
                    rho_los = self.transform.rho[chan].sel(t=t, method="nearest").values
                else:
                    rho_los = self.transform.rho[chan].values
                if half_los:
                    rho_ind = slice(0, np.argmin(rho_los) + 1)
                else:
                    rho_ind = slice(0, len(rho_los))
                rho_los = rho_los[rho_ind]
                rho_min = np.min(rho_los)

                dfunction = distribution_function[rho_ind]
                indices = np.arange(0, len(dfunction))
                avrg, dlo, dhi, ind_in, ind_out = ph.calc_moments(
                    dfunction, indices, simmetry=False
                )

                # Position of emissivity
                pos_tmp = rho_los[int(avrg)]
                err_in_tmp = np.abs(rho_los[int(avrg)] - rho_los[int(avrg - dlo)])
                if pos_tmp <= rho_min:
                    pos_tmp = rho_min
                    err_in_tmp = 0.0
                if (err_in_tmp > pos_tmp) and (err_in_tmp > (pos_tmp - rho_min)):
                    err_in_tmp = pos_tmp - rho_min
                err_out_tmp = np.abs(rho_los[int(avrg)] - rho_los[int(avrg + dhi)])
                _pos.append(pos_tmp)
                _err_in.append(err_in_tmp)
                _err_out.append(err_out_tmp)

                # Moment analysis of input 1D profile
                if profile_1d is not None:
                    profile_interp = profile_1d.interp(rho_poloidal=rho_los)
                    if "element" in profile_interp.dims:
                        profile_interp = profile_interp.sel(element=element)
                    if "t" in profile_1d.dims:
                        profile_interp = profile_interp.sel(t=t, method="nearest")
                    profile_interp = profile_interp.values
                    _value, _, _, _, _ = ph.calc_moments(
                        dfunction,
                        profile_interp,
                        ind_in=ind_in,
                        ind_out=ind_out,
                        simmetry=False,
                    )
                _result.append(_value)

            result.append(DataArray(np.array(_result), coords=[("t", times)]))
            pos.append(DataArray(np.array(_pos), coords=[("t", times)]))
            err_in.append(DataArray(np.array(_err_in), coords=[("t", times)]))
            err_out.append(DataArray(np.array(_err_out), coords=[("t", times)]))

        result = xr.concat(result, self.transform.x1_name).assign_coords(
            {self.transform.x1_name: self.transform.x1}
        )
        pos = xr.concat(pos, self.transform.x1_name).assign_coords(
            {self.transform.x1_name: self.transform.x1}
        )
        err_in = xr.concat(err_in, self.transform.x1_name).assign_coords(
            {self.transform.x1_name: self.transform.x1}
        )
        err_out = xr.concat(err_out, self.transform.x1_name).assign_coords(
            {self.transform.x1_name: self.transform.x1}
        )

        return result, pos, err_in, err_out

    def _make_intensity(self,):
        """
        Uses the intensity recipes to get intensity from
        Te/ne/f_abundance/hydrogen_density/calibration_factor and atomic data.
        Returns DataArrays of emission type with co-ordinates of line label and
        spatial co-ordinate
        TODO: selection criteri, element and charge have to follow atomic data (see _calculate_emission)
        """
        calibration = self.int_cal
        database = self.pec_database
        Te = self.Te
        electron_density = self.Ne
        fz = self.Fz["ar"]
        argon_density = self.Nimp.sel(element="ar")
        hydrogen_density = self.Nh

        intensity = {}
        for key, value in database.items():
            _Te = Te

            if value.type == "excit" or value.type == "diel":
                _ion_charge = self.ion_charge
                _density = electron_density
            elif value.type == "recom":
                if self.ion_charge == 16:  # TODO: do we still not have newest data?
                    _Te = Te.where(Te < 4000, 4000)
                _ion_charge = self.ion_charge + 1
                _density = electron_density
            elif value.type == "cxr":
                _ion_charge = self.ion_charge + 1
                _density = hydrogen_density
                if self.ion_charge == 16:
                    _Te = Te.where(Te < 4000, 4000)
            elif value.type == "ise" or value.type == "isi" or value.type == "li_diel":
                _ion_charge = self.ion_charge - 1
                _density = electron_density
            else:
                print(f"Emission type {value.type} not recognised")

            intensity[key] = (
                value.interp(electron_temperature=_Te)
                * fz.sel(ion_charges=_ion_charge)
                * argon_density
                * _density
                * calibration
            )

        return intensity

    def _make_spectra(
        self, window: LabeledArray = None,
    ):

        # Add convolution of broadening
        # -> G(x, mu1, sig1) * G(x, mu2, sig2) = G(x, mu1+mu2, sig1**2 + sig2**2)
        # instrument function / photon noise / photon -> counts
        # Background Noise
        """
        TODO: Doppler Shift / Add make_intensity to this method / Background moved to plot
        Parameters
        ----------
        window

        Returns
        -------

        """
        if window is None:
            window = self.window

        intensity = self.intensity
        spectra = {}
        spectra["total"] = 0.0  # TODO: incorrect assignment
        ion_mass = self.ion_mass
        for key, value in intensity.items():
            line_name = value.line_name
            # TODO: substitute with value.element
            element = self.element
            _spectra = []
            if len(np.shape(self.t)) == 0:
                times = np.array([self.t])
            else:
                times = self.t

            for t in times:
                Ti = self.Ti.sel(element=element)
                if "t" in self.Ti.dims:
                    Ti = Ti.sel(t=t)

                if "t" in self.Ti.dims:
                    integral = value.sel(t=t)
                else:
                    integral = value

                center = integral.wavelength
                ion_temp = Ti.expand_dims(dict(line_name=line_name)).transpose()
                x = window.expand_dims(dict(line_name=line_name)).transpose()
                y = doppler_broaden(x, integral, center, ion_mass, ion_temp)
                # TODO: include also doppler shift
                # y = doppler_shift(x, integral, center, ion_mass, ion_temp)
                _spectra.append(y)
            spectra[key] = xr.concat(_spectra, "t")
            spectra["total"] = spectra["total"] + spectra[key].sum(["line_name"])
        return spectra

    def _plot_spectrum(self, spectra: dict, background: float = 0.0):

        plt.figure()
        spectra["background"] = self.window * 0 + background

        avoid = ["total", "background"]
        for key, value in spectra.items():
            if not any([x in key for x in avoid]):
                plt.plot(
                    self.window,
                    value.sum(["rho_poloidal"]).sum("line_name")
                    + spectra["background"],
                    label=key,
                )

        plt.plot(
            self.window,
            spectra["total"].sum(["rho_poloidal"]) + spectra["background"],
            "k*",
            label="Total",
        )
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (AU)")
        plt.legend()
        plt.show(block=True)
        return

    def _build_bckc_dictionary(self):
        self.bckc = {}
        for quant in self.quantities:
            datatype = self.quantities[quant]
            if datatype[0] == ("intensity"):
                line = str(quant.split("_")[1])
                quantity = f"int_{line}"
                self.bckc[quantity] = self.measured_intensity[line]
                self.bckc[quantity].attrs["emiss"] = self.emission[line]
            elif datatype == ("temperature", "electrons"):
                line = str(quant.split("_")[1])
                quantity = f"te_{line}"
                self.bckc[quantity] = self.measured_Te[line]
                self.bckc[quantity].attrs["emiss"] = self.emission[line]
            elif datatype == ("temperature", "ions"):
                line = str(quant.split("_")[1])
                quantity = f"ti_{line}"
                self.bckc[quantity] = self.measured_Ti[line]
                self.bckc[quantity].attrs["emiss"] = self.emission[line]
            elif datatype == ("spectra", "passive"):
                if self.calc_spectra:
                    quantity = quant
                    self.bckc["spectra"] = self.measured_spectra
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            self.bckc[quantity].attrs["datatype"] = datatype
            if quant != "spectra":
                self.bckc[quantity].attrs["pos"] = {
                    "value": self.pos[line],
                    "err_in": self.err_in[line],
                    "err_out": self.err_out[line],
                }

        self.bckc["int_k/int_w"] = self.bckc["int_k"] / self.bckc["int_w"]
        self.bckc["int_n3/int_w"] = self.bckc["int_n3"] / self.bckc["int_w"]
        self.bckc["int_n3/int_tot"] = self.bckc["int_n3"] / self.bckc["int_tot"]

    def __call__(
        self,
        Te: DataArray = None,
        Ti: DataArray = None,
        Ne: DataArray = None,
        Nimp: DataArray = None,
        Fz: dict = None,
        Nh: DataArray = None,
        t: LabeledArray = None,
        calc_spectra=False,
        calc_rho: bool = False,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Te - electron temperature (eV)
        Ti - ion temperature (eV)
        Ne - electron density (m**-3)
        Nimp - impurity density (m**-3)
        fractional_abundance - fractional abundance
        Nh - neutral density (m**-3)
        t - time (s)

        Returns
        -------

        """
        self.calc_spectra = calc_spectra
        if self.plasma is not None:
            if t is None:
                t = self.plasma.t
            Te = self.plasma.electron_temperature.interp(t=t)
            Ne = self.plasma.electron_density.interp(t=t)
            Nh = self.plasma.neutral_density.interp(t=t)
            Fz = {}
            _Fz = self.plasma.fz
            for elem in _Fz.keys():
                Fz[elem] = _Fz[elem].interp(t=t)

            Ti = self.plasma.ion_temperature.interp(t=t)
            Nimp = self.plasma.impurity_density.interp(t=t)
        else:
            if (
                Ne is None
                or Te is None
                or Nh is None
                or Fz is None
                or Ti is None
                or Nimp is None
            ):
                raise ValueError("Give inputs of assign plasma class!")

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nh = Nh
        self.Fz = Fz
        self.Ti = Ti
        self.Nimp = Nimp
        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]
        self.transform.check_rho(t=t)

        # TODO: check that inputs have compatible dimensions/coordinates

        # Calculate emission on natural coordinates of input profiles
        self._calculate_emission()

        # Make spectra
        if calc_spectra:
            self.intensity = self._make_intensity()
            self.spectra = self._make_spectra()

        # Integrate emission along the LOS
        self._calculate_los_integral(calc_rho=calc_rho,)

        # Estimate temperatures from moment analysis
        self._calculate_temperatures()

        # Build back-calculated dictionary to compare with experimental data
        self._build_bckc_dictionary()

        return self.bckc


def doppler_broaden(x, integral, center, ion_mass, ion_temp):
    sigma = (
        np.sqrt(
            constants.e
            / (ion_mass * constants.proton_mass * constants.c ** 2)
            * ion_temp
        )
        * center
    )
    gaussian_broadened = gaussian(x, integral, center, sigma,)
    return gaussian_broadened


def gaussian(x, integral, center, sigma):
    return (
        integral
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    )


def interp_pec(pec, Ne, Te):
    if "electron_density" in pec.coords:
        pec_interp = pec.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
    else:
        pec_interp = pec.interp(electron_temperature=Te, method="cubic",)

    return pec_interp


def select_type(pec, type="excit"):
    if "index" in pec.dims:
        pec = pec.swap_dims({"index": "type"})
    return pec.sel(type=type)


def transition_rules(transition_type, fz, charge, Ne, Nh, Nimp):
    if transition_type == "recom":
        mult = fz.sel(ion_charges=charge + 1) * Ne * Nimp
    elif transition_type == "cxr":
        mult = fz.sel(ion_charges=charge + 1) * Nh * Nimp
    else:
        mult = fz.sel(ion_charges=charge) * Ne * Nimp

    return mult


def select_transition(adf15_data, transition: str, wavelength: float):

    """
    Given adf15 data in input, select pec for specified spectral line, given
    transition and wavelength identifiers

    Parameters
    ----------
    adf15_data
        adf15 data
    transition
        transition for spectral line as specified in adf15
    wavelength
        wavelength of spectral line as specified in adf15

    Returns
    -------
    pec data of desired spectral line

    """

    pec = deepcopy(adf15_data)

    dim = [
        d for d in pec.dims if d != "electron_temperature" and d != "electron_density"
    ][0]
    if dim != "transition":
        pec = pec.swap_dims({dim: "transition"})
    pec = pec.sel(transition=transition, drop=True)

    if len(np.unique(pec.coords["wavelength"].values)) > 1:
        pec = pec.swap_dims({"transition": "wavelength"})
        try:
            pec = pec.sel(wavelength=wavelength, drop=True)
        except KeyError:
            pec = pec.sel(wavelength=wavelength, method="nearest", drop=True)

    return pec


def example_run(plasma=None, plot=False, calc_spectra=False):

    # TODO: solve issue of LOS sometimes crossing bad EFIT reconstruction outside of the separatrix
    if plasma is None:
        plasma = example_plasma()

    # Create new diagnostic
    diagnostic_name = "xrcs"
    nchannels = 3
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.43, -0.43, nchannels)
    los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
    origin = los_start
    direction = los_end - los_start

    transform = LineOfSightTransform(
        origin[:, 0],
        origin[:, 1],
        origin[:, 2],
        direction[:, 0],
        direction[:, 1],
        direction[:, 2],
        name=diagnostic_name,
        machine_dimensions=plasma.machine_dimensions,
        passes=1,
    )
    model = Helike_spectroscopy(diagnostic_name,)
    model.set_transform(transform)
    model.set_flux_transform(plasma.flux_transform)
    model.set_plasma(plasma)

    bckc = model(calc_spectra=calc_spectra)

    channels = model.transform.x1
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))

    # Plot spectra
    if plot:
        it = int(len(plasma.t) / 2)
        tplot = plasma.t[it]
        plt.figure()
        for chan in channels:
            if (chan % 2) == 0:
                bckc["spectra"].sel(
                    channel=chan, t=plasma.t.mean(), method="nearest"
                ).plot(label=f"CH{chan}", color=cols[chan])
        plt.xlabel("Time (s)")
        plt.ylabel("Te and Ti from moment analysis (eV)")
        plt.legend()

        plt.figure()
        plasma.equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
            levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        )
        for chan in channels:
            plt.plot(
                model.transform.R[chan],
                model.transform.z[chan],
                linewidth=3,
                color=cols[chan],
                alpha=0.7,
                label=f"CH{chan}",
            )

        plt.xlim(0, 1.0)
        plt.ylim(-0.6, 0.6)
        plt.axis("scaled")
        plt.legend()

        # Plot LOS mapping on equilibrium
        plt.figure()
        for chan in channels:
            model.transform.rho[chan].sel(t=tplot, method="nearest").plot(
                color=cols[chan], label=f"CH{chan}",
            )
        plt.xlabel("Path along the LOS")
        plt.ylabel("Rho-poloidal")
        plt.legend()

        # Plot back-calculated values
        plt.figure()
        for chan in channels:
            bckc["int_w"].sel(channel=chan).plot(label=f"CH{chan}", color=cols[chan])
        plt.xlabel("Time (s)")
        plt.ylabel("w-line intensity (W/m^2)")
        plt.legend()

        plt.figure()
        for chan in channels:
            bckc["ti_w"].sel(channel=chan).plot(
                label=f"CH{chan} ti_w", color=cols[chan]
            )
            bckc["te_kw"].sel(channel=chan).plot(
                label=f"CH{chan} te_kw", color=cols[chan], linestyle="dashed"
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Te and Ti from moment analysis (eV)")
        plt.legend()

        # Plot the temperatures profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        elem = model.Ti.element[0].values
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.Ti.rho_poloidal,
                model.Ti.sel(t=t, element=elem),
                color=cols_time[i],
            )
            plt.plot(
                model.Te.rho_poloidal,
                model.Te.sel(t=t),
                color=cols_time[i],
                linestyle="dashed",
            )
        plt.plot(
            model.Ti.rho_poloidal,
            model.Ti.sel(t=t, element=elem),
            color=cols_time[i],
            label=f"Ti",
        )
        plt.plot(
            model.Te.rho_poloidal,
            model.Te.sel(t=t),
            color=cols_time[i],
            label=f"Te",
            linestyle="dashed",
        )
        plt.xlabel("rho")
        plt.ylabel("Ti and Te profiles (eV)")
        plt.legend()

        # Plot the emission profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        plt.figure()
        for i, t in enumerate(plasma.t.values):
            plt.plot(
                model.emission["w"].rho_poloidal,
                model.emission["w"].sel(t=t),
                color=cols_time[i],
                label=f"t={t:1.2f} s",
            )
        plt.xlabel("rho")
        plt.ylabel("w-line local radiated power (W/m^3)")
        plt.legend()

    return plasma, model, bckc
