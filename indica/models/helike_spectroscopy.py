from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
from scipy import constants
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.datatypes import ELEMENTS
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_run as example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers import ADASReader
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.readers.marchuk import MARCHUKReader

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

    def __init__(
        self,
        name: str,
        instrument_method="get_helike_spectroscopy",
        etendue: float = 1.0,
        calibration: float = 5.0e-18,
        marchuk: bool = True,
        full_run: bool = False,
        element: str = "ar",
        window_len: int = 1030,
        window_lim: list = [0.394, 0.401],
        window_masks: list = [],
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            String identifier for the spectrometer
        fract_abu
            dictionary of fractional abundance objects
        marchuk
            Use Marchuk PECs instead of ADAS adf15 files

        Returns
        -------

        """
        self.name = name
        self.instrument_method = instrument_method
        self.marchuk = marchuk

        self.element: str = element
        z_elem, a_elem, name_elem = ELEMENTS[element]
        self.ion_charge: int = z_elem - 2  # He-like
        self.ion_mass: float = a_elem

        self.etendue = etendue
        self.calibration = calibration
        self.full_run = full_run
        self.adf15 = ADF15
        self.pecs: dict

        if self.marchuk:
            marchuck_reader = MARCHUKReader(
                element=self.element,
                charge=self.ion_charge,
            )
            self.pecs = marchuck_reader.pecs
        else:
            self.pecs = self._set_adas_pecs()

        self.window_masks = window_masks
        window = np.linspace(window_lim[0], window_lim[1], window_len)
        mask = np.zeros(shape=window.shape)
        if window_masks:
            for mslice in window_masks:
                mask[(window > mslice.start) & (window < mslice.stop)] = 1
        else:
            mask[:] = 1

        self.window = DataArray(mask, coords=[("wavelength", window)])
        self.mask = self.window.interp(wavelength=self.pecs["emiss_coeff"].wavelength)
        self.pecs["emiss_coeff"] = self.pecs["emiss_coeff"].where(
            self.mask,
            drop=True,
        )
        self.pecs["emiss_coeff"] = self.pecs["emiss_coeff"].where(
            (self.pecs["emiss_coeff"].wavelength < self.window.wavelength.max())
            & (self.pecs["emiss_coeff"].wavelength > self.window.wavelength.min()),
            drop=True,
        )

        # wavelength indexes used for line ratio moment analysis
        self.line_ranges = {
            "w": slice(0.39489, 0.39494),
            "n3": slice(0.39543, 0.39574),
            "n345": slice(0.39496, 0.39574),
            "qra": slice(0.39810, 0.39865),
            "k": slice(0.39890, 0.39910),
            "z": slice(0.39935, 0.39950),
        }

        self.line_emission: dict = {}
        self.emission_los: dict = {}
        self.los_integral_intensity: dict = {}
        self.measured_intensity: dict = {}
        self.measured_Te: dict = {}
        self.measured_Ti: dict = {}
        self.pos: dict = {}
        self.err_in: dict = {}
        self.err_out: dict = {}

        self.Te: DataArray
        self.Ne: DataArray
        self.Nimp: DataArray
        self.Fz: dict
        self.Nh: DataArray

    def _set_adas_pecs(self):
        """
        Read ADAS adf15 data
        """
        self.adasreader = ADASReader()

        adf15 = self.adf15
        pec: dict = deepcopy(adf15)
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
                    .drop_vars("electron_density")
                )

        self.pec = pec

    def _transition_matrix(self, element="ar", charge=16):
        """vectorisation of the transition matrix used to convert
        Helike PECs to emissivity"""
        # fmt: off
        _Nimp = self.Nimp.sel(element=element, )
        _Fz = self.Fz[element]
        transition_matrix = xr.concat([
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge, ),
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge, ),
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charges=charge + 1, ),
            self.Nh * _Nimp * _Fz.sel(ion_charges=charge + 1, ),
        ], "type").assign_coords(
            type=["excit", "diel", "li_diel", "ise", "isi", "recom", "cxr", ])
        # fmt: on
        return transition_matrix

    def _calculate_line_emission(self, line_labels=["w", "k"]):
        """
        Calculate emission of all spectral lines in line_label list

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
        elem, charge = self.pecs["element"], self.pecs["charge"]

        mult = self._transition_matrix(element=elem, charge=charge)
        line_emission = {}

        # filter out unused sections of the database based on wavelength.
        # Only for the first time called...
        if not hasattr(self, "filtered_pecs"):
            _pecs = self.pecs["emiss_coeff"]
            _bool_arrays = [
                (_pecs.wavelength < self.line_ranges[line_name].stop)
                & (_pecs.wavelength > self.line_ranges[line_name].start)
                for line_name in line_labels
            ]
            _temp_array = _bool_arrays[0]
            for bool_array in _bool_arrays:
                _temp_array = _temp_array | bool_array
            self.filtered_pecs = _pecs.isel(line_name=_temp_array)

        # cubic method fails due to how scipy handles NaNs
        _pecs = self.filtered_pecs.interp(
            electron_temperature=self.Te,
        )
        for line_name in line_labels:
            _line_emission = (
                _pecs.isel(
                    line_name=(_pecs.wavelength < self.line_ranges[line_name].stop)
                    & (_pecs.wavelength > self.line_ranges[line_name].start)
                )
                * mult
            )
            _line_emission = _line_emission.sum(["type", "line_name"])
            # TODO: convert PEC wavelengths to nm as per convention at TE!
            ev_wavelength = ph.nm_eV_conversion(nm=self.line_ranges[line_name].start)
            line_emission[line_name] = (
                xr.where(_line_emission >= 0, _line_emission, 0) * ev_wavelength
            ) * self.calibration

        if "k" in line_emission.keys() and "w" in line_emission.keys():
            line_emission["kw"] = line_emission["k"] * line_emission["w"]
        if "n3" in line_emission.keys() and "w" in line_emission.keys():
            line_emission["n3w"] = line_emission["n345"] * line_emission["w"]
        if (
            "n3" in line_emission.keys()
            and "n345" in line_emission.keys()
            and "w" in line_emission.keys()
        ):
            line_emission["tot"] = line_emission["n345"] + line_emission["w"]
            line_emission["n3tot"] = line_emission["n345"] * line_emission["w"]
        self.line_emission = line_emission

    def _calculate_los_integral(self, calc_rho=False):
        for line in self.line_emission.keys():
            self.measured_intensity[line] = self.los_transform.integrate_on_los(
                self.line_emission[line],
                t=self.line_emission[line].t,
                calc_rho=calc_rho,
            )
            self.emission_los[line] = self.los_transform.along_los

        for line in self.line_emission.keys():
            (
                _,
                self.pos[line],
                self.err_in[line],
                self.err_out[line],
            ) = self._moment_analysis(line)

        if self.calc_spectra:
            self.measured_spectra = self.los_transform.integrate_on_los(
                self.spectra,
                t=self.spectra.t,
                calc_rho=calc_rho,
            )
            # TODO: LOS integral removes NaNs so manually add them back (find better solution)
            self.measured_spectra[self.measured_spectra == 0] = np.nan

    def _calculate_temperatures(self):
        x1 = self.los_transform.x1
        x1_name = self.los_transform.x1_name

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
                if x1.__len__() == 1:
                    self.measured_Ti[line] = Ti_tmp
                else:
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
                if x1.__len__() == 1:
                    self.measured_Te[line] = Te_tmp
                else:
                    self.measured_Te[line] = xr.concat(Te_tmp, x1_name).assign_coords(
                        {x1_name: x1}
                    )

    def _moment_analysis(
        self,
        line: str,
        profile_1d: DataArray = None,
        half_los: bool = True,
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

        element = self.line_emission[line].element.values
        result_list: list = []
        pos_list: list = []
        err_in_list: list = []
        err_out_list: list = []

        if len(np.shape(self.t)) == 0:
            times = np.array(
                [
                    self.t,
                ]
            )
        else:
            times = self.t

        for chan in self.los_transform.x1:
            _value = None
            _result = []
            _pos, _err_in, _err_out = [], [], []
            for t in times:
                if "t" in self.emission_los[line][chan].dims:
                    distribution_function = (
                        self.emission_los[line][chan].sel(t=t).values
                    )
                else:
                    distribution_function = self.emission_los[line][chan].values

                if "t" in self.los_transform.rho[chan].dims:
                    rho_los = (
                        self.los_transform.rho[chan].sel(t=t, method="nearest").values
                    )
                else:
                    rho_los = self.los_transform.rho[chan].values

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
                pos_tmp, err_in_tmp, err_out_tmp = np.nan, np.nan, np.nan
                if np.isfinite(avrg) and np.isfinite(dlo) and np.isfinite(dhi):
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

            # TODO: Clean up the handling of "t" and "channel" dims
            result_list.append(DataArray(np.array(_result), coords=[("t", times)]))
            pos_list.append(DataArray(np.array(_pos), coords=[("t", times)]))
            err_in_list.append(DataArray(np.array(_err_in), coords=[("t", times)]))
            err_out_list.append(DataArray(np.array(_err_out), coords=[("t", times)]))

        result = xr.concat(result_list, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        pos = xr.concat(pos_list, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        err_in = xr.concat(err_in_list, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        err_out = xr.concat(err_out_list, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        # Return without channel / t as dims
        if self.los_transform.x1.__len__() == 1:
            result = result.sel(channel=self.los_transform.x1[0])
            pos = pos.sel(channel=self.los_transform.x1[0])
            err_in = err_in.sel(channel=self.los_transform.x1[0])
            err_out = err_out.sel(channel=self.los_transform.x1[0])

        if type(self.t) == float:
            result = result.sel(t=self.t)
            pos = pos.sel(t=self.t)
            err_in = err_in.sel(t=self.t)
            err_out = err_out.sel(t=self.t)

        return result, pos, err_in, err_out

    def _calculate_intensity(
        self,
    ):
        """
        Returns DataArrays of emission type with co-ordinates of line label and
        spatial co-ordinate
        """
        elem, charge = self.pecs["element"], self.pecs["charge"]
        mult = self._transition_matrix(element=elem, charge=charge)
        _pecs = self.pecs["emiss_coeff"]

        # Swapping to dataset and then dropping line_names with NaNs is much faster
        _pecs_ds = _pecs.to_dataset("type")
        temp = [
            _pecs_ds[type]
            .dropna("line_name", how="all")
            .interp(electron_temperature=self.Te, assume_sorted=True)
            for type in _pecs_ds.data_vars.keys()
        ]
        _intensity = xr.merge(temp).to_array("type")
        intensity = (_intensity * mult * self.calibration).sum("type")
        return intensity

    def _make_spectra(
        self,
    ):
        """
        TODO: Doppler Shift / Add convolution of broadening
        -> G(x, mu1, sig1) * G(x, mu2, sig2) = G(x, mu1+mu2, sig1**2 + sig2**2)
        instrument function / photon noise / photon -> counts
        Background Noise

        """
        element = self.pecs["element"]
        _spectra = doppler_broaden(
            self.window[self.window > 0].wavelength,
            self.intensity,
            self.intensity.wavelength,
            self.ion_mass,
            self.Ti.sel(element=element),
        )
        _spectra = _spectra.sum("line_name")
        # extend spectra to same coords as self.window.wavelength with NaNs
        # to maintain same shape as mds data
        if "t" in _spectra.dims:
            empty = xr.DataArray(
                np.nan,
                dims=_spectra.dims,
                coords=dict(
                    wavelength=self.window[self.window < 1].wavelength,
                    rho_poloidal=_spectra.rho_poloidal,
                    t=_spectra.t,
                ),
            )
        else:
            empty = xr.DataArray(
                np.nan,
                dims=_spectra.dims,
                coords=dict(
                    wavelength=self.window[self.window < 1].wavelength,
                    rho_poloidal=_spectra.rho_poloidal,
                ),
            )
        spectra = xr.concat([_spectra, empty], "wavelength")
        spectra = spectra.sortby("wavelength")
        return spectra

    def _build_bckc_dictionary(self):
        self.bckc = {}
        for quant in self.quantities:
            datatype = self.quantities[quant]
            if datatype[0] == ("intensity"):
                line = str(quant.split("_")[1])
                quant = f"int_{line}"
                self.bckc[quant] = self.measured_intensity[line]
                self.bckc[quant].attrs["emiss"] = self.line_emission[line]
            elif datatype == ("temperature", "electrons"):
                line = str(quant.split("_")[1])
                quant = f"te_{line}"
                self.bckc[quant] = self.measured_Te[line]
                self.bckc[quant].attrs["emiss"] = self.line_emission[line]
            elif datatype == ("temperature", "ions"):
                line = str(quant.split("_")[1])
                quant = f"ti_{line}"
                self.bckc[quant] = self.measured_Ti[line]
                self.bckc[quant].attrs["emiss"] = self.line_emission[line]
            elif datatype == ("spectra", "passive"):
                self.bckc["spectra"] = self.measured_spectra
                self.bckc["spectra"].attrs["error"] = np.sqrt(
                    self.measured_spectra
                )  # poisson noise
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            self.bckc[quant].attrs["datatype"] = datatype
            if quant != "spectra" and hasattr(self, "spectra"):
                self.bckc[quant].attrs["pos"] = {
                    "value": self.pos[line],
                    "err_in": self.err_in[line],
                    "err_out": self.err_out[line],
                }
        if "int_k" in self.bckc.keys() and "int_w" in self.bckc.keys():
            self.bckc["int_k/int_w"] = self.bckc["int_k"] / self.bckc["int_w"]
        if "int_n3" in self.bckc.keys() and "int_w" in self.bckc.keys():
            self.bckc["int_n3/int_w"] = self.bckc["int_n3"] / self.bckc["int_w"]
        if "int_n3" in self.bckc.keys() and "int_tot" in self.bckc.keys():
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
        calc_spectra=True,
        calc_rho: bool = False,
        minimum_lines: bool = False,
        moment_analysis: bool = False,
        **kwargs,
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
        self.calc_rho = calc_rho
        self.minimum_lines = minimum_lines
        self.moment_analysis = moment_analysis

        if self.moment_analysis and bool(self.window_masks):
            raise ValueError(
                "moment_analysis cannot be used when window_masks is not set to None"
            )

        if self.plasma is not None:
            if t is None:
                t = self.plasma.time_to_calculate
            Te = self.plasma.electron_temperature.interp(
                t=t,
            )
            Ne = self.plasma.electron_density.interp(
                t=t,
            )
            Nh = self.plasma.neutral_density.interp(
                t=t,
            )
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
                raise ValueError("Give inputs or assign plasma class!")

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nh = Nh
        self.Fz = Fz
        self.Ti = Ti
        self.Nimp = Nimp
        self.quantities: dict = AVAILABLE_QUANTITIES[self.instrument_method]

        # TODO: check that inputs have compatible dimensions/coordinates

        # At the moment due to how _build_bckc/_calculate_temperature work
        # quantities has to be altered depending on model settings used
        quant: dict = {}
        if moment_analysis:
            if minimum_lines:
                line_labels = ["w", "k"]
                names = ["int_w", "int_k", "te_kw", "ti_w"]
            else:
                line_labels = list(self.line_ranges.keys())
                names = [
                    "int_w",
                    "int_k",
                    "int_tot",
                    "int_n3",
                    "te_n3w",
                    "te_kw",
                    "ti_z",
                    "ti_w",
                ]
            quant = dict(quant, **{x: self.quantities[x] for x in names})

        if calc_spectra:
            quant = dict(quant, **{x: self.quantities[x] for x in ["spectra"]})
        self.quantities = quant

        # Calculate emission on natural coordinates of input profiles
        if moment_analysis:
            self._calculate_line_emission(line_labels=line_labels)

        # Make spectra
        if calc_spectra:
            self.intensity = self._calculate_intensity()
            self.spectra = self._make_spectra()

        self._calculate_los_integral(
            calc_rho=calc_rho,
        )
        if moment_analysis:
            self._calculate_temperatures()
        self._build_bckc_dictionary()

        return self.bckc


# fmt: off
def doppler_broaden(x, integral, center, ion_mass, ion_temp):
    mass = (ion_mass * constants.proton_mass * constants.c ** 2)
    sigma = np.sqrt(constants.e / mass * ion_temp) * center
    gaussian_broadened = gaussian(x, integral, center, sigma, )
    return gaussian_broadened


def gaussian(x, integral, center, sigma):
    return (integral / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)))
# fmt: on


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


def example_run(pulse: int = None, plasma=None, plot=False, **kwargs):
    # TODO: LOS sometimes crossing bad EFIT reconstruction
    if plasma is None:
        plasma = example_plasma(
            pulse=pulse, impurities=("ar",), impurity_concentration=(0.001,), n_rad=10
        )
        plasma.time_to_calculate = plasma.t[5]
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

    los_transform = LineOfSightTransform(
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
    los_transform.set_equilibrium(plasma.equilibrium)
    model = Helike_spectroscopy(
        diagnostic_name,
        window_masks=[],
    )
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    bckc = model(**kwargs)
    channels = model.los_transform.x1
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))

    # Plot spectra
    if plot:
        # tplot = plasma.time_to_calculate
        if "spectra" in bckc.keys():
            plt.figure()
            for chan in channels:
                _spec = bckc["spectra"]
                if "channel" in _spec.dims:
                    _spec = _spec.sel(channel=chan, method="nearest")
                if "t" in bckc["spectra"].dims:
                    _spec.sel(t=plasma.time_to_calculate.mean(), method="nearest").plot(
                        label=f"CH{chan}", color=cols[chan]
                    )
                else:
                    _spec.plot(label=f"CH{chan}", color=cols[chan])
            plt.xlabel("wavelength (nm)")
            plt.ylabel("spectra")
            plt.legend()

        # model.los_transform.plot_los(tplot,
        # plot_all=model.los_transform.x1.__len__() > 1)

        if "int_w" in bckc.keys() & "t" in bckc["int_w"].dims:
            plt.figure()
            for chan in channels:
                if "channel" in bckc["int_w"].dims:
                    bckc["int_w"].sel(channel=chan).plot(
                        label=f"CH{chan}", color=cols[chan]
                    )
                else:
                    bckc["int_w"].plot(label=f"CH{chan}", color=cols[chan])
            plt.xlabel("Time (s)")
            plt.ylabel("w-line intensity (W/m^2)")
            plt.legend()

        if "ti_w" in bckc.keys() & "te_kw" in bckc.keys():
            plt.figure()
            for chan in channels:
                if "channel" in bckc["ti_w"].dims:
                    bckc["ti_w"].sel(channel=chan).plot(
                        label=f"CH{chan} ti_w", color=cols[chan]
                    )
                    bckc["te_kw"].sel(channel=chan).plot(
                        label=f"CH{chan} te_kw", color=cols[chan], linestyle="dashed"
                    )
                else:
                    bckc["ti_w"].plot(label=f"CH{chan} ti_w", color=cols[chan])
                    bckc["te_kw"].plot(
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
                plasma.ion_temperature.rho_poloidal,
                plasma.ion_temperature.sel(t=t, element=elem),
                color=cols_time[i],
            )
            plt.plot(
                plasma.electron_temperature.rho_poloidal,
                plasma.electron_temperature.sel(t=t),
                color=cols_time[i],
                linestyle="dashed",
            )

        plt.xlabel("rho")
        plt.ylabel("Ti and Te profiles (eV)")

        # Plot the emission profiles
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(plasma.t), dtype=float))
        if "w" in model.line_emission.keys():
            plt.figure()
            if "t" in model.line_emission["w"].dims:
                for i, t in enumerate(plasma.t.values):
                    plt.plot(
                        model.line_emission["w"].rho_poloidal,
                        model.line_emission["w"].sel(t=t),
                        color=cols_time[i],
                        label=f"t={t:1.2f} s",
                    )
            else:
                plt.plot(
                    model.line_emission["w"].rho_poloidal,
                    model.line_emission["w"],
                    color=cols_time[i],
                    label=f"t={t:1.2f} s",
                )
            plt.xlabel("rho")
            plt.ylabel("w-line local radiated power (W/m^3)")
            plt.legend()
        plt.show(block=True)
    return plasma, model, bckc


if __name__ == "__main__":
    example_run(plot=True, moment_analysis=True, calc_spectra=True, minimum_lines=False)
