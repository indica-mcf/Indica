import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.converters.line_of_sight import LineOfSightTransform
from indica.models.abstractdiagnostic import DiagnosticModel
from indica.models.plasma import example_plasma
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers.available_quantities import AVAILABLE_QUANTITIES
from indica.readers.marchuk import MARCHUKReader
from indica.utilities import assign_datatype
from indica.utilities import get_element_info
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams

# TODO: why resonance lines in upper case, others lower?
LINE_RANGES = {
    "w": slice(0.39489, 0.39494),
    "n3": slice(0.39543, 0.39574),
    "n345": slice(0.39496, 0.39574),
    "qra": slice(0.39810, 0.39865),
    "k": slice(0.39890, 0.39910),
    "z": slice(0.39935, 0.39950),
}


class HelikeSpectrometer(DiagnosticModel):
    """
    Data and methods to model XRCS spectrometer measurements

    TODO: calibration and Etendue to be correctly included
    """

    def __init__(
        self,
        name: str,
        instrument_method="get_helike_spectroscopy",
        etendue: float = 1.0,
        calibration: float = 1.0e-18,
        element: str = "ar",
        window_len: int = 1030,
        window_lim=None,
        window: np.array = None,
        window_masks=None,
        line_labels=None,
        background=None,
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            String identifier for the spectrometer

        """
        if window_lim is None:
            window_lim = [0.394, 0.401]
        if window_masks is None:
            window_masks = []
        if line_labels is None:
            line_labels = ["w", "k", "n3", "n345", "z", "qra"]

        self.name = name
        self.instrument_method = instrument_method
        self.element: str = element
        z_elem, a_elem, name_elem, _ = get_element_info(element)
        self.ion_charge: int = z_elem - 2  # He-like
        self.ion_mass: float = a_elem
        self.etendue = etendue
        self.calibration = calibration
        self.window_masks = window_masks
        self.line_ranges = LINE_RANGES
        self.line_labels = line_labels
        self.background = background

        if window is None:
            window = np.linspace(window_lim[0], window_lim[1], window_len)
        mask = np.zeros(shape=window.shape)
        if self.window_masks:
            for mslice in self.window_masks:
                mask[(window > mslice.start) & (window < mslice.stop)] = 1
        else:
            mask[:] = 1
        self.window = DataArray(mask, coords=[("wavelength", window)])
        self._get_atomic_data(self.window)

        self.line_emission: dict
        self.emission_los: dict
        self.measured_intensity: dict
        self.measured_Te: dict
        self.measured_Ti: dict
        self.measured_Nimp: dict
        self.pos: dict
        self.pos_err_in: dict
        self.pos_err_out: dict
        self.spectra: DataArray

        self.Ti: DataArray
        self.Te: DataArray
        self.Ne: DataArray
        self.Nimp: DataArray
        self.Fz: dict
        self.Nh: DataArray

    def _get_atomic_data(self, window: DataArray):
        """
        Read atomic data and keep only lines in desired wavelength window
        """
        marchuck_reader = MARCHUKReader(
            element=self.element,
            charge=self.ion_charge,
        )

        pecs = marchuck_reader.pecs
        mask = window.interp(wavelength=pecs.wavelength)
        pecs = pecs.where(
            mask,
            drop=True,
        )
        pecs = pecs.where(
            (pecs.wavelength < window.wavelength.max())
            & (pecs.wavelength > window.wavelength.min()),
            drop=True,
        )
        self.mask = mask
        self.pecs = pecs

    def _transition_matrix(self, element="ar", charge=16):
        """vectorisation of the transition matrix used to convert
        Helike PECs to emissivity"""
        # fmt: off
        _Nimp = self.Nimp.sel(element=element, )
        _Fz = self.Fz[element]
        transition_matrix = xr.concat([
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge, ),
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge, ),
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge - 1, ),
            self.Ne * _Nimp * _Fz.sel(ion_charge=charge + 1, ),
            self.Nh * _Nimp * _Fz.sel(ion_charge=charge + 1, ),
        ], "type").assign_coords(
            type=["excit", "diel", "li_diel", "ise", "isi", "recom", "cxr", ])
        # fmt: on
        return transition_matrix

    def _calculate_intensity(
        self,
    ):
        """
        Returns DataArrays of emission type with co-ordinates of line label and
        spatial co-ordinate
        """
        mult = self._transition_matrix(element=self.element, charge=self.ion_charge)

        # Swapping to dataset and then dropping line_names with NaNs is much faster
        _pecs_ds = self.pecs.to_dataset("type")
        temp = [
            _pecs_ds[type]
            .dropna("line_name", how="all")
            .interp(electron_temperature=self.Te, assume_sorted=True)
            for type in _pecs_ds.data_vars.keys()
        ]
        _intensity = xr.merge(temp).to_array("type")
        intensity = (_intensity * mult * self.calibration).sum("type")
        self.intensity = intensity

        return intensity

    def _make_spectra(self, calc_rho: bool = False):
        """
        TODO: Doppler Shift / Add convolution of broadening
        -> G(x, mu1, sig1) * G(x, mu2, sig2) = G(x, mu1+mu2, sig1**2 + sig2**2)
        instrument function / photon noise / photon -> counts
        Background Noise

        """
        element = self.element
        _spectra = ph.doppler_broaden(
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
        self.spectra = spectra

        measured_spectra = self.los_transform.integrate_on_los(
            self.spectra,
            t=self.spectra.t,
            calc_rho=calc_rho,
        )
        measured_spectra = measured_spectra.assign_coords(
            {"wavelength": self.spectra.wavelength}
        )
        measured_spectra = xr.where(measured_spectra == 0, np.nan, measured_spectra)
        self.measured_spectra = measured_spectra.sortby("wavelength")
        self.spectra_los = self.los_transform.along_los

    def _moment_analysis(self):
        """
        Perform moment analysis using a lines defined in line_labels,
        calculating the position of emissivity, and expected Te and Ti
        """

        line_emission = {}
        for line_name in self.line_labels:
            _line_emission = self.intensity.where(
                (self.intensity.wavelength > self.line_ranges[line_name].start)
                & (self.intensity.wavelength < self.line_ranges[line_name].stop),
                drop=True,
            )
            line_emission[line_name] = _line_emission.sum("line_name")

        if "k" in line_emission.keys() and "w" in line_emission.keys():
            line_emission["kw"] = line_emission["k"] * line_emission["w"]
        if "n3" in line_emission.keys() and "w" in line_emission.keys():
            line_emission["n3w"] = line_emission["n3"] * line_emission["w"]
        if (
            "n3" in line_emission.keys()
            and "n345" in line_emission.keys()
            and "w" in line_emission.keys()
        ):
            line_emission["tot"] = line_emission["n345"] + line_emission["w"]
            line_emission["n3tot"] = line_emission["tot"]
        self.line_emission = line_emission

        rho_mean = {}
        rho_err_in = {}
        rho_err_out = {}
        measured_intensity = {}
        emission_los = {}
        measured_Te = {}
        measured_Ti = {}
        measured_Nimp = {}
        for line in self.line_emission.keys():
            channels = self.los_transform.x1
            if len(self.los_transform.x1) == 1:
                channels = self.los_transform.x1[0]

            emission = self.line_emission[line]
            los_integral = self.los_transform.integrate_on_los(emission, t=emission.t)
            emission_los = self.los_transform.along_los.sel(channel=channels).mean(
                "beamlet"
            )
            emission_sum = emission_los.sum("los_position", skipna=True)
            rho_los = self.los_transform.rho.sel(channel=channels)

            rho_mean[line] = (emission_los * rho_los).sum(
                "los_position", skipna=True
            ) / emission_sum
            rho_err = rho_los  # rho_los.where(indx_err, np.nan,)
            where_in = rho_err < rho_mean[line]
            where_out = rho_err > rho_mean[line]
            rho_in = xr.where(where_in, rho_los, np.nan)
            rho_out = xr.where(where_out, rho_los, np.nan)
            rho_err_in[line] = (
                (emission_los * (rho_in - rho_mean[line]) ** 2).sum(
                    "los_position", skipna=True
                )
                / emission_sum
            ) ** 0.5
            rho_err_out[line] = (
                (emission_los * (rho_out - rho_mean[line]) ** 2).sum(
                    "los_position", skipna=True
                )
                / emission_sum
            ) ** 0.5
            measured_intensity[line] = los_integral
            emission_los[line] = emission_los

            Te_along_los = self.los_transform.map_profile_to_los(
                self.Te, t=emission.t
            ).sel(channel=channels)
            measured_Te[line] = (emission_los * Te_along_los).sum(
                "los_position", skipna=True
            ) / emission_sum

            Ti_along_los = self.los_transform.map_profile_to_los(
                self.Ti.sel(element=self.element), t=emission.t
            ).sel(channel=channels)
            measured_Ti[line] = (emission_los * Ti_along_los).sum(
                "los_position", skipna=True
            ) / emission_sum

            Nimp_along_los = self.los_transform.map_profile_to_los(
                self.Nimp.sel(element=self.element), t=emission.t
            ).sel(channel=channels)
            measured_Nimp[line] = (emission_los * Nimp_along_los).sum(
                "los_position", skipna=True
            ) / emission_sum

        self.pos = rho_mean
        self.pos_err_in = rho_err_in
        self.pos_err_out = rho_err_out
        self.measured_intensity = measured_intensity
        self.emission_los = emission_los
        self.measured_Te = measured_Te
        self.measured_Ti = measured_Ti
        self.measured_Nimp = measured_Nimp

    def _build_bckc_dictionary(self):
        self.bckc = {}
        if "spectra" in self.quantities and hasattr(self, "measured_spectra"):
            self.bckc["spectra"] = self.measured_spectra

        if self.moment_analysis:
            for quantity in self.quantities:
                if quantity == "spectra":
                    continue

                datatype = self.quantities[quantity]
                line = str(quantity.split("_")[1])
                if "int" in quantity and line in self.measured_intensity.keys():
                    self.bckc[quantity] = self.measured_intensity[line]
                elif "te" in quantity and line in self.measured_Te.keys():
                    self.bckc[quantity] = self.measured_Te[line]
                elif "ti" in quantity and line in self.measured_Ti.keys():
                    self.bckc[quantity] = self.measured_Ti[line]
                else:
                    print(
                        f"{quantity} not available in model "
                        f"for {self.instrument_method}"
                    )
                    continue

                self.bckc[quantity].attrs["emiss"] = self.line_emission[line]
                assign_datatype(self.bckc[quantity], datatype)

                if line in self.pos.keys():
                    self.bckc[quantity].attrs["pos"] = self.pos[line]
                    self.bckc[quantity].attrs["pos_err_in"] = self.pos_err_in[line]
                    self.bckc[quantity].attrs["pos_err_out"] = self.pos_err_out[line]

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
        calc_rho: bool = False,
        moment_analysis: bool = False,
        background: int = None,
        pixel_offset: int = None,
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
        self.calc_rho = calc_rho
        self.moment_analysis = moment_analysis

        if self.moment_analysis and bool(self.window_masks):
            raise ValueError(
                "moment_analysis cannot be used when window_masks is not set to None"
            )

        if hasattr(self, "plasma"):
            if t is None:
                t = self.plasma.time_to_calculate
            Te = self.plasma.electron_temperature.sel(
                t=t,
            )
            Ne = self.plasma.electron_density.sel(
                t=t,
            )
            Nh = self.plasma.neutral_density.sel(
                t=t,
            )
            Fz = {}
            _Fz = self.plasma.fz
            for elem in _Fz.keys():
                Fz[elem] = _Fz[elem].sel(t=t)

            Ti = self.plasma.ion_temperature.sel(t=t)
            Nimp = self.plasma.impurity_density.sel(t=t)
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

        if background is None:
            if self.background is not None:
                background = self.background.sel(t=t)

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nh = Nh
        self.Fz = Fz
        self.Ti = Ti
        self.Nimp = Nimp
        self.quantities: dict = AVAILABLE_QUANTITIES[self.instrument_method]

        # TODO: check that inputs have compatible dimensions/coordinates
        self._calculate_intensity()
        self._make_spectra()

        if moment_analysis:
            self._moment_analysis()

        if background is not None:
            self.measured_spectra = self.measured_spectra + background

        if pixel_offset is not None:
            self.measured_spectra = self.measured_spectra.shift(
                wavelength=round(pixel_offset), fill_value=np.nan
            )

        self._build_bckc_dictionary()
        return self.bckc

    def plot(self):
        set_plot_rcparams("profiles")

        self.los_transform.plot()

        plt.figure()
        channels = self.los_transform.x1
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.plasma.t), dtype=float))
        if "spectra" in self.bckc.keys():
            spectra = self.bckc["spectra"].sel(channel=np.median(channels))
            for i, t in enumerate(np.array(self.t, ndmin=1)):
                plt.plot(
                    spectra.wavelength,
                    spectra.sel(t=t),
                    color=cols_time[i],
                    label=f"t={t:1.2f} s",
                )
            plt.ylabel("Brightness (a.u.)")
            plt.xlabel("Wavelength (nm)")
            plt.legend()
            set_axis_sci()

        # Plot the temperatures profiles
        plt.figure()
        elem = self.Ti.element[0].values
        for i, t in enumerate(self.t):
            plt.plot(
                self.plasma.ion_temperature.rho_poloidal,
                self.plasma.ion_temperature.sel(t=t, element=elem),
                color=cols_time[i],
            )
            plt.plot(
                self.plasma.electron_temperature.rho_poloidal,
                self.plasma.electron_temperature.sel(t=t),
                color=cols_time[i],
                linestyle="dashed",
            )
        plt.xlabel("rho")
        set_axis_sci()
        plt.ylabel("Ti and Te profiles (eV)")

        # Plot the emission profiles
        if self.moment_analysis:
            if "w" in self.line_emission.keys():
                plt.figure()
                if "t" in self.line_emission["w"].dims:
                    for i, t in enumerate(self.plasma.t.values):
                        plt.plot(
                            self.line_emission["w"].rho_poloidal,
                            self.line_emission["w"].sel(t=t),
                            color=cols_time[i],
                            label=f"t={t:1.2f} s",
                        )
                else:
                    plt.plot(
                        self.line_emission["w"].rho_poloidal,
                        self.line_emission["w"],
                        color=cols_time[i],
                        label=f"t={t:1.2f} s",
                    )
                plt.xlabel("rho")
                plt.ylabel("w-line emissivity (W/m^3)")
                set_axis_sci()
                plt.legend()

        # Plot moment analysis of measured temperatures
        if "ti_w" in self.bckc.keys() & "te_kw" in self.bckc.keys():
            plt.figure()
            for i, t in enumerate(self.plasma.t.values):
                self.bckc["ti_w"].mean("beamlet").sel(t=t).plot(
                    label=f"t={t:1.2f} s",
                    marker="o",
                    color=cols_time[i],
                )
                self.bckc["te_kw"].mean("beamlet").sel(t=t).plot(
                    color=cols_time[i],
                    marker="x",
                    linestyle="dashed",
                )

            plt.xlabel("Channel")
            plt.ylabel("Measured Ti and Te (eV)")
            set_axis_sci()
            plt.title("")
            # plt.legend()

        plt.show(block=True)


def helike_transform_example(nchannels):
    los_end = np.full((nchannels, 3), 0.0)
    los_end[:, 0] = 0.17
    los_end[:, 1] = 0.0
    los_end[:, 2] = np.linspace(0.2, -0.5, nchannels)
    los_start = np.array([[0.9, 0, 0]] * los_end.shape[0])
    los_start[:, 2] = -0.1
    origin = los_start
    direction = los_end - los_start

    los_transform = LineOfSightTransform(
        origin[0:nchannels, 0],
        origin[0:nchannels, 1],
        origin[0:nchannels, 2],
        direction[0:nchannels, 0],
        direction[0:nchannels, 1],
        direction[0:nchannels, 2],
        name="xrcs",
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        passes=1,
    )
    return los_transform


def example_run(
    pulse: int = None, plasma=None, plot=False, moment_analysis: bool = False, **kwargs
):
    # TODO: LOS sometime crossing bad EFIT reconstruction
    if plasma is None:
        plasma = example_plasma(
            pulse=pulse, impurities=("ar",), impurity_concentration=(0.001,), n_rad=10
        )
        # plasma.time_to_calculate = plasma.t[3:5]
        # Create new diagnostic
    diagnostic_name = "xrcs"
    los_transform = helike_transform_example(3)
    los_transform.set_equilibrium(plasma.equilibrium)
    model = HelikeSpectrometer(
        diagnostic_name,
        window_masks=[],
    )
    model.set_los_transform(los_transform)
    model.set_plasma(plasma)

    bckc = model(moment_analysis=moment_analysis, **kwargs)

    # Plot spectra
    if plot:
        model.plot()

    return plasma, model, bckc


if __name__ == "__main__":
    example_run(plot=True, moment_analysis=True)
