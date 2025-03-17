import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s')

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

from indica.available_quantities import READER_QUANTITIES
from indica.converters import LineOfSightTransform
from indica.defaults.load_defaults import load_default_objects
from indica.models.abstract_diagnostic import AbstractDiagnostic
from indica.numpy_typing import LabeledArray
import indica.physics as ph
from indica.readers import ADASReader
from indica.utilities import build_dataarrays
from indica.utilities import get_element_info
from indica.utilities import set_axis_sci
from indica.utilities import set_plot_rcparams


ADF15 = {
    "ne": dict(file_type="pju",
               year="96"),

    "c": dict(file_type="pju",
              year="96"),

    "ar": dict(file_type="llu",
               year="transport")

    # "ar": dict(file_type="ic",
    #            year="40")

}


def read_and_format_adf15(elements: list, wavelength_bounds: slice = None) -> dict[str, xr.DataArray]:
    """
    DataArrays have dimensions of:
    * electron temperature
    * electron density
    * wavelength of transition
    * type of transition (excit, recom, chexc)
    * ion charge of parent ion
    """

    reader = ADASReader()
    pecs = {}
    for element in elements:
        file_type = ADF15[element]["file_type"]
        year = ADF15[element]["year"]
        _pecs = []
        element_info = get_element_info(element)
        for charge in range(element_info[0]):
            _pec = reader.get_adf15(element=element, charge=str(charge), filetype=file_type, year=year)
            _pec = _pec.swap_dims({"index": "wavelength"}).drop_vars("index")
            types = np.unique(_pec.type)
            pec = []
            for _type in types:
                type_pec = _pec.where(_pec.type == _type, drop=True).drop_vars("type")
                type_pec = type_pec.expand_dims(
                    {"type": (_type,)}, )
                pec.append(type_pec)
            _pec = xr.concat(pec, "type")

            # TODO: sum duplicate wavelengths
            # w_index = _pec.wavelength.to_index()
            # _pec.sel(wavelength=w_index.duplicated())
            # no_duplicates = _pec.drop_duplicates("wavelength")
            _pec = _pec.drop_duplicates("wavelength")

            _pec = _pec.expand_dims(
                {"ion_charge": (charge,)}, )
            _pec = _pec.assign_coords({"wavelength": _pec.wavelength * 0.1})  # Angstrom -> nm
            if wavelength_bounds is not None:
                _pec = _pec.where(
                    (_pec.wavelength > wavelength_bounds.start) & (_pec.wavelength < wavelength_bounds.stop), drop=True)
            _pecs.append(_pec)

        element_pec = xr.concat(_pecs, "ion_charge")
        element_pec = element_pec.interpolate_na("electron_temperature").interpolate_na("electron_density")
        pecs[element] = element_pec
    return pecs


class PassiveSpectrometer(AbstractDiagnostic):
    """
    Methods to model passive spectrometer measurements
    """

    def __init__(
            self,
            name: str,
            pecs: dict,
            instrument_method="get_spectrometer",
            window: np.array = None,
    ):

        self.transform: LineOfSightTransform
        self.name = name
        self.pecs = pecs
        self.instrument_method = instrument_method
        self.quantities = READER_QUANTITIES[self.instrument_method]
        self.window = xr.DataArray(window, {"window": window})

        self.intensity: dict[str, xr.DataArray] = None


    def _transition_matrix(self, element: str = "ar") -> xr.DataArray:
        """ Returns transition matrix used to convert
        PECs to emissivity """
        # fmt: off
        _Nimp = self.Nimp.sel(element=element, )
        _Fz = self.Fz[element]
        transition_matrix = xr.concat([
            self.Ne * _Nimp * _Fz,
            self.Ne * _Nimp * _Fz,
            self.Nh * _Nimp * _Fz,
        ], "type").assign_coords(
            type=["excit", "recom", "chexc", ])
        # fmt: on
        return transition_matrix

    def calculate_intensity(
            self,
    ) -> dict[str, xr.DataArray]:
        """
        Returns emissivity with dims:
        * time
        * rho poloidal
        * wavelength of transition
        """
        self.intensity = {}
        for element, pec in self.pecs.items():
            mult = self._transition_matrix(element=element, )
            pec = pec.interp(electron_temperature=self.Te, electron_density=self.Ne)
            self.intensity[element] = (pec * mult).sum("type").sum("ion_charge")
        return self.intensity

    def make_spectra(self, ) -> xr.DataArray:
        """
        Returns

        """
        spectra = []
        for element, intensity in self.intensity.items():
            e_info = get_element_info(element)
            _spectra = ph.doppler_broaden(
                self.window,
                intensity,
                intensity.wavelength,
                e_info[1],  # atomic mass
                self.Ti,  # + self.instrumental_broadening,
            )
            spectra.append(_spectra.sum("wavelength").expand_dims("element").rename({"window": "wavelength"}))

        # TODO: add instrument functions add broadening "filter"
        self.spectra = xr.concat(spectra, "element").sum("element")
        measured_spectra = self.transform.integrate_on_los(
            self.spectra,
            t=self.spectra.t,
            calc_rho=True,
        )
        self.measured_spectra = measured_spectra.assign_coords(
            {"wavelength": self.spectra.wavelength}
        )
        self.spectra_los = self.transform.along_los
        return self.measured_spectra

    def _build_bckc_dictionary(self):
        bckc = {"t": self.t, "channel": np.arange(len(self.transform.x1)), "wavelength": self.measured_spectra.wavelength,
                "location": self.transform.origin, "direction": self.transform.direction,
                "spectra": self.measured_spectra, }
        self.bckc = build_dataarrays(bckc, self.quantities, transform=self.transform)

    def __call__(
            self,
            Te: DataArray = None,
            Ti: DataArray = None,
            Ne: DataArray = None,
            Nimp: DataArray = None,
            Fz: dict = None,
            Nh: DataArray = None,
            t: LabeledArray = None,
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

        self.t = t
        self.Te = Te
        self.Ne = Ne
        self.Nh = Nh
        self.Fz = Fz
        self.Ti = Ti
        self.Nimp = Nimp

        self.calculate_intensity()
        self.make_spectra()
        self._build_bckc_dictionary()
        return self.bckc

    def plot(self):
        set_plot_rcparams("profiles")

        self.transform.plot(np.mean(self.t))

        plt.figure()
        channels = self.transform.x1
        cols_time = cm.gnuplot2(np.linspace(0.1, 0.75, len(self.t), dtype=float))
        if "spectra" in self.bckc.keys():
            spectra = self.bckc["spectra"]
            if "channel" in spectra.dims:
                spectra = spectra.sel(channel=np.median(channels))
            for i, t in enumerate(np.array(self.t, ndmin=1)):
                plt.plot(
                    spectra.wavelength,
                    spectra.sel(t=t),
                    color=cols_time[i],
                    label=f"t={t:1.2f} s",
                )
            plt.ylabel("Emissivity (W/m^3/nm)")
            plt.xlabel("Wavelength (nm)")
            plt.legend()
            set_axis_sci()
        plt.show(block=True)


if __name__ == "__main__":
    origin_x = np.array([1.0, 1.0, 1.0], dtype=float)
    origin_y = np.array([0.0, 0.0, 0.0], dtype=float)
    origin_z = np.array([0.0, 0.0, 0.0], dtype=float)
    direction_x = np.array([-1, -1, -1], dtype=float)
    direction_y = np.array([0.2, 0.25, 0.3], dtype=float)
    direction_z = np.array([0.0, 0.0, 0.0], dtype=float)

    transform = LineOfSightTransform(name="test", origin_x=origin_x,
                                     origin_y=origin_y,
                                     origin_z=origin_z,
                                     direction_x=direction_x,
                                     direction_y=direction_y,
                                     direction_z=direction_z,
                                     machine_dimensions=(
                                         (0, 1),
                                         (-1., 1.0),
                                     )
                                     )
    # transform.plot()
    log = logging.getLogger()
    log.info("starting")
    log.info("loading default objects")
    plasma = load_default_objects("st40", "plasma")
    plasma.electron_temperature *= 0.5
    equilibrium = load_default_objects("st40", "equilibrium")
    transform.set_equilibrium(equilibrium=equilibrium)
    plasma.set_equilibrium(equilibrium)
    # atomic_data
    w_up = 500
    w_low = 3
    window = np.linspace(w_low, w_up, 1000)

    log.info("reading and formatting adf15")
    pecs = read_and_format_adf15(["c"], wavelength_bounds=slice(w_low, w_up))

    log.info("Initialising PassiveSpectrometer")
    p_spec = PassiveSpectrometer(name="test", pecs=pecs, window=window)
    p_spec.set_plasma(plasma)
    p_spec.set_transform(transform)
    log.info("running model")
    p_spec()
    log.info("plotting")
    p_spec.plot()
    plt.show(block=True)
    log.info("finished")

