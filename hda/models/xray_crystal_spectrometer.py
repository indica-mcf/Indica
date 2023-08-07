from copy import deepcopy
import pickle

import hda.physics as ph
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from xarray import DataArray

from indica.converters import FluxSurfaceCoordinates
from indica.converters import LinesOfSightTransform
from indica.numpy_typing import LabeledArray
from indica.readers import ADASReader

#TODO: add Marchuk PECs to repo or to .indica/ (more easily available to others)

MARCHUK = "/home/marco.sertoli/.indica/adas/Marchuk_Argon_PEC.pkl"
ADF15 = {
    "w": {
        "element": "ar",
        "file": ("16", "llu", "transport"),
        "charge": 16,
        "transition": "(1)1(1.0)-(1)0(0.0)",
        "wavelength": 4.0,
    }
}


class XRCSpectrometer:
    """
    Data and methods to model XRCS spectrometer measurements

    TODO: calibration and Etendue to be correctly included
    """

    def __init__(
        self,
        name="",
        etendue: float = 1.0,
        calibration: float = 1.e-18,
        fract_abu: dict = None,
        marchuk: bool = True,
        adf15: dict = None,
        extrapolate: str = None,
        full_run: bool = False,
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
        marchuck
            set to true if Marchuk data to be used in place of ADAS adf15 files
        extrapolate
            Go beyond validity limit of machuk's data
        full_run
            Applies to fractional abundance calculation (see FractAbu.. object) and to interpolation
            of the PECs on electron density

        Returns
        -------

        """
        self.etendue = etendue
        self.calibration = calibration

        self.full_run = full_run
        if fract_abu is not None:
            self.set_fract_abu(fract_abu)

        if marchuk:
            self.set_marchuk_pecs(extrapolate=extrapolate)
        else:
            self.set_adas_pecs(adf15=adf15, extrapolate=extrapolate)

        self.name = name

    def set_los_transform(self, transform: LinesOfSightTransform, passes: int = 1):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform
        self.passes = passes

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)

    def set_fract_abu(self, fract_abu: dict):
        """
        Set dictionary of fractional abundance objects FractionalAbundance to calculate
        ionization balance of elements contribution to emission
        """
        self.fract_abu = fract_abu
        self.elements = list(fract_abu)

    def set_adas_pecs(self, adf15: dict = None, extrapolate: str = None):
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

    def set_marchuk_pecs(self, extrapolate: str = None):
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

    # method previously called "radiation_characteristics"
    def calculate_emission(
        self,
        Te: DataArray,
        Ne: DataArray,
        Nimp: DataArray = None,
        Nh: DataArray = None,
        tau: DataArray = None,
    ):
        """
        Calculate emission of all spectral lines included in the model

        Parameters
        ----------
        Te
            Electron temperature
        Ne
            Electron densit
        Nh
            Neutral (thermal) hydrogen density
        Nimp
            Total impurity densities as defined in plasma.py
        tau
            Residence time for the calculation of the ionisation balance
        Returns
        -------

        """

        if not hasattr(self, "fract_abu"):
            raise Exception(
                "Cannot calculate line emission without fractional abundance object"
            )

        self.Te = Te
        self.Ne = Ne
        self.tau = tau
        if Nh is None:
            Nh = xr.full_like(Ne, 0.0)
        self.Nh = Nh
        if Nimp is None:
            Nimp = []
            for elem in self.elements:
                Nimp.append(deepcopy(Ne * 1.0e-3))
            Nimp = xr.concat(Nimp, "element").assign_coords({"element": self.elements})
        self.Nimp = Nimp

        self.fz = {}
        self.emission = {}

        for line, pec in self.pec.items():
            elem, charge, wavelength = pec["element"], pec["charge"], pec["wavelength"]
            coords = pec["emiss_coeff"].coords

            if elem not in self.fz.keys():
                _fz = self.fract_abu[elem](
                    Te, Ne=Ne, Nh=Nh, tau=tau, full_run=self.full_run
                )
                self.fz[elem] = xr.where(_fz >= 0, _fz, 0)

            # Sum contributions from all transition types
            _emission = []
            if "index" in coords or "type" in coords:
                for t in coords["type"]:
                    _pec = interp_pec(select_type(pec["emiss_coeff"], type=t), Ne, Te)
                    mult = transition_rules(
                        t, self.fz[elem], charge, Ne, Nh, Nimp.sel(element=elem)
                    )
                    _emission.append(_pec * mult)
            else:
                _pec = interp_pec(pec["emiss_coeff"], Ne, Te)
                _emission.append(
                    _pec
                    * self.fz[elem].sel(ion_charges=charge)
                    * Ne
                    * Nimp.sel(element=elem)
                )

            _emission = xr.concat(_emission, "type").sum("type")
            # TODO: convert all wavelengths when reading PECs to nm as per convention at TE!
            ev_wavelength = ph.nm_eV_conversion(nm=wavelength / 10.0)
            self.emission[line] = xr.where(_emission >= 0, _emission, 0) * ev_wavelength

        if "k" in self.emission.keys() and "w" in self.emission.keys():
            self.emission["kw"] = self.emission["k"] * self.emission["w"]
        if "n3" in self.emission.keys() and "w" in self.emission.keys():
            self.emission["n3w"] = self.emission["n3"] * self.emission["w"]
        if (
            "n3" in self.emission.keys()
            and "n345" in self.emission.keys()
            and "w" in self.emission.keys()
        ):
            self.emission["n3tot"] = (
                self.emission["n3"] * self.emission["n345"] * self.emission["w"]
            )

        return self.emission, self.fz

    def map_to_los(self, t: LabeledArray = None):
        """
        Map emission  to LOS

        Parameters
        ----------
        t
            time (s)

        Returns
        -------
        Return emission along line of sight

        """
        if not hasattr(self, "emission"):
            raise Exception("Calculate emission characteristics before mapping to LOS")

        along_los = {}
        for line in self.emission.keys():
            along_los[line] = self.los_transform.map_to_los(self.emission[line], t=t)

        self.along_los = along_los
        return along_los

    def integrate_on_los(self, t: LabeledArray = None):
        """
        Calculate the integral along the line of sight
        For line intensities, the units are W sterad^-1 m^-2

        Parameters
        ----------
        t
            time (s)

        Returns
        -------
        Return line integral and interpolated density along the line of sight

        """
        if not hasattr(self, "emission"):
            raise Exception("Calculate emission characteristics before mapping to LOS")

        along_los = {}
        los_integral = {}
        for line in self.emission.keys():
            _los_integral, along_los[line] = self.los_transform.integrate_on_los(
                self.emission[line], t=t
            )
            los_integral[line] = _los_integral * self.etendue * self.calibration

        if "k" in los_integral.keys() and "w" in los_integral.keys():
            los_integral["int_k/int_w"] = los_integral["k"] / los_integral["w"]
        if "n3" in los_integral.keys() and "w" in los_integral.keys():
            los_integral["int_n3/int_w"] = los_integral["n3"] / los_integral["w"]
        if (
            "n3" in los_integral.keys()
            and "w" in los_integral.keys()
            and "n345" in los_integral.keys()
        ):
            los_integral["int_n3/int_tot"] = los_integral["n3"] / (
                los_integral["n345"] + los_integral["w"]
            )

        self.along_los = along_los
        self.los_integral = los_integral
        return los_integral, along_los

    def calculate_emission_position(
        self,
        t: float,
        line: str = "w",
        half_los: bool = True,
        distribution_function: LabeledArray = None,
    ):
        """
        Calculate emission position and uncertainty using moment analysis
        Parameters
        ----------
        t
            time (s)
        line
            identifier of measured spectral line
        half_los
            set to True if only half of the LOS to be used for the analysis
        """

        if not hasattr(self, "emission"):
            raise Exception("Calculate line emissions before applying moment analysis")
        if not hasattr(self, "los_transform"):
            raise Exception("Assign los_transform before applying moment analysis")
        if not hasattr(self.los_transform, "flux_transform"):
            raise Exception(
                "Assign flux_transform to los_transform before applying moment analysis"
            )

        emiss_los = self.map_to_los(t=t)
        rho_los = self.los_transform.rho.sel(t=t, method="nearest").values
        if half_los:
            rho_ind = slice(0, np.argmin(rho_los) + 1)
        else:
            rho_ind = slice(0, len(rho_los))
        rho_min = np.min(rho_los[rho_ind])

        if distribution_function is None:
            distribution_function = emiss_los[line].values

        dfunction = distribution_function[rho_ind]
        indices = np.arange(0, len(dfunction))
        avrg, dlo, dhi, ind_in, ind_out = ph.calc_moments(
            dfunction, indices, simmetry=False
        )
        pos = rho_los[int(avrg)]
        pos_err_in = np.abs(rho_los[int(avrg)] - rho_los[int(avrg - dlo)])
        if pos <= rho_min:
            pos = rho_min
            pos_err_in = 0.0
        if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
            pos_err_in = pos - rho_min
        pos_err_out = np.abs(rho_los[int(avrg)] - rho_los[int(avrg + dhi)])

        return pos, pos_err_in, pos_err_out

    def moment_analysis(
        self,
        profile_1d: DataArray,
        t: float,
        line="w",
        method="moment",
        half_los: bool = True,
        distribution_function: LabeledArray = None,
    ):
        """
        Calculate measured ion temperature
        Parameters
        -------
        profile_1d
            1D profile on which to perform the moment analysis
        t
            time (s)
        line
            identifier of measured spectral line
        method
            Method for the calculation ("moment" = use moment_analysis, "gaussians" = sum gaussians, ...)
        t
            time (s)
        half_los
            set to True if only half of the LOS to be used for the analysis
        """

        if not hasattr(self, "emission"):
            raise Exception("Calculate line emissions before applying moment analysis")
        if not hasattr(self, "los_transform"):
            raise Exception("Assign los_transform before applying moment analysis")
        if not hasattr(self.los_transform, "flux_transform"):
            raise Exception(
                "Assign flux_transform to los_transform before applying moment analysis"
            )
        if method != "moment":
            raise ValueError("Moment analysis currently only method available")

        emiss_los = self.map_to_los(t=t)
        rho_los = self.los_transform.rho.sel(t=t, method="nearest").values
        if half_los:
            rho_ind = slice(0, np.argmin(rho_los) + 1)
        else:
            rho_ind = slice(0, len(rho_los))

        # Position of emissivity and indices
        if distribution_function is None:
            distribution_function = emiss_los[line].values

        dfunction = distribution_function[rho_ind]
        indices = np.arange(0, len(dfunction))
        avrg, dlo, dhi, ind_in, ind_out = ph.calc_moments(
            dfunction, indices, simmetry=False
        )

        # Moment analysis of input 1D profile
        profile_interp = profile_1d.interp(rho_poloidal=rho_los[rho_ind])
        if "t" in profile_1d.dims:
            profile_interp = profile_interp.sel(t=t, method="nearest")
        profile_interp = profile_interp.values
        value, _, _, _, _ = ph.calc_moments(
            dfunction, profile_interp, ind_in=ind_in, ind_out=ind_out, simmetry=False,
        )

        return value


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


def get_marchuk(extrapolate: str = None, as_is=False):
    print("Using Marchukc PECs")

    el_dens = np.array([1.0e15, 1.0e17, 1.0e19, 1.0e21, 1.0e23])
    adf15 = {
        "w": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 16,
            "transition": "",
            "wavelength": 4.0,
        },
        "z": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 16,
            "transition": "",
            "wavelength": 4.0,
        },
        "k": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 16,
            "transition": "",
            "wavelength": 4.0,
        },
        "n3": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 16,
            "transition": "",
            "wavelength": 4.0,
        },
        "n345": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 16,
            "transition": "",
            "wavelength": 4.0,
        },
        "qra": {
            "element": "ar",
            "file": MARCHUK,
            "charge": 15,
            "transition": "",
            "wavelength": 4.0,
        },
    }

    data = pickle.load(open(MARCHUK, "rb"))
    data *= 1.0e-6  # cm**3 --> m**3
    data = data.rename({"el_temp": "electron_temperature"})

    if as_is:
        return data

    Te = data.electron_temperature.values
    if extrapolate is not None:
        new_data = data.interp(electron_temperature=Te)
        for line in data.line_name:
            y = data.sel(line_name=line).values
            ifin = np.where(np.isfinite(y))[0]
            extrapolate_method = {
                "extrapolate": "extrapolate",
                "constant": (np.log(y[ifin[0]]), np.log(y[ifin[-1]])),
            }
            fill_value = extrapolate_method[extrapolate]

            func = interp1d(
                np.log(Te[ifin]),
                np.log(y[ifin]),
                fill_value=fill_value,
                bounds_error=False,
            )
            new_data.loc[dict(line_name=line)] = np.exp(func(np.log(Te)))
            data = new_data
    else:
        # Restrict data to where all are finite
        ifin = np.array([True] * len(Te))
        for line in data.line_name:
            ifin *= np.where(np.isfinite(data.sel(line_name=line).values), True, False)
        ifin = np.where(ifin == True)[0]
        Te = Te[ifin]
        line_name = data.line_name.values
        new_data = []
        for line in data.line_name:
            y = data.sel(line_name=line).values[ifin]
            new_data.append(DataArray(y, coords=[("electron_temperature", Te)]))
        data = xr.concat(new_data, "line_name").assign_coords(line_name=line_name)

    data = data.expand_dims({"electron_density": el_dens})

    # Reorder data in correct format
    pecs = {}
    w, z, k, n3, n345, qra = [], [], [], [], [], []
    for t in ["w_exc", "w_rec", "w_cxr"]:
        w.append(data.sel(line_name=t, drop=True))
    pecs["w"] = (
        xr.concat(w, "index")
        .assign_coords(index=[0, 1, 2])
        .assign_coords(type=("index", ["excit", "recom", "cxr"]))
    )

    for t in ["z_exc", "z_rec", "z_cxr", "z_isi", "z_diel"]:
        z.append(data.sel(line_name=t, drop=True))
    pecs["z"] = (
        xr.concat(z, "index")
        .assign_coords(index=[0, 1, 2, 3, 4])
        .assign_coords(type=("index", ["excit", "recom", "cxr", "isi", "diel"]))
    )

    pecs["k"] = (
        xr.concat([data.sel(line_name="k_diel", drop=True)], "index")
        .assign_coords(index=[0])
        .assign_coords(type=("index", ["diel"]))
    )

    pecs["n3"] = (
        xr.concat([data.sel(line_name="n3_diel", drop=True)], "index")
        .assign_coords(index=[0])
        .assign_coords(type=("index", ["diel"]))
    )

    pecs["n345"] = (
        xr.concat([data.sel(line_name="n345_diel", drop=True)], "index")
        .assign_coords(index=[0])
        .assign_coords(type=("index", ["diel"]))
    )

    for t in ["qra_ise", "qra_lidiel"]:
        qra.append(data.sel(line_name=t, drop=True))
    pecs["qra"] = (
        xr.concat(qra, "index")
        .assign_coords(index=[0, 1])
        .assign_coords(type=("index", ["ise", "diel"]))
    )

    return adf15, pecs
