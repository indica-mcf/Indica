from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
import pickle
from xarray import DataArray
from scipy.interpolate import interp1d
from typing import Tuple

from indica.readers import ADASReader
from indica.converters.line_of_sight_multi import LineOfSightTransform
from indica.numpy_typing import LabeledArray
from indica.readers.available_quantities import AVAILABLE_QUANTITIES

from indica.readers import ST40Reader
from indica.models.plasma import example_run as example_plasma
from indica.equilibrium import Equilibrium
from indica.converters import FluxSurfaceCoordinates
import matplotlib.cm as cm

import indica.physics as ph

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


class Helike_spectroscopy:
    """
    Data and methods to model XRCS spectrometer measurements

    TODO: calibration and Etendue to be correctly included
    """

    def __init__(
        self,
        name: str,
        origin: LabeledArray = None,
        direction: LabeledArray = None,
        dl: float = 0.005,
        passes: int = 1,
        machine_dimensions: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (1.83, 3.9),
            (-1.75, 2.0),
        ),
        instrument_method="get_helike_spectroscopy",
        etendue: float = 1.0,
        calibration: float = 1.0e-18,
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
        extrapolate
            Go beyond validity limit of Machuk's data

        Returns
        -------

        """

        self.name = name
        self.instrument_method = instrument_method

        if origin is not None and direction is not None:
            self.los_transform = LineOfSightTransform(
                origin[:, 0],
                origin[:, 1],
                origin[:, 2],
                direction[:, 0],
                direction[:, 1],
                direction[:, 2],
                name=name,
                dl=dl,
                machine_dimensions=machine_dimensions,
                passes=passes,
            )

        self.etendue = etendue
        self.calibration = calibration
        self.full_run = full_run

        if marchuk:
            self._set_marchuk_pecs(extrapolate=extrapolate)
        else:
            self._set_adas_pecs(adf15=adf15, extrapolate=extrapolate)

        self.bckc = {}
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
        self.fz = None
        self.Nh = None
        self.t = None

    def set_los_transform(self, transform: LineOfSightTransform):
        """
        Parameters
        ----------
        transform
            line of sight transform of the modelled diagnostic
        passes
            number of passes along the line of sight
        """
        self.los_transform = transform
        self.bckc = {}

    def set_flux_transform(self, flux_transform: FluxSurfaceCoordinates):
        """
        set flux surface transform for flux mapping of the line of sight
        """
        self.los_transform.set_flux_transform(flux_transform)
        self.bckc = {}

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
                        self.fz[elem],
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
                    * self.fz[elem].sel(ion_charges=charge)
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

    def _calculate_los_integral(self):
        x1 = self.los_transform.x1
        x2 = self.los_transform.x2

        for line in self.emission.keys():
            self.measured_intensity[line] = self.los_transform.integrate_on_los(
                self.emission[line], x1, x2, t=self.emission[line].t,
            )
            self.emission_los[line] = self.los_transform.along_los
            (
                _,
                self.pos[line],
                self.err_in[line],
                self.err_out[line],
            ) = self._moment_analysis(line, self.t)

        self.t = self.measured_intensity[line].t

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
                ) = self._moment_analysis(line, self.t, profile_1d=self.Ti)
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
                ) = self._moment_analysis(line, self.t, profile_1d=self.Te)
                self.measured_Te[line] = xr.concat(Te_tmp, x1_name).assign_coords(
                    {x1_name: x1}
                )

    def _moment_analysis(
        self, line: str, t: float, profile_1d: DataArray = None, half_los: bool = True,
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
        rho_los = self.los_transform.rho
        result = []
        pos, err_in, err_out = [], [], []
        for chan in self.los_transform.x1:
            _value = None
            _result = []
            _pos, _err_in, _err_out = [], [], []
            for _t in t:
                _rho_los = rho_los[chan].sel(t=_t, method="nearest").values
                if half_los:
                    rho_ind = slice(0, np.argmin(_rho_los) + 1)
                else:
                    rho_ind = slice(0, len(_rho_los))
                _rho_los = _rho_los[rho_ind]
                rho_min = np.min(_rho_los)

                distribution_function = self.emission_los[line][chan].sel(t=_t).values
                dfunction = distribution_function[rho_ind]
                indices = np.arange(0, len(dfunction))
                avrg, dlo, dhi, ind_in, ind_out = ph.calc_moments(
                    dfunction, indices, simmetry=False
                )

                # Position of emissivity
                pos_tmp = _rho_los[int(avrg)]
                err_in_tmp = np.abs(_rho_los[int(avrg)] - _rho_los[int(avrg - dlo)])
                if pos_tmp <= rho_min:
                    pos_tmp = rho_min
                    err_in_tmp = 0.0
                if (err_in_tmp > pos_tmp) and (err_in_tmp > (pos_tmp - rho_min)):
                    err_in_tmp = pos_tmp - rho_min
                err_out_tmp = np.abs(_rho_los[int(avrg)] - _rho_los[int(avrg + dhi)])
                _pos.append(pos_tmp)
                _err_in.append(err_in_tmp)
                _err_out.append(err_out_tmp)

                # Moment analysis of input 1D profile
                if profile_1d is not None:
                    profile_interp = profile_1d.interp(rho_poloidal=_rho_los)
                    if "element" in profile_interp.dims:
                        profile_interp = profile_interp.sel(element=element)
                    if "t" in profile_1d.dims:
                        profile_interp = profile_interp.sel(t=_t, method="nearest")
                    profile_interp = profile_interp.values
                    _value, _, _, _, _ = ph.calc_moments(
                        dfunction,
                        profile_interp,
                        ind_in=ind_in,
                        ind_out=ind_out,
                        simmetry=False,
                    )
                _result.append(_value)

            result.append(DataArray(np.array(_result), coords=[("t", t)]))
            pos.append(DataArray(np.array(_pos), coords=[("t", t)]))
            err_in.append(DataArray(np.array(_err_in), coords=[("t", t)]))
            err_out.append(DataArray(np.array(_err_out), coords=[("t", t)]))

        result = xr.concat(result, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        pos = xr.concat(pos, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        err_in = xr.concat(err_in, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )
        err_out = xr.concat(err_out, self.los_transform.x1_name).assign_coords(
            {self.los_transform.x1_name: self.los_transform.x1}
        )

        return result, pos, err_in, err_out

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
            else:
                print(f"{quant} not available in model for {self.instrument_method}")
                continue

            self.bckc[quantity].attrs["pos"] = {
                "value": self.pos[line],
                "err_in": self.err_in[line],
                "err_out": self.err_out[line],
            }
            self.bckc[quantity].attrs["datatype"] = datatype

    def __call__(
        self,
        Te: DataArray,
        Ti: DataArray,
        Ne: DataArray,
        Nimp: DataArray,
        fractional_abundance: dict,
        Nh: DataArray = None,
        t: LabeledArray = None,
    ):
        """
        Calculate diagnostic measured values

        Parameters
        ----------
        Ne
            Electron density profile
        t

        Returns
        -------

        """

        self.Te = Te
        self.Ti = Ti
        self.Ne = Ne
        self.Nimp = Nimp
        self.fz = fractional_abundance
        if Nh is None:
            Nh = xr.full_like(Ne, 0.0)
        self.Nh = Nh
        self.t = t

        self.quantities = AVAILABLE_QUANTITIES[self.instrument_method]

        # TODO: check that inputs have compatible dimensions/coordinates

        # Calculate emission on natural coordinates of input profiles
        self._calculate_emission()

        # Integrate emission along the LOS
        self._calculate_los_integral()

        # Estimate temperatures from moment analysis
        self._calculate_temperatures()

        # Build back-calculated dictionary to compare with experimental data
        self._build_bckc_dictionary()

        return self.bckc


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


def example_run(use_real_transform=False):

    # TODO: solve issue of LOS sometimes crossing bad EFIT reconstruction outside of the separatrix

    plasma = example_plasma()
    plasma.build_atomic_data()

    # Read equilibrium data and initialize Equilibrium and Flux-surface transform objects
    pulse = 9229
    it = int(len(plasma.t) / 2)
    tplot = plasma.t[it]
    reader = ST40Reader(pulse, plasma.tstart - plasma.dt, plasma.tend + plasma.dt)

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)
    flux_transform = FluxSurfaceCoordinates("poloidal")
    flux_transform.set_equilibrium(equilibrium)

    # Assign transforms to plasma object
    plasma.set_equilibrium(equilibrium)
    plasma.set_flux_transform(flux_transform)

    # Create new diagnostic
    data = None
    diagnostic_name = "xrcs"
    if use_real_transform:
        data = reader.get("sxr", diagnostic_name, 0)
        trans = data[list(data)[0]].transform
        origin = trans.origin
        direction = trans.direction
    else:
        nchannels = 11
        los_end = np.full((nchannels, 3), 0.0)
        los_end[:, 0] = 0.17
        los_end[:, 1] = 0.0
        los_end[:, 2] = np.linspace(0.43, -0.43, nchannels)
        los_start = np.array([[0.8, 0, 0]] * los_end.shape[0])
        origin = los_start
        direction = los_end - los_start

    model = Helike_spectroscopy(
        diagnostic_name,
        origin=origin,
        direction=direction,
        machine_dimensions=plasma.machine_dimensions,
    )
    model.set_flux_transform(plasma.flux_transform)
    model.los_transform.convert_to_rho(
        model.los_transform.x1, model.los_transform.x2, t=tplot
    )

    bckc = model(
        plasma.electron_temperature,
        plasma.ion_temperature,
        plasma.electron_density,
        plasma.impurity_density,
        plasma.fz,
        Nh=plasma.neutral_density,
        t=plasma.t,
    )

    plt.figure()
    equilibrium.rho.sel(t=tplot, method="nearest").plot.contour(
        levels=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    )
    channels = model.los_transform.x1
    cols = cm.gnuplot2(np.linspace(0.1, 0.75, len(channels), dtype=float))
    for chan in channels:
        plt.plot(
            model.los_transform.R[chan],
            model.los_transform.z[chan],
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
        model.los_transform.rho[chan].sel(t=tplot, method="nearest").plot(
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
        bckc["ti_w"].sel(channel=chan).plot(label=f"CH{chan} ti_w", color=cols[chan])
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
    for i, t in enumerate(plasma.t):
        plt.plot(
            model.Ti.rho_poloidal, model.Ti.sel(t=t, element=elem), color=cols_time[i],
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
    for i, t in enumerate(plasma.t):
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
