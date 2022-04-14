import scipy.constants as constants
from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
import pickle
from xarray import DataArray
from scipy import constants
from scipy.interpolate import interp1d

from indica.readers import ADASReader
from indica.operators.atomic_data import FractionalAbundance
from indica.converters import LinesOfSightTransform

from hda.profiles import Profiles

from indica.numpy_typing import ArrayLike

MARCHUK = "/home/marco.sertoli/python/Indica/hda/Marchuk_Argon_PEC.pkl"
ADF11 = {"ar": {"scd": "89", "acd": "89", "ccd": "89"}}
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

    Parameters
    ----------
    reader
        ADASreader class to read atomic data
    name
        String identifier for the measurement type / spectrometer

    Examples
    ---------
    For passive C5+ measurements:
        spectrometer("c", "5",
                    transition="n=8-n=7", wavelength=5292.7)

    For passive he-like Ar measurements:
        spectrometer("ar", "16",
                    transition="(1)1(1.0)-(1)0(0.0)", wavelength=4.0)

    """

    def __init__(
        self,
        name="",
        adf11: dict = None,
        adf15: dict = None,
        marchuk: bool = False,
        extrapolate:str=None,
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            Identifier for the spectrometer
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)
        marchuk
            Use marchuk PECs instead of ADAS

        Returns
        -------

        """

        self.adasreader = ADASReader()
        self.name = name
        self.set_ion_data(adf11=adf11)
        self.set_pec_data(adf15=adf15, marchuk=marchuk, extrapolate=extrapolate)

    def __call__(
        self,
        Te,
        Ne,
        Nimp: dict = None,
        Nh=None,
        Ti: ArrayLike = None,
        tau=None,
        rho_los: ArrayLike = None,
        dl: float = None,
        use_satellites=False,
        half_los=True,
        fast=False,
    ):

        bckc = {}
        self.radiation_characteristics(
            Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau, fast=fast
        )
        if rho_los is not None and dl is not None:
            self.los_integral(rho_los, dl)
        if Ti is not None:
            bckc = self.moment_analysis(
                Ti,
                rho_los=rho_los,
                dl=dl,
                half_los=half_los,
                use_satellites=use_satellites,
            )

        return bckc

    def set_transform(self, transform: LinesOfSightTransform):
        """
        Set line-of sight transform to perform coordinate conversion and line-of-sight integrals

        Parameters
        ----------
        transform
            Line of sight transform

        Returns
        -------

        """
        self.transform = transform

    def test_flow(self):
        """
        Test module with standard inputs
        """
        Ne = Profiles(datatype=("density", "electron"))
        Ne.peaking = 1
        Ne.build_profile()

        Te = Profiles(datatype=("temperature", "electron"))
        Te.y0 = 1.0e3
        Te.y1 = 20
        Te.wped = 1
        Te.peaking = 8
        Te.build_profile()

        # Ti = Profiles(datatype=("temperature", "ion"))
        Ti = deepcopy(Te)
        Ti.datatype = ("temperature", "ion")

        Ne = Ne.yspl
        Te = Te.yspl
        Ti = Ti.yspl
        Ti /= 2.0

        Nh_1 = 1.0e15
        Nh_0 = Nh_1 / 10
        Nh = Ne.rho_poloidal ** 5 * (Nh_1 - Nh_0) + Nh_0

        plt.figure()
        for line, pec in self.pec.items():
            pec_to_plot = pec["emiss_coeff"].sel(electron_density=1.0e19, method="nearest")
            if "type" in pec["emiss_coeff"].coords:
                for t in pec["emiss_coeff"].type:
                    select_type(pec_to_plot, type=t).plot(label=f"{line} {t.values}")
            else:
                pec_to_plot.plot(label=f"{line}")

        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(np.max(pec_to_plot)/1.e3, np.max(pec_to_plot))
        plt.title("PEC")
        plt.legend()

        fz0, emiss0 = self.radiation_characteristics(Te, Ne)
        bckc0 = self.moment_analysis(Ti)
        plt.figure()
        for line in emiss0.keys():
            emiss0[line].plot(label=line, marker="o")
        # plt.yscale("log")
        plt.title("Line emission shells")
        plt.legend()

        plt.figure()
        for line in emiss0.keys():
            plt.plot(
                emiss0[line].electron_temperature,
                emiss0[line].values,
                label=line,
                marker="o",
            )
        # plt.yscale("log")
        plt.title("Line emission shells")
        plt.legend()

        fz1, emiss1 = self.radiation_characteristics(Te, Ne, Nh=Nh)
        bckc1 = self.moment_analysis(Ti)

        plt.figure()
        emiss0["w"].plot(label="Nh = 0", marker="o")
        emiss1["w"].plot(label="Nh != 0", marker="x")
        plt.title("w-line PEC * fz")
        plt.legend()

        plt.figure()
        self.Te.plot(label="Te", color="red")
        plt.plot(
            bckc0["te_kw"].rho_poloidal, bckc0["te_kw"].values, marker="o", color="red"
        )
        plt.plot(
            bckc1["te_kw"].rho_poloidal, bckc1["te_kw"].values, marker="x", color="red"
        )
        plt.hlines(
            bckc0["te_kw"].values,
            bckc0["te_kw"].rho_poloidal - bckc0["te_kw"].rho_poloidal_err["in"],
            bckc0["te_kw"].rho_poloidal + bckc0["te_kw"].rho_poloidal_err["out"],
            color="red",
        )
        plt.hlines(
            bckc1["te_kw"].values,
            bckc1["te_kw"].rho_poloidal - bckc1["te_kw"].rho_poloidal_err["in"],
            bckc1["te_kw"].rho_poloidal + bckc1["te_kw"].rho_poloidal_err["out"],
            color="red",
        )
        self.Ti.plot(label="Ti", color="black")
        plt.plot(
            bckc0["ti_w"].rho_poloidal, bckc0["ti_w"].values, marker="o", color="black"
        )
        plt.plot(
            bckc1["ti_w"].rho_poloidal, bckc1["ti_w"].values, marker="x", color="black"
        )
        plt.hlines(
            bckc0["ti_w"].values,
            bckc0["ti_w"].rho_poloidal - bckc0["ti_w"].rho_poloidal_err["in"],
            bckc0["ti_w"].rho_poloidal + bckc0["ti_w"].rho_poloidal_err["out"],
            color="black",
        )
        plt.hlines(
            bckc1["ti_w"].values,
            bckc1["ti_w"].rho_poloidal - bckc1["ti_w"].rho_poloidal_err["in"],
            bckc1["ti_w"].rho_poloidal + bckc1["ti_w"].rho_poloidal_err["out"],
            color="black",
        )
        plt.title("w-line PEC * fz")
        plt.legend()

        # return (fz0, fz1), (emiss0, emiss1)

    def set_ion_data(self, adf11: dict = None):
        """
        Read adf11 data and build fractional abundance objects for all elements
        whose lines are to included in the modelled spectra

        Parameters
        ----------
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)

        """

        fract_abu = {}
        if adf11 is None:
            adf11 = ADF11

        scd, acd, ccd = {}, {}, {}
        for elem in adf11.keys():
            scd[elem] = self.adasreader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd[elem] = self.adasreader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd[elem] = self.adasreader.get_adf11("ccd", elem, adf11[elem]["ccd"])

            fract_abu[elem] = FractionalAbundance(scd[elem], acd[elem], CCD=ccd[elem],)

        self.adf11 = adf11
        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        self.fract_abu = fract_abu

    def set_pec_data(self, adf15: dict = None, marchuk=False, extrapolate:str=None):
        """
        Read adf15 data and extract PECs for all lines to be included
        in the modelled spectra

        Parameters
        ----------
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)

        """

        if adf15 is None:
            adf15 = ADF15

        elements = []
        if marchuk:
            adf15, adf15_data = get_marchuk(extrapolate=extrapolate)
        pec = deepcopy(adf15)
        for line in adf15.keys():
            element = adf15[line]["element"]
            transition = adf15[line]["transition"]
            wavelength = adf15[line]["wavelength"]
            if marchuk:
                pec[line]["emiss_coeff"] = adf15_data[line]
            else:
                charge, filetype, year = adf15[line]["file"]
                adf15_data = self.adasreader.get_adf15(
                    element, charge, filetype, year=year
                )
                pec[line]["emiss_coeff"] = select_transition(
                    adf15_data, transition, wavelength
                )
            elements.append(element)

        # For fast computation, drop electron density dimension, calculate fz vs. Te only
        pec_fast = deepcopy(pec)
        for line in pec:
            pec_fast[line]["emiss_coeff"] = (
                pec[line]["emiss_coeff"]
                .sel(electron_density=4.0e19, method="nearest")
                .drop("electron_density")
            )

        fz_fast = {}
        Te = np.linspace(0, 1, 51) ** 3 * (10.0e3 - 50) + 50
        Te = DataArray(Te)
        Ne = DataArray(np.array([5.0e19] * 51))
        for elem in elements:
            _fz = self.fract_abu[elem](Ne, Te)
            _fz = xr.where(_fz >= 0, _fz, 0).assign_coords(
                electron_temperature=("dim_0", Te)
            )
            fz_fast[elem] = _fz.swap_dims({"dim_0": "electron_temperature"}).drop(
                "dim_0"
            )

        self.adf15 = adf15
        self.pec = pec
        self.pec_fast = pec_fast
        self.fz_fast = fz_fast
        self.elements = elements

    def radiation_characteristics(
        self, Te, Ne, Nimp: dict = None, Nh=None, tau=None, fast=False,
    ):
        """

        Parameters
        ----------
        Te
            Electron temperature for interpolation of atomic data
        Ne
            Electron density for interpolation of atomic data
        Nh
            Neutral (thermal) hydrogen density for interpolation of atomic data
        Nimp
            Total impurity densities for calculation of emission profiles

        Returns
        -------

        """

        if Nh is None:
            Nh = xr.full_like(Ne, 0.0)

        if fast:
            if not np.all(Nh.values == 0) or tau is not None:
                print("\n Fast calculation available only with: Nh = None, tau = None")
                raise ValueError

        self.Ne = Ne
        self.Nh = Nh
        self.tau = tau
        if Nimp is None:
            Nimp = {}
            for elem in self.elements:
                Nimp[elem] = xr.full_like(Ne, 1.0)
        self.Nimp = Nimp
        self.Te = Te

        fz = {}
        emiss = {}
        for line, pec in self.pec.items():
            elem = pec["element"]
            charge = pec["charge"]
            wavelength = pec["wavelength"]
            coords = pec["emiss_coeff"].coords

            # Calculate fractional abundance if not already available
            if elem not in fz.keys():
                if fast:
                    fz[elem] = (
                        self.fz_fast[elem]
                        .interp(electron_temperature=Te)
                        .drop("electron_temperature")
                    )
                else:
                    _fz = self.fract_abu[elem](Ne, Te, Nh, tau=tau)
                    fz[elem] = xr.where(_fz >= 0, _fz, 0)

            # Sum contributions from all transition types
            _emiss = []
            if "index" in coords or "type" in coords:
                for t in coords["type"]:
                    _pec = interp_pec(select_type(pec["emiss_coeff"], type=t), Ne, Te)
                    mult = transition_rules(t, fz[elem], charge, Ne, Nh, Nimp[elem])
                    _emiss.append(_pec * mult)
            else:
                _pec = interp_pec(pec["emiss_coeff"], Ne, Te)
                _emiss.append(_pec * fz[elem].sel(ion_charges=charge) * Ne * Nimp[elem])

            _emiss = xr.concat(_emiss, "type").sum("type")
            ev_wavelength = constants.e * 1.239842e3 / (wavelength)
            emiss[line] = xr.where(_emiss >= 0, _emiss, 0) * ev_wavelength

        self.fast = fast
        self.fz = fz
        self.emiss = emiss

        return fz, emiss

    def moment_analysis(
        self,
        Ti: ArrayLike,
        rho_los: ArrayLike = None,
        dl: float = None,
        use_satellites=False,
        half_los=True,
    ):
        """
        Infer the spectrometer measurement of electron and ion temperatures

        Parameters
        ----------
        Ti
            Ion temperature profile
        rho_los
            rho_poloidal for interpolation along the LOS
        dl
            LOS radial precision for integration
        use_satellites
            Use convolution of spectral line emission shells for moment analysis
        half_los
            Calculate moment analysis for half of the LOS only

        Returns
        -------

        """
        coord = "rho_poloidal"
        data = {}

        self.Ti = deepcopy(Ti)

        if dl is not None and rho_los is not None:
            self.dl = dl
            if half_los:
                rho_los = rho_los[0 : np.argmin(rho_los.values)]
        if rho_los is None:
            rho_los = self.Te.rho_poloidal
            dl = 1.0

        self.rho_los = rho_los
        Te = self.Te.interp(rho_poloidal=rho_los)
        Ti = self.Ti.interp(rho_poloidal=rho_los)
        intensity, emiss_interp = self.los_integral(rho_los, dl)

        rho_tmp = rho_los.values
        rho_min = np.min(rho_tmp)

        for line in self.pec.keys():
            # Ion temperature and position of emissivity
            x = np.array(range(len(emiss_interp[line])))
            y = emiss_interp[line]
            avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)
            pos = rho_tmp[int(avrg)]
            pos_err_in = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
            if pos <= rho_min:
                pos = rho_min
                pos_err_in = 0.0
            if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
                pos_err_in = pos - rho_min
            pos_err_out = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)])

            x = emiss_interp[line]
            y = Ti
            ti_w, err_in, err_out, _, _ = calc_moments(
                x, y, ind_in=ind_in, ind_out=ind_out, simmetry=False
            )
            datatype = ("temperature", "ion")
            attrs = {
                "datatype": datatype,
                "err": {"in": err_in, "out": err_out},
                f"{coord}_err": {"in": pos_err_in, "out": pos_err_out},
            }
            data[f"ti_{line}"] = DataArray([ti_w], coords=[(coord, [pos])], attrs=attrs)

            datatype = ("intensity", "spectral_line")
            attrs = {
                "datatype": datatype,
                "err": {"in": err_in, "out": err_out},
                f"{coord}_err": {"in": pos_err_in, "out": pos_err_out},
            }
            data[f"int_{line}"] = DataArray(
                [intensity[line]], coords=[(coord, [pos])], attrs=attrs
            )

        # Electron temperature(s) and position
        if use_satellites:
            emiss_shell = {
                "te_kw": emiss_interp["w"] * emiss_interp["k"],
                "te_n3w": emiss_interp["w"] * emiss_interp["n3"],
            }
        else:
            emiss_shell = {"te": emiss_interp["w"]}

        for key in emiss_shell.keys():
            x = np.array(range(len(emiss_shell[key])))
            y = emiss_shell[key]
            avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)
            pos = rho_tmp[int(avrg)]
            pos_err_in = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
            if pos <= rho_min:
                pos = rho_min
                pos_err_in = 0.0
            if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
                pos_err_in = pos - rho_min
            pos_err_out = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)])

            x = emiss_shell[key]
            y = Te
            te_val, err_in, err_out, _, _ = calc_moments(
                x, y, ind_in=ind_in, ind_out=ind_out, simmetry=False
            )
            datatype = ("temperature", "electron")
            attrs = {
                "datatype": datatype,
                "err": {"in": err_in, "out": err_out},
                f"{coord}_err": {"in": pos_err_in, "out": pos_err_out},
            }
            if use_satellites:
                data[key] = DataArray([te_val], coords=[(coord, [pos])], attrs=attrs)
            else:
                data["te_kw"] = DataArray(
                    [te_val], coords=[(coord, [pos])], attrs=attrs
                )
                data["te_n3w"] = DataArray(
                    [te_val], coords=[(coord, [pos])], attrs=attrs
                )

        self.data = data

        return data

    def los_integral(self, rho_los, dl):
        """
        Calculate line of sight integral for a specified number of passes through the plasma

        Returns
        -------
        los_int
            Integral along the line of sight

        """

        intensity = {}
        emiss_interp = {}

        for line in self.pec.keys():
            emiss_interp[line] = self.emiss[line].interp(rho_poloidal=rho_los)
            emiss_interp[line] = xr.where(rho_los <= 1, emiss_interp[line], 0,)

            intensity[line] = emiss_interp[line].sum() * dl

        self.intensity = intensity

        return intensity, emiss_interp


def calc_moments(y: ArrayLike, x: ArrayLike, ind_in=None, ind_out=None, simmetry=False):
    x_avrg = np.nansum(y * x) / np.nansum(y)

    if (ind_in is None) and (ind_out is None):
        ind_in = x <= x_avrg
        ind_out = x >= x_avrg
        if simmetry is True:
            ind_in = ind_in + ind_out
            ind_out = ind_in

    x_err_in = np.sqrt(
        np.nansum(y[ind_in] * (x[ind_in] - x_avrg) ** 2) / np.nansum(y[ind_in])
    )

    x_err_out = np.sqrt(
        np.nansum(y[ind_out] * (x[ind_out] - x_avrg) ** 2) / np.nansum(y[ind_out])
    )

    return x_avrg, x_err_in, x_err_out, ind_in, ind_out


def interp_pec(pec, Ne, Te):
    pec_interp = pec.indica.interp2d(
        electron_temperature=Te,
        electron_density=Ne,
        method="cubic",
        assume_sorted=True,
    )
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


def get_marchuk(extrapolate:str=None, as_is=False):
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
            extrapolate_method = {"extrapolate":"extrapolate",
                                  "constant":(np.log(y[ifin[0]]), np.log(y[ifin[-1]]))}
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
