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
        recom=False,
        adf11: dict = None,
        adf15: dict = None,
        marchuk: bool = False,
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            Identifier for the spectrometer
        recom
            Set to True if
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)
        marchuk
            Use marchuk PECs instead of ADAS

        Returns
        -------

        """

        self.ADASReader = ADASReader()
        self.name = name
        self.recom = recom
        self.set_ion_data(adf11=adf11)
        self.set_pec_data(adf15=adf15, marchuk=marchuk)

    def __call__(
        self,
        Te,
        Ne,
        Nimp: dict = None,
        Nh=None,
        Ti: ArrayLike = None,
        tau=None,
        recom=False,
        rho_los: ArrayLike = None,
        dl: float = None,
        use_satellites=False,
        half_los=True,
    ):

        bckc = {}
        self.radiation_characteristics(Te, Ne, Nimp=Nimp, Nh=Nh, tau=tau, recom=recom)
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
        self.set_ion_data()
        self.set_pec_data()
        Ne = Profiles(datatype=("density", "electron")).yspl
        Te = Profiles(datatype=("temperature", "electron")).yspl
        Ti = Profiles(datatype=("temperature", "electron")).yspl
        Ti /= 2.0

        Nh_1 = 1.0e15
        Nh_0 = Nh_1 / 10
        Nh = Ne.rho_poloidal ** 5 * (Nh_1 - Nh_0) + Nh_0

        fz0, emiss0 = self.radiation_characteristics(Te, Ne)
        te_kw0, ti_w0 = self.moment_analysis(Ti)

        fz1, emiss1 = self.radiation_characteristics(Te, Ne, Nh=Nh)
        te_kw1, ti_w1 = self.moment_analysis(Ti)

        plt.figure()
        emiss0["w"].plot(label="Nh = 0", marker="o")
        emiss1["w"].plot(label="Nh != 0", marker="x")
        plt.title("w-line PEC * fz")
        plt.legend()

        plt.figure()
        self.Te.plot(label="Te", color="red")
        plt.plot(te_kw0.rho_poloidal, te_kw0.values, marker="o", color="red")
        plt.plot(te_kw1.rho_poloidal, te_kw1.values, marker="x", color="red")
        plt.hlines(
            te_kw0.values,
            te_kw0.rho_poloidal - te_kw0.rho_poloidal_err["in"],
            te_kw0.rho_poloidal + te_kw0.rho_poloidal_err["out"],
            color="red",
        )
        plt.hlines(
            te_kw1.values,
            te_kw1.rho_poloidal - te_kw1.rho_poloidal_err["in"],
            te_kw1.rho_poloidal + te_kw1.rho_poloidal_err["out"],
            color="red",
        )
        self.Ti.plot(label="Ti", color="black")
        plt.plot(ti_w0.rho_poloidal, ti_w0.values, marker="o", color="black")
        plt.plot(ti_w1.rho_poloidal, ti_w1.values, marker="x", color="black")
        plt.hlines(
            ti_w0.values,
            ti_w0.rho_poloidal - ti_w0.rho_poloidal_err["in"],
            ti_w0.rho_poloidal + ti_w0.rho_poloidal_err["out"],
            color="black",
        )
        plt.hlines(
            ti_w1.values,
            ti_w1.rho_poloidal - ti_w1.rho_poloidal_err["in"],
            ti_w1.rho_poloidal + ti_w1.rho_poloidal_err["out"],
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
            scd[elem] = self.ADASReader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd[elem] = self.ADASReader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd[elem] = self.ADASReader.get_adf11("ccd", elem, adf11[elem]["ccd"])

            fract_abu[elem] = FractionalAbundance(scd[elem], acd[elem], CCD=ccd[elem],)

        self.adf11 = adf11
        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        self.fract_abu = fract_abu

    def set_pec_data(self, adf15: dict = None, marchuk=False):
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

        if marchuk:
            adf15_marchuk = pickle.load(open(MARCHUK, "rb"))
            adf15_marchuk = adf15_marchuk.rename(
                {"line name": "line_name", "el temp": "electron_temperature"}
            )
            adf15_marchuk *= 1.0e-6  # cm**3 --> m**3
            Te = adf15_marchuk.electron_temperature.values
            dTe = Te[1] - Te[0]
            Te = np.append(Te, np.arange(Te[-1] + dTe, 10.0e3, dTe))
            extrap = adf15_marchuk.interp(electron_temperature=Te)

            plt.figure()
            for line in extrap.line_name:
                x = adf15_marchuk.electron_temperature.values
                y = adf15_marchuk.sel(line_name=line).values
                func = interp1d(
                    np.log(x), np.log(y), fill_value="extrapolate", kind="quadratic"
                )
                extrap.loc[dict(line_name=line)] = np.exp(func(np.log(Te)))
                extrap = xr.where(extrap < 1.0e-21, 1.0e-21, extrap)
                extrap.sel(line_name=line).plot(label=line.values)

                ylim = plt.ylim()
                plt.vlines(
                    adf15_marchuk.electron_temperature.max(),
                    ylim[0],
                    ylim[1],
                    color="black",
                )
                plt.title("Marchuck PECs extrapolated")
                plt.xlabel("Te (eV)")
                plt.ylabel("(m$^3$)")
                plt.yscale("log")

                plt.legend()

            adf15_marchuk = extrap

        adf15_data = {}
        pec = deepcopy(adf15)
        elements = []
        for line in adf15.keys():
            element = adf15[line]["element"]
            transition = adf15[line]["transition"]
            wavelength = adf15[line]["wavelength"]
            charge, filetype, year = adf15[line]["file"]

            identifier = f"{element}_{charge}_{filetype}_{year}"
            if identifier not in adf15_data.keys():
                adf15_data[identifier] = self.ADASReader.get_adf15(
                    element, charge, filetype, year=year
                )

            pec[line]["emiss_coeff"] = select_transition(
                adf15_data[identifier], transition, wavelength
            )

            elements.append(element)

        if marchuk:
            print("Using Marchukc PECs, only EXCITATION!")

            el_dens = np.array([1.0e17, 1.0e18, 1.0e19, 1.0e20, 1.0e21, 1.0e22, 1.0e23])
            adf15_new, pec_new = {}, {}
            lines_new = np.unique([line.split("_")[0] for line in adf15_marchuk.line_name.values])
            for k in lines_new:
                adf15_new[k] = deepcopy(adf15[line])
                adf15_new[k].pop("transition")
                adf15_new[k]["file"] = MARCHUK

                pec_new[k] = deepcopy(pec[line])
                pec_new[k].pop("transition")
                pec_new[k]["file"] = MARCHUK

                if k[0] != "w":
                    emiss_coeff = adf15_marchuk.sel(line_name=k, drop=True)
                else:
                    type = ["excit", "recom"]
                    excit = adf15_marchuk.sel(line_name="w_exc", drop=True)
                    adas = pec["w"]["emiss_coeff"].sel(electron_density=1.e19, method="nearest", drop=True)
                    excit = adas.interp(electron_temperature=excit.electron_temperature)
                    excit.plot(color="black")

                    recom = adf15_marchuk.sel(line_name="w_rec", drop=True)
                    emiss_coeff = (
                        xr.concat([excit, recom], "index")
                        .assign_coords(index=[0, 1])
                        .assign_coords(type=("index", type))
                    )

                emiss_coeff = emiss_coeff.expand_dims({"electron_density": el_dens})
                pec_new[k]["emiss_coeff"] = emiss_coeff

            pec = pec_new
            adf15 = adf15_new

        self.adf15 = adf15
        self.pec = pec
        self.elements = elements

    def radiation_characteristics(
        self, Te, Ne, Nimp: dict = None, Nh=None, tau=None, recom=False,
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
        recom
            Set to True if recombination emission is to be included

        Returns
        -------

        """

        if Nh is None:
            Nh = xr.full_like(Ne, 0.0)
        self.Ne = Ne
        self.Nh = Nh
        if Nimp is not None:
            self.Nimp = Nimp
        self.Te = Te

        fz = {}
        emiss = {}
        for line, pec in self.pec.items():
            elem = pec["element"]
            charge = pec["charge"]
            wavelength = pec["wavelength"]
            coords = pec["emiss_coeff"].coords

            if elem not in fz.keys():
                _fz = self.fract_abu[elem](Ne, Te, Nh, tau=tau)
                fz[elem] = xr.where(_fz >= 0, _fz, 0)

            if "index" in coords:
                emiss_coeff = interp_pec(
                    select_type(pec["emiss_coeff"], type="excit"), Ne, Te
                )
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

                if recom is True and "recom" in pec.type:
                    emiss_coeff = interp_pec(
                        select_type(pec["emiss_coeff"], type="recom"), Ne, Te
                    )
                    _emiss += emiss_coeff * fz[elem].sel(ion_charges=charge + 1)
            else:
                emiss_coeff = interp_pec(pec["emiss_coeff"], Ne, Te)
                _emiss = emiss_coeff * fz[elem].sel(ion_charges=charge)

            _emiss *= Ne
            if Nimp is not None:
                _emiss *= Nimp[elem]
            else:
                _emiss *= Ne

            ev_wavelength = constants.e * 1.239842e3 / (wavelength)
            emiss[line] = xr.where(_emiss >= 0, _emiss, 0) * ev_wavelength

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
        intensity, emiss = self.los_integral(rho_los, dl)

        rho_tmp = rho_los.values
        rho_min = np.min(rho_tmp)

        for line in self.pec.keys():
            # Ion temperature and position of emissivity
            x = np.array(range(len(emiss[line])))
            y = emiss[line]
            avrg, dlo, dhi, ind_in, ind_out = calc_moments(y, x, simmetry=False)
            pos = rho_tmp[int(avrg)]
            pos_err_in = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg - dlo)])
            if pos <= rho_min:
                pos = rho_min
                pos_err_in = 0.0
            if (pos_err_in > pos) and (pos_err_in > (pos - rho_min)):
                pos_err_in = pos - rho_min
            pos_err_out = np.abs(rho_tmp[int(avrg)] - rho_tmp[int(avrg + dhi)])

            x = emiss[line]
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
            data[f"{line}_int"] = DataArray(
                [intensity[line]], coords=[(coord, [pos])], attrs=attrs
            )

        # Electron temperature(s) and position
        if use_satellites:
            emiss_shell = {
                "te_kw": emiss["w"] * emiss["k"],
                "te_n3w": emiss["w"] * emiss["n3"],
            }
        else:
            emiss_shell = {"te": emiss["w"]}

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
