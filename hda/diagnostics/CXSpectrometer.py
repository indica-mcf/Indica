from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
import pickle
from xarray import DataArray
from scipy import constants
from scipy.interpolate import interp1d

from indica.readers import ADASReader
from hda.read_st40 import ST40data
from indica.operators.atomic_data import FractionalAbundance
# from indica.converters import LinesOfSightTransform
from indica.converters.lines_of_sight_jw import LinesOfSightTransform

from hda.profiles import Profiles
from hda.plasma import Plasma

from indica.numpy_typing import ArrayLike


ADF11 = {"c": {"scd": "96", "acd": "96", "ccd": "96"}}
ADF15 = {
    "529": {
        "element": "c",
        "file": ("5", "bnd", "96"),
        "charge": 5,
        "transition": "n=8-n=7",
        "wavelength": 5292.7,
    }
}


class CXSpectrometer:
    """
    """

    def __init__(
        self,
        name="",
        adf11: dict = None,
        adf12: dict = None,
        adf15: dict = None,
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

        Returns
        -------

        """

        self.adasreader = ADASReader()
        self.name = name
        self.set_ion_data(adf11=adf11)
        self.set_pec_data(adf15=adf15)

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

        # Read ST40 data, from 9779 for Princeton spectrometer
        st40_data = ST40data(pulse=9779, tstart=0.02, tend=0.12)
        st40_data.get_princeton()
        st40_data.get_efit()
        data = st40_data.data["princeton"]
        efit_data = st40_data.data

        # Define LOS transform,
        # 1 LOS transform object for one diagnostic, fibre number is an attribute
        machine_dimensions = ((0.175, 0.8), (-0.6, 0.6))
        dl = 0.01
        los_transform = LinesOfSightTransform(
            data["location"],
            data["direction"],
            machine_dimensions=machine_dimensions,
            name='princeton',
            dl=dl
        )
        print(f'los_transform = {los_transform}')
        print(f'los_transform.x_start = {los_transform.x_start}')
        print(f'los_transform.x2 = {los_transform.x2}')
        print(f'los_transform.dl = {los_transform.dl}')

        if False:
            # Test methods
            i_fibre = 3
            x, y, z = los_transform.convert_to_xyz(i_fibre, 0, 0)
            R, Z = los_transform.convert_to_Rz(i_fibre, 0, 0)

            plt.figure()
            plt.plot(x, y, '.-')
            plt.title(f'fibre = {i_fibre+1}')

            plt.figure()
            plt.plot(R, Z, '.-')
            plt.title(f'fibre = {i_fibre+1}')

            plt.show(block=True)

        # Load Equilibrium... use EFIT ST40 data, initialise equilibrium class, initialise flux coord transform,
        # assign equilibrium class to coordinate transforms.
        plasma_obj = Plasma(tstart=0.02, tend=0.12, dt=0.01, machine_dimensions=machine_dimensions)
        plasma_obj.build_data(efit_data)
        print(plasma_obj)

        # Load Profiles... use interp method of DataArray, linear.
        # rho =
        # Te = Profiles(datatype=("temperature", "electron"), xspl=rho)
        # Ne = Profiles(datatype=("density", "electron"), xspl=rho)
        # Nimp = Profiles(datatype=("density", "impurity"), xspl=rho)
        # Vrot = Profiles(datatype=("rotation", "ion"), xspl=rho)

        # Load Beam... ??? After!

        # Spectrometer settings
        lambda0 = 529.059  # [nm]
        mi = 13.0107  # [amu]
        c = constants.speed_of_light
        Avogadro = constants.Avogadro
        kB = constants.k
        e = constants.e

        # Calculate passive emission
        wavelength = data["wavelength"]
        intensity = np.zeros_like(wavelength, dtype=float)
        for i in range(los_transform.n):

            # R, Z coordinates for forward model
            r, z = los_transform.convert_to_Rz(i, 0, 0)

            print(f'r = {r}')
            print('aa'**2)

            # Interpolate for magnetic flux coordinate

            # Interpolate for Te, Ti, ne, ni, zeff, velocity

            # Calculate Bremsstrahlung emission

            # Calculate Recombination and Excitation emission

            # Calculate Passive charge exchange emission

            # Calculate Active charge exchange emission, with beam model

        print('aa'**2)

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

        # plt.figure()
        #for line, pec in self.pec.items():
        #     print(f'line = {line}')
        #     print(f'pec = {pec}')
        #     for t in pec["emiss_coeff"].type:
        #         print(f't = {t}')
        #         select_type(pec["emiss_coeff"], type=t).sel(
        #             electron_density=5.0e19
        #         ).plot(label=f"{line} {t.values}")
        # plt.yscale("log")
        # plt.title("PEC")
        # plt.legend()
        # plt.show()

        #



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

    def set_pec_data(self, adf15: dict = None):
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
        pec = deepcopy(adf15)
        for line in adf15.keys():
            element = adf15[line]["element"]
            transition = adf15[line]["transition"]
            wavelength = adf15[line]["wavelength"]
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
        self, Te, Ne, Nimp: dict = None, Nh=None, tau=None, recom=False, fast=False,
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
                    if recom * (t == "recom") or t != "recom":
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
    print(f'pec = {pec}')
    print(f'type = {type}')
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
