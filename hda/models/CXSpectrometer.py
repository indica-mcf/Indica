from copy import deepcopy

from hda.profiles import Profiles
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interpn
from xarray import DataArray

from indica.converters import FluxSurfaceCoordinates
from indica.converters.line_of_sight import LinesOfSightTransform
from indica.equilibrium import Equilibrium
from indica.models import indica2bbnbi
from indica.models import neutral_beam
from indica.operators.atomic_data import FractionalAbundance
from indica.readers import ADASReader
from indica.readers import ST40Reader

# import xarray as xr

# from indica.converters.lines_of_sight_jw import LinesOfSightTransform


ADF11 = {"c": {"scd": "96", "acd": "96", "ccd": "96"}}
# ADF12 = {}
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
    """ """

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
            Dictionary with details of ionisation balance data
            (see ADF11 class var)
        adf15
            Dictionary with details of photon emission coefficient data
            (see ADF15 class var)
        Returns
        -------
        """

        self.adasreader = ADASReader()
        self.name = name
        self.set_ion_data(adf11=adf11)
        # self.set_cx_data(adf12=adf12)
        # self.set_pec_data(adf15=adf15)

    def set_transform(self, transform: LinesOfSightTransform):
        """
        Set line-of sight transform to perform coordinate conversion and
        line-of-sight integrals
        Parameters
        ----------
        transform
            Line of sight transform
        Returns
        -------
        """
        self.transform = transform

    def test_flow(self, check_plots=True, run_bbnbi=False):
        """
        Test module with standard inputs
        """

        # Read ST40 data, from 9779 for Princeton spectrometer
        # st40_data = ST40data(pulse=9780, tstart=0.04, tend=0.085)
        # st40_data.get_princeton()
        # efit_pulse = None
        # efit_run = 0
        # equil_data = ST40data(pulse=10014, tstart=0.04, tend=0.085)
        # equil_data.get_all(sxr=False, efit_pulse=10014, efit_rev=1)
        # data = st40_data.data["princeton"]

        pulse = 9780
        tstart = 0.07
        tend = 0.10
        st40reader = ST40Reader(pulse, tstart, tend)

        # xrcs_revision = 0
        efit_revision = 0
        # xrcs = st40reader.get("sxr", "xrcs", xrcs_revision)
        efit = st40reader.get("", "efit", efit_revision)

        # Define LOS transform,
        location = np.array([1.077264, -0.36442098, 0.0], dtype=float)
        direction = np.array([-0.99663717, -0.08194118, 0.0], dtype=float)
        machine_dimensions = ((0.175, 0.8), (-0.6, 0.6))
        dl = 0.002
        los_transform = LinesOfSightTransform(
            location[0],
            location[1],
            location[2],
            direction[0],
            direction[1],
            direction[2],
            machine_dimensions=machine_dimensions,
            name="princeton",
            dl=dl,
        )
        # print(f'los_transform = {los_transform}')
        # print(f'los_transform.x_start = {los_transform.x_start}')
        # print(f'los_transform.x2 = {los_transform.x2}')
        # print(f'los_transform.dl = {los_transform.dl}')

        if check_plots:
            # Test methods
            index = DataArray(np.linspace(0, 1.0, 100, dtype=float))
            x_los, y_los = los_transform.convert_to_xy(0, index, 0)

            # ivc
            angles = np.linspace(0, 2 * np.pi, 1000)
            x_inner = machine_dimensions[0][0] * np.cos(angles)
            y_inner = machine_dimensions[0][0] * np.sin(angles)

            x_outer = machine_dimensions[0][1] * np.cos(angles)
            y_outer = machine_dimensions[0][1] * np.sin(angles)

            plt.figure()
            plt.plot(x_los, y_los, "b.-", label="LOS")
            plt.plot(x_inner, y_inner, "k", label="machine dimensions")
            plt.plot(x_outer, y_outer, "k")
            plt.legend()
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.show(block=False)

        # Load Equilibrium... use EFIT ST40 data, initialise equilibrium class,
        # initialise flux coord transform,
        # assign equilibrium class to coordinate transforms.
        equilibrium = Equilibrium(efit)
        flux_surface_coord = FluxSurfaceCoordinates("poloidal")
        flux_surface_coord.set_equilibrium(equilibrium)

        # plasma_obj = Plasma(
        #     tstart=0.02, tend=0.12, dt=0.01,
        #     machine_dimensions=machine_dimensions)
        # binned_data = plasma_obj.build_data(
        #     equil_data.data
        # )  # Must use equil_data.get_all()!!

        # Load Profiles... default profiles from profiles.py
        rho = np.linspace(0.0, 1.0, 99, dtype=float)
        te = Profiles(datatype=("temperature", "electron"), xspl=rho)
        # ti = Profiles(datatype=("temperature", "ion"), xspl=rho)
        # ne = Profiles(datatype=("density", "electron"), xspl=rho)
        # nimp = Profiles(datatype=("density", "impurity"), xspl=rho)
        # nh = Profiles(datatype=("neutral_density", "impurity"), xspl=rho)
        # vrot = Profiles(datatype=("rotation", "ion"), xspl=rho)
        zeff = deepcopy(te)
        zeff.y0 = 2.0
        zeff.y1 = 2.0
        zeff.wped = 0.0
        zeff.build_profile()

        # Run BBNBI to calculate beam density
        if run_bbnbi:
            pl = np.load(
                "/home/jari.varje/indica_bbnbi/pl.npy", allow_pickle=True
            ).item()
            time = 0.05
            indica2bbnbi.indica2bbnbi(
                time,
                equilibrium,
                pl.el_dens,
                pl.el_temp,
                pl.ion_dens,
                pl.ion_temp,
                neutral_beam.analytical_beam_defaults,
            )
            indica2bbnbi.run()  # bypass if temp.h5 exists
        hist, dims = indica2bbnbi.bbnbi2indica()

        # Extract data from bbnbi object -> make into DataArray variable
        r_b = dims[0]
        phi_b = dims[1] * np.pi / 180
        z_b = dims[2]
        n_b = DataArray(
            data=hist,
            dims=["R", "Phi", "Z"],
            coords=dict(
                R=(["R"], r_b),
                Phi=(["Phi"], phi_b),
                Z=(["Z"], z_b),
            ),
        )
        i_min = np.argmin(np.abs(z_b - 0.0))

        # Find beam density along line of sight chord
        r_los = los_transform.R
        phi_los = los_transform.phi
        z_los = los_transform.z
        nb_los = np.zeros(len(r_los))
        for i_los in range(len(r_los)):
            r_here = r_los[i_los]
            phi_here = phi_los[i_los]
            if phi_here < 0.0:
                phi_here = (2 * np.pi) + phi_here
            z_here = z_los[i_los]
            try:
                nb_los[i_los] = interpn(
                    (r_b, phi_b, z_b),
                    n_b.data,
                    np.array([r_here, phi_here, z_here]),
                )
            except ValueError:
                nb_los[i_los] = np.nan

        x = np.linspace(-1.0, 1.0, 201)
        y = np.linspace(-1.0, 1.0, 201)
        nb_interp = np.zeros((len(y), len(x)))
        for i in range(len(x)):
            print(f"i = {i}")
            for j in range(len(y)):
                r_here = np.sqrt(x[i] ** 2 + y[j] ** 2)
                phi_here = np.arctan2(y[j], x[i])
                if phi_here < 0.0:
                    phi_here = (2 * np.pi) + phi_here
                try:
                    nb_interp[j, i] = interpn(
                        (r_b, phi_b, z_b),
                        n_b.data,
                        np.array([r_here, phi_here, 0.0]),
                    )
                except ValueError:
                    nb_interp[j, i] = np.nan

        if check_plots:
            plt.figure()
            plt.pcolor(phi_b, r_b, hist[:, :, i_min])
            plt.xlabel("phi (rad)")
            plt.ylabel("r (m)")

            # Contour plot in ST40 reference
            # Test methods
            index = DataArray(np.linspace(0, 1.0, 100, dtype=float))
            x_los, y_los = los_transform.convert_to_xy(0, index, 0)

            # ivc
            angles = np.linspace(0, 2 * np.pi, 1000)
            x_inner = machine_dimensions[0][0] * np.cos(angles)
            y_inner = machine_dimensions[0][0] * np.sin(angles)

            x_outer = machine_dimensions[0][1] * np.cos(angles)
            y_outer = machine_dimensions[0][1] * np.sin(angles)

            plt.figure()
            plt.pcolor(x, y, nb_interp)
            plt.plot(x_inner, y_inner, "w")
            plt.plot(x_outer, y_outer, "w")
            plt.plot(x_los, y_los, "r")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")

            x_los = los_transform.x
            y_los = los_transform.y
            d_los = np.sqrt((x_los - x_los[0]) ** 2 + (y_los - y_los[0]) ** 2)
            plt.figure()
            plt.plot(d_los, nb_los)
            plt.xlabel("Distance along line of sight (m)")
            plt.ylabel("Neutral beam density ()")

            plt.show(block=True)

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

            fract_abu[elem] = FractionalAbundance(
                scd[elem],
                acd[elem],
                CCD=ccd[elem],
            )

        self.adf11 = adf11
        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        self.fract_abu = fract_abu


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
    print(f"pec = {pec}")
    print(f"type = {type}")
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
