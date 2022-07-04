"""
Purpose of this code is to generate class for helike spectrum from Marchuk's atomic data

methods including excitation, recombination, CX, cascades, dielectronic recombination, inner shell collisions, ion-ion collisions
etc.

"""

from hda.profiles import Profiles
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
import xarray as xr

from indica.operators.atomic_data import FractionalAbundance
from indica.readers import ADASReader


def diel_calc(atomic_data, Te, label: str = "he"):
    """
    Calculates intensity of dielectronic recombination

    Parameters
    ----------
    atomic_data
        array of atomic data read from Marchuk's database
    Te
        electron temperature (eV)
    label
        "he" for helium like collision or "li" for lithium like inner collision

    Returns
    -------
    Intensity along Te vector
    """
    a0 = 5.29E-9  # bohr radius / cm
    Te = Te / Ry
    Es = atomic_data[:, 1] * percmtoeV / Ry  # Ry
    F2 = atomic_data[:, 4] * 1e13  # 1/sfrom hda.profiles import Profiles

    if label == "he":
        g0 = 1

    elif label == "li":
        g0 = 2
        Esli = 1 / (atomic_data[:, 0] * 1e-8) - atomic_data[:, 1]
        # Difference between energy levels
        Es = Esli * percmtoeV / Ry
    else:
        return None
    I = (1 / g0) * ((4 * np.pi ** (3 / 2) * a0 ** 3) / Te[:, None] ** (3 / 2)) * F2[None,] * np.exp(
        -(Es[None,] / Te[:, None]))

    return I
    background = 0


# Constants
Ry = 13.605  # eV
percmtoeV = 1.239841E-4  # Convert 1/cm to eV
Mi = 39.948

# Keys for ADF11
ADF11 = {"ar": {"scd": "89", "acd": "89", "ccd": "89"}}


class CrystalSpectrometer:
    """
    Class for the crystal spectrometer which generates a database of the line intensities from atomic data,
    and when given temperature and density profiles makes the xray spectrum.

    Parameters
    ----------
    window
        wavelength vector to build the spectrum on
    int_cal
        intensity calibration
    ADASReader
        ADASReader class to read atomic data

    Examples
    ---------
    spec = Crystal_Spectrometer()
    spec.test_workflow()

    or:

    spec.intensity = spec.make_intensity(spec.database_offset, el_temp=Te, el_dens=Ne, fract_abu=fz, Ar_dens=NAr,
                                             H_dens=Nh, int_cal=int_cal)
    spec.spectra = spec.make_spectra(spec.intensity, Ti, background)
    spec.plot_spectrum(spec.spectra)
    """

    def __init__(self,
                 window: np.typing.ArrayLike = np.linspace(.394, .401, 1000),
                 int_cal: float = 1e-30,
                 ADASReader=ADASReader
                 # etendue / instrument function / geometry
                 # atomic data / build_database
                 ):

        # spectrometer specific
        self.window = window
        self.int_cal = int_cal
        self.adasreader = ADASReader()

        # constants
        self.Ry = 13.605  # eV
        self.percmtoeV = 1.239841E-4  # Convert 1/cm to eV
        self.Mi = 39.948

        self.database = self.build_database()
        self.database_offset = self.wavelength_offset(self.database, offset=2e-5)

        self.set_ion_data(ADF11)

    def build_database(self, Te=np.linspace(200, 10000, 1000)):
        """
        Reads Marchuks Atomic data and builds DataArrays for each emission type
        with co-ordinates line label and electron temperature

        Input is the Te vector for dielectronic recombination data
        """
        head = "/data/st40/atomic_data/helike_crystal/Marchuk_data/"

        lines_main = ["W", "X", "Y", "Z"]
        lines_ise = ["q", "r", "s", "t", "u", "v", "w"]
        lines_isi = ["Z"]
        lines_casc = ["q", "r", "s", "t"]

        # Wavelengths from "Modelling of helium like spectra at TEXTOR and TORE SUPRA" (Marchuk's Thesis)
        w0 = .39492
        x0 = .39660
        y0 = .39695
        z0 = .39943
        q0 = .39815
        r0 = .39836
        s0 = .39678
        t0 = .39687
        u0 = 1  # Ignore
        v0 = 1  # Ignore

        wavelengths_main = np.array([w0, x0, y0, z0])
        wavelengths_ise = np.array([q0, r0, s0, t0, u0, v0, v0])
        wavelengths_isi = np.array([z0])
        wavelengths_casc = np.array([q0, r0, s0, t0])

        # Read in atomic data
        # Excitation / Recombination / Charge Exchange / Inner-Shell Excitation / Inner-Shell Ionisation / Cascades
        # Missing ion - ion excitation (not relevant till ne>>1E20 1/m3)

        # exc = np.loadtxt(head + "DirectRates.dat", skiprows=1)
        exc = np.loadtxt(head + "WXYZ_R-matrix.txt", comments="#")
        recom = np.loadtxt(head + "RecombRates.dat", skiprows=1)
        cxr = np.loadtxt(head + "ChargeRates.dat", skiprows=1)
        ise = np.loadtxt(head + "LiCollSatt.dat", skiprows=1)
        isi = np.loadtxt(head + "InnerRates.dat", skiprows=1)
        casc = np.loadtxt(head + "Cascade.dat", skiprows=5)

        # Dielectronic recombination / wavelength; Es; Ar; Aa; F2; Satellites
        n2 = np.loadtxt(head + "n2dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n3 = np.loadtxt(head + "n3dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n4 = np.loadtxt(head + "n4dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n5 = np.loadtxt(head + "n5dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        lin2 = np.loadtxt(head + "n2lidielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        # Use line labels from file
        lines_n2 = np.genfromtxt(head + "n2dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n3 = np.genfromtxt(head + "n3dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n4 = np.genfromtxt(head + "n4dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n5 = np.genfromtxt(head + "n5dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_lin2 = np.genfromtxt(head + "n2lidielsat.dat", skip_header=1, usecols=(6), dtype="str")

        rates_n2 = diel_calc(n2, Te)
        rates_n3 = diel_calc(n3, Te)
        rates_n4 = diel_calc(n4, Te)
        rates_n5 = diel_calc(n5, Te)
        rates_lin2 = diel_calc(lin2, Te, label="li")

        # Convert data to DataArrays
        exc_array = xr.DataArray(data=exc[:, 1:],
                                 coords={"el_temp": exc[:, 0] * 1e3, "line_name": lines_main,
                                         "type": "EXC",
                                         "wavelength": (("el_temp", "line_name"), wavelengths_main[None, :] * np.ones(
                                             shape=(len(exc[:, 1]), len(wavelengths_main))))},
                                 dims=["el_temp", "line_name"])
        recom_array = xr.DataArray(data=recom[:, 1:],
                                   coords={"el_temp": recom[:, 0] * 1e3, "line_name": lines_main,
                                           "type": "REC",
                                           "wavelength": (("el_temp", "line_name"), wavelengths_main[None, :] * np.ones(
                                               shape=(len(recom[:, 1]), len(wavelengths_main))))},
                                   dims=["el_temp", "line_name"])
        cxr_array = xr.DataArray(data=cxr[:, 1:5],
                                 coords={"el_temp": cxr[:, 0] * 1e3, "line_name": lines_main,
                                         "type": "CXR",
                                         "wavelength": (("el_temp", "line_name"), wavelengths_main[None, :] * np.ones(
                                             shape=(len(cxr[:, 1]), len(wavelengths_main))))},
                                 dims=["el_temp", "line_name"])

        ise_array = xr.DataArray(data=ise[:, 1:],
                                 coords={"el_temp": ise[:, 0] * 1e3, "line_name": lines_ise,
                                         "type": "ISE",
                                         "wavelength": (("el_temp", "line_name"), wavelengths_ise[None, :] * np.ones(
                                             shape=(len(ise[:, 1]), len(wavelengths_ise))))},
                                 dims=["el_temp", "line_name"])
        isi_array = xr.DataArray(data=isi[:, 1:],
                                 coords={"el_temp": isi[:, 0] * 1e3, "line_name": lines_isi,
                                         "type": "ISI",
                                         "wavelength": (("el_temp", "line_name"), wavelengths_isi[None, :] * np.ones(
                                             shape=(len(isi[:, 1]), len(wavelengths_isi))))},
                                 dims=["el_temp", "line_name"])

        casc_factor_array = xr.DataArray(data=casc[:, 1:],
                                         coords={"el_temp": casc[:, 0] * 1e3, "line_name": lines_casc,
                                                 "type": "N2CASC",
                                                 "wavelength": (
                                                     ("el_temp", "line_name"), wavelengths_casc[None, :] * np.ones(
                                                         shape=(len(casc[:, 1]), len(wavelengths_casc))))},
                                         dims=["el_temp", "line_name"])

        n2_array = xr.DataArray(data=rates_n2,
                                coords={"el_temp": Te, "line_name": lines_n2,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n2[:, 0] * 0.1 * np.ones(
                                            shape=(len(Te), len(n2[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n3_array = xr.DataArray(data=rates_n3,
                                coords={"el_temp": Te, "line_name": lines_n3,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n3[:, 0] * 0.1 * np.ones(
                                            shape=(len(Te), len(n3[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n4_array = xr.DataArray(data=rates_n4,
                                coords={"el_temp": Te, "line_name": lines_n4,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n4[:, 0] * 0.1 * np.ones(
                                            shape=(len(Te), len(n4[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n5_array = xr.DataArray(data=rates_n5,
                                coords={"el_temp": Te, "line_name": lines_n5,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n5[:, 0] * 0.1 * np.ones(
                                            shape=(len(Te), len(n5[:, 0]))))},
                                dims=["el_temp", "line_name"])

        lin2_array = xr.DataArray(data=rates_lin2,
                                  coords={"el_temp": Te, "line_name": lines_lin2,
                                          "type": "LIDIREC",
                                          "wavelength": (("el_temp", "line_name"), lin2[:, 0] * 0.1 * np.ones(
                                              shape=(len(Te), len(lin2[:, 0]))))},
                                  dims=["el_temp", "line_name"])

        # Temporary !!! Adjust wavelengths for match with experiment
        # n3_array = n3_array.assign_coords(wavelength=(n3_array.wavelength + offset_n3))
        # n2_array = n2_array.assign_coords(wavelength=(n2_array.wavelength + offset_n2))
        #
        # a = xr.where((n2_array.wavelength>0.3985) & (n2_array.wavelength<0.3987), n2_array.wavelength-0.000025, n2_array.wavelength)
        # n2_array = n2_array.assign_coords(wavelength=(a))
        #
        # k = xr.where((n2_array.wavelength>0.3989) & (n2_array.wavelength<0.3991), n2_array.wavelength+0.00003, n2_array.wavelength)
        # n2_array = n2_array.assign_coords(wavelength=(k))

        # Cascade functions
        casc_factor = casc_factor_array.interp(el_temp=Te, method="quadratic")

        q_idx = n2_array.line_name.str.contains("q!").values
        r_idx = n2_array.line_name.str.contains("r!").values
        s_idx = n2_array.line_name.str.contains("s!").values
        t_idx = n2_array.line_name.str.contains("t!").values

        q_casc = n2_array.sel(line_name=q_idx) * casc_factor[:, 0]
        r_casc = n2_array.sel(line_name=r_idx) * casc_factor[:, 1]
        s_casc = n2_array.sel(line_name=s_idx) * casc_factor[:, 2]
        t_casc = n2_array.sel(line_name=t_idx) * casc_factor[:, 3]
        casc_array = xr.concat([q_casc, r_casc, s_casc, t_casc], dim="line_name")
        casc_array["type"] = "N2CASC"
        casc_array["wavelength"] = (("el_temp", "line_name"), wavelengths_casc[None, :] * np.ones(
            shape=(len(Te), len(wavelengths_casc))))

        # Atomic data
        database = dict(EXC=exc_array, REC=recom_array, CXR=cxr_array, ISE=ise_array,
                        ISI=isi_array, N2=n2_array, N3=n3_array, N4=n4_array,
                        N5=n5_array, LIN2=lin2_array, N2CASC=casc_array)
        LOG = "Finished building database"
        print(LOG)
        return database

    def wavelength_offset(self, database, offset):
        # constant offset applied to wavelengths
        database_offset = {}
        for key, item in database.items():
            item = item.assign_coords(wavelength=(item.wavelength - offset))
            database_offset[key] = item

        return database_offset

    def make_intensity(self, database, el_temp=None, el_dens=None, fract_abu=None,
                       Ar_dens=None, H_dens=None, int_cal=None):

        """
        Uses the intensity recipes to get intensity from Te/ne/f_abundance/H_dens/calibration_factor
        and atomic data at each spatial point.
        Returns DataArrays of emission type with co-ordinates of line label and rho co-ordinate
        # """

        intensity = {}
        for key, value in database.items():
            if value.type == "EXC":
                I = value.interp(el_temp=el_temp) * fract_abu[16,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "REC":
                # Truncate to max value at 4keV
                el_temp = el_temp.where(el_temp < 4000, 4000)
                I = value.interp(el_temp=el_temp) * fract_abu[17,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "CXR":
                el_temp = el_temp.where(el_temp < 4000, 4000)
                I = value.interp(el_temp=el_temp) * fract_abu[17,] * \
                    Ar_dens * H_dens * int_cal
            elif value.type == "ISE":
                I = value.interp(el_temp=el_temp) * fract_abu[15,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "ISI":
                I = value.interp(el_temp=el_temp) * fract_abu[15,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "DIREC":
                I = value.interp(el_temp=el_temp) * fract_abu[16,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "N2CASC":
                I = value.interp(el_temp=el_temp) * fract_abu[16,] * \
                    Ar_dens * el_dens * int_cal
            elif value.type == "LIDIREC":
                I = value.interp(el_temp=el_temp) * fract_abu[15,] * \
                    Ar_dens * el_dens * int_cal
            else:
                print("Wrong Emission Type")
            intensity[key] = I

        print("Generated intensity profile")
        return intensity

    def make_spectra(self, intensity: dict, ion_temp: xr.DataArray, background=0):

        # Add convolution of signals as wrapper
        # -> G(x, mu1, sig1) * G(x, mu2, sig2) = G(x, mu1+mu2, sig1**2 + sig2**2)
        # instrument function / photon noise / photon -> counts
        # Background Noise
        def gaussian(x, integral, center, sigma):
            return integral / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

        def doppler_broaden(i):
            sigma = np.sqrt(constants.e / (Mi * constants.proton_mass * constants.c ** 2) * ion_temp) * i.wavelength
            return gaussian(self.window[:, None, None], i, i.wavelength.expand_dims(dict(window=self.window)),
                            sigma.expand_dims(dict(window=self.window)))

        spectra = {}
        spectra["total"] = 0
        for key, value in intensity.items():
            i = value.expand_dims(dict(window=self.window), 0)
            y = doppler_broaden(i)
            spectra[key] = y
            spectra["total"] = spectra["total"] + y.sum(["line_name"])

        spectra["total"] = spectra["total"].rename({"window": "wavelength"})
        spectra["total"] = spectra["total"].drop_vars(["type", "ion_charges"])
        spectra["background"] = self.window * 0 + background
        return spectra

    def plot_spectrum(self, spectra: dict):

        plt.figure()
        avoid = ["total", "background"]
        for key, value in spectra.items():
            if not any([x in key for x in avoid]):
                plt.plot(self.window, value.sum(["rho_poloidal", "line_name"]) + spectra["background"], label=key)

        plt.plot(self.window, spectra["total"].sum(["rho_poloidal"]) + spectra["background"], "k*", label="Total")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (AU)")
        plt.legend()
        plt.show(block=True)
        return

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

        scd, acd, ccd = {}, {}, {}
        for elem in adf11.keys():
            scd[elem] = self.adasreader.get_adf11("scd", elem, adf11[elem]["scd"])
            acd[elem] = self.adasreader.get_adf11("acd", elem, adf11[elem]["acd"])
            ccd[elem] = self.adasreader.get_adf11("ccd", elem, adf11[elem]["ccd"])

            fract_abu[elem] = FractionalAbundance(scd[elem], acd[elem], CCD=ccd[elem], )

        self.adf11 = adf11
        self.scd = scd
        self.acd = acd
        self.ccd = ccd
        self.fract_abu = fract_abu

    def test_workflow(self):

        # pulse = 25009780
        # tstart = 0.05
        # tend = 0.10
        # reader = ST40Reader(pulse, tstart, tend)
        # # spectra, dims = reader._get_data("hda", "run60", ":intensity", 0)

        Ne = Profiles(datatype=("density", "electron"))
        Ne.peaking = 1
        Ne.build_profile()
        Te = Profiles(datatype=("temperature", "electron"))
        Te.y0 = 2.0e3
        Te.y1 = 20
        Te.wped = 1
        Te.peaking = 4
        Te.build_profile()
        Ti = Profiles(datatype=("temperature", "ion"))
        Ti.y0 = 2.0e3
        Ti.y1 = 100
        Ti.wped = 1
        Ti.peaking = 5
        Ti.build_profile()

        Ne = Ne.yspl
        Te = Te.yspl
        Ti = Ti.yspl

        Nh_1 = 5.0e16
        Nh_0 = Nh_1 / 1000
        Nh = Ne.rho_poloidal ** 2 * (Nh_1 - Nh_0) + Nh_0
        NAr = Ne.rho_poloidal ** 2 * (1 / 100 * Ne) + (1 / 100 * Ne)

        tau = None
        # tau = 1.0e-3

        fz = self.fract_abu["ar"](Ne, Te, Nh, tau=tau)

        background = 250
        int_cal = 1.3e-27

        # self.database_offset = self.wavelength_offset(self.database, offset=1e-5)

        self.intensity = self.make_intensity(self.database_offset, el_temp=Te, el_dens=Ne, fract_abu=fz, Ar_dens=NAr,
                                             H_dens=Nh, int_cal=int_cal)
        self.spectra = self.make_spectra(self.intensity, Ti, background)
        self.plot_spectrum(self.spectra)
        return


if __name__ == "__main__":
    spec = Crystal_Spectrometer()
    spec.test_workflow()

    print()
