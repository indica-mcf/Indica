"""
Purpose of this code is to generate class for helike spectrum from Marchuk's atomic data

methods including excitation, recombination, CX, cascades, dielectronic recombination, inner shell collisions, ion-ion collisions
etc.

"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import constants


def calculate_Te_kw(R):
    # Te(keV) = (4.1716913) + (-119.91436)*R + (2132.1279)*R^2 + (-21900.396)*R^3 + (138367.33)*R^4 + (-560282.2)*R^5 + (1480395.4)*R^6 + (-2537587.3)*R^7 + (2718292.6)*R^8 + (-1652671.4)*R^9 + (435199.6)*R^10
    # keV -> eV
    return 1e3 * (
            (4.1716913) + (-119.91436) * R + (2132.1279) * R ** 2 + (-21900.396) * R ** 3 + (138367.33) * R ** 4 + (
        -560282.2) * R ** 5 + (1480395.4) * R ** 6 + (-2537587.3) * R ** 7 + (2718292.6) * R ** 8 + (
                -1652671.4) * R ** 9 + (435199.6) * R ** 10)


def calculate_Te_Rosen(R):
    # A.S Rosen et. al 2014 (0.0223<x<0.2449) for n3 / w + n4 + n5
    return 1e3 * (0.1552 * (R ** (-0.7781)))


# dielectronic intensity conversion
def diel_calc(atomic_data, Te, label="he"):
    a0 = 5.29E-9  # bohr radius / cm
    Te = Te / Ry
    Es = atomic_data[:, 1] * percmtoeV / Ry  # Ry
    F2 = atomic_data[:, 4] * 1e13  # 1/s
    if label == "he":
        g0 = 1

    elif label == "li":
        g0 = 2
        Esli = 1 / (atomic_data[:, 0] * 1e-8) - atomic_data[:, 1]
        # Difference between energy levels
        Es = Esli * percmtoeV / Ry
    else:
        return None
    I = (1 / g0) * ((4 * np.pi ** (3 / 2) * a0 ** 3) / Te[:,None] ** (3 / 2)) * F2[None,] * np.exp(-(Es[None,] / Te[:,None]))

    return I


# Start with excitation and recombination

Ry = 13.605  # eV
percmtoeV = 1.239841E-4  # Convert 1/cm to eV
Mi = 39.948


class Crystal_Spectrometer:
    def __init__(self,

                 window=np.linspace(.394, .401, 1000),
                 intensity_calib=None
                 # etendue / instrument function / geometry
                 # atomic data / build_data
                 ):

        # spectrometer specific
        self.window = window
        self.intensity_calib = intensity_calib

        # constants
        self.Ry = 13.605  # eV
        self.percmtoeV = 1.239841E-4  # Convert 1/cm to eV
        self.Mi = 39.948

        self.database = self.build_database()

    def build_database(self, Te=np.linspace(200, 10000, 1000)):
        """
        Reads Marchuks Atomic data and builds DataArrays for each emission type
        with co-ordinates line label and electron temperature

        Only input is the Te co-ordinate for dielectronic recombination data
        """
        head = "./Data_Argon/"

        lines_main = ["W", "X", "Y", "Z"]
        lines_ise = ["q", "r", "s", "t", "u", "v", "w"]
        lines_isi = ["Z"]
        lines_casc = ["q", "r", "s", "t"]

        # wavelength -> nm
        w0 = .39492
        x0 = .39660
        y0 = .39695
        z0 = .39943
        q0 = .398150
        r0 = .398360
        s0 = .396780
        t0 = .396870
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
                                        "wavelength": (("el_temp", "line_name"), n2[:, 0]*0.1 * np.ones(
                                            shape=(len(Te), len(n2[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n3_array = xr.DataArray(data=rates_n3,
                                coords={"el_temp": Te, "line_name": lines_n3,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n3[:, 0]*0.1 * np.ones(
                                            shape=(len(Te), len(n3[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n4_array = xr.DataArray(data=rates_n4,
                                coords={"el_temp": Te, "line_name": lines_n4,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n4[:, 0]*0.1 * np.ones(
                                            shape=(len(Te), len(n4[:, 0]))))},
                                dims=["el_temp", "line_name"])

        n5_array = xr.DataArray(data=rates_n5,
                                coords={"el_temp": Te, "line_name": lines_n5,
                                        "type": "DIREC",
                                        "wavelength": (("el_temp", "line_name"), n5[:, 0]*0.1 * np.ones(
                                            shape=(len(Te), len(n5[:, 0]))))},
                                dims=["el_temp", "line_name"])

        lin2_array = xr.DataArray(data=rates_lin2,
                                  coords={"el_temp": Te, "line_name": lines_lin2,
                                          "type": "LIDIREC",
                                          "wavelength": (("el_temp", "line_name"), lin2[:, 0]*0.1 * np.ones(
                                              shape=(len(Te), len(lin2[:, 0]))))},
                                  dims=["el_temp", "line_name"])

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
                             ISI=isi_array, N2=n2_array,       N3=n3_array,   N4=n4_array,
                             N5=n5_array,   LIN2=lin2_array,   N2CASC=casc_array)
        LOG = "Finished building database"
        print(LOG)
        return database

    def make_intensity(self, el_temp=None, el_dens=None, frac_abund=None,
                       Ar_dens=None, neutrals=None, intensity_calib=None):

        """
        Uses the intensity recipes to get intensity from Te/ne/f_abundance/neutrals/calibration_factor
        and atomic data at each spatial point.
        Returns DataArrays of emission type with co-ordinates of line label and rho co-ordinate
        """
        if not hasattr(self, "database"):
            self.build_database()
            print("Building database")

        intensity = {}
        for key, value in self.database.items():
            if value.type == "EXC":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 16, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "REC":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 17, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "CXR":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 17, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * neutrals[:,None] * intensity_calib
            elif value.type == "ISE":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 15, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "ISI":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 15, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "DIREC":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 16, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "N2CASC":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 16, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            elif value.type == "LIDIREC":
                I = value.interp(el_temp = el_temp) * frac_abund[:, 15, None] * \
                    Ar_dens[:,None] * el_dens[:,None] * intensity_calib
            else:
                print("Wrong Emission Type")
            intensity[key] = I

        print("Generated intensity profile")
        return intensity

    def make_spectrum(self, intensity, ion_temp):

        # instrument function / photon noise / photon -> counts
        def gaussian(x, integral, center, sigma):
            return integral / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

        def doppler_broaden(i):
            sigma = np.sqrt(constants.e / (Mi * constants.proton_mass * constants.c ** 2) * ion_temp[:,None]) * i.wavelength
            return gaussian(self.window[:,None,None], i, i.wavelength.expand_dims(dict(window=self.window)), sigma.expand_dims(dict(window=self.window)))

        spectra = {}
        for key, value in intensity.items():
            i = value.expand_dims(dict(window=self.window), 0)
            y=doppler_broaden(i)
            spectra[key] = y
        return spectra

    def plot_spectrum(self, spectra):
        tot=0
        for key, value in spectra.items():
            tot=tot+value.sum(["el_temp", "line_name"])
            plt.plot(self.window, value.sum(["el_temp", "line_name"]), label=key)

        plt.plot(self.window, tot, "k*", label="Total")
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Intensity (AU)")
        plt.legend()
        plt.show()
        return


    def test_workflow(self):

        # Spectrometer Class
        rho = 10
        frac_abund = np.zeros((rho, 18))
        frac_abund[:,15] = np.linspace(0.49, 0.10, rho) # Li-like
        frac_abund[:,16] = np.linspace(0.5, 0.75, rho) # He-like
        frac_abund[:,17] = np.linspace(0.01, 0.10, rho) # H-like

        neutrals = 1e-6 * np.ones(rho)
        intensity_calib = 1

        el_temp = np.linspace(1000,3000,rho)
        ion_temp = np.linspace(1000,9000,rho)
        el_dens = np.linspace(1e19,1e20,rho)
        Ar_dens = el_dens * 1e-6

        self.intensity = self.make_intensity(el_temp=el_temp, el_dens=el_dens, frac_abund=frac_abund, Ar_dens=Ar_dens,
                                             neutrals=neutrals, intensity_calib=intensity_calib)
        self.spectrum = self.make_spectrum(self.intensity, ion_temp)
        self.plot_spectrum(self.spectrum)
        return

spec = Crystal_Spectrometer()
spec.test_workflow()

