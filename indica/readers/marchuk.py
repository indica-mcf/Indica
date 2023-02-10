from copy import deepcopy
import os
import indica

import numpy as np
import xarray as xr

# Constants
RY = 13.605  # eV
PERCMTOEV = 1.239841e-4  # Convert 1/cm to eV

head = os.path.dirname(indica.__file__)
FILEHEAD = os.path.join(head, "data/Data_Argon/")

def diel_calc(atomic_data: np.typing.ArrayLike, Te: xr.DataArray, label: str = "he"):
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
    a0 = 5.29e-9  # bohr radius / cm
    Te = Te / RY
    Es = atomic_data[:, 1] * PERCMTOEV / RY  # Ry
    F2 = atomic_data[:, 4] * 1e13  # 1/s

    if label == "he":
        g0 = 1

    elif label == "li":
        g0 = 2
        Esli = 1 / (atomic_data[:, 0] * 1e-8) - atomic_data[:, 1]
        # Difference between energy levels
        Es = Esli * PERCMTOEV / RY
    else:
        raise ValueError(f"wrong label given: {label}")

    intensity = 1 / g0 * 4 * np.pi ** (3 / 2) * a0**3 / Te[:, None] ** (3 / 2) \
        * F2[None,] * np.exp(-Es[None,] / Te[:, None])
    return intensity


class MARCHUKReader:
    """
    Class for interacting with Marchuks data format and return PECs in format:
    Dataset of dims(line name, electron temperature) and data variables of emission type
    """

    def __init__(
        self,
        extrapolate: bool = True,
        filehead: str = None,
        element: str = "ar",
        charge: int = 16,
    ):

        if filehead is None:
            filehead = FILEHEAD
        self.filehead = filehead
        self.extrapolate = extrapolate
        self.element = element
        self.charge = charge

        self.pec_rawdata = self.build_pec_database()
        self.pecs = self.make_dataset()

    def build_pec_database(
        self,
        Te: np.typing.ArrayLike = np.linspace(200, 10000, 10000),
    ):
        """
        Reads Marchuk's Atomic data and builds DataArrays for each emission type
        with co-ordinates of line label and electron temperature
        Input is the Te vector for dielectronic recombination data
        """

        lines_main = ["W", "X", "Y", "Z"]
        lines_ise = ["q", "r", "s", "t", "u", "v", "w"]
        lines_isi = ["Z"]
        lines_casc = ["q", "r", "s", "t"]

        # Wavelengths from the paper from O. Marchuk
        # "Modelling of helium like spectra at TEXTOR and TORE SUPRA"
        w0 = 0.39492
        x0 = 0.39660
        y0 = 0.39695
        z0 = 0.39943
        q0 = 0.39815
        r0 = 0.39836
        s0 = 0.39678
        t0 = 0.39687
        u0 = 1  # Ignore
        v0 = 1  # Ignore

        wavelengths_main = np.array([w0, x0, y0, z0])
        wavelengths_ise = np.array([q0, r0, s0, t0, u0, v0, v0])
        wavelengths_isi = np.array([z0])
        wavelengths_casc = np.array([q0, r0, s0, t0])

        # Read in atomic data
        # Excitation / Recombination / Charge Exchange /
        # Inner-Shell Excitation / Inner-Shell Ionisation / Cascades
        # Missing ion - ion excitation (not relevant till ne>>1E20 1/m3)

        # exc = np.loadtxt(head + "DirectRates.dat", skiprows=1)
        conversion_factor = 1e-6  # cm^3 -> m^3
        exc = np.loadtxt(self.filehead + "WXYZ_R-matrix.txt", comments="#")
        recom = np.loadtxt(self.filehead + "RecombRates.dat", skiprows=1)
        cxr = np.loadtxt(self.filehead + "ChargeRates.dat", skiprows=1)
        ise = np.loadtxt(self.filehead + "LiCollSatt.dat", skiprows=1)
        isi = np.loadtxt(self.filehead + "InnerRates.dat", skiprows=1)
        casc = np.loadtxt(self.filehead + "Cascade.dat", skiprows=5)

        # Dielectronic recombination / wavelength; Es; Ar; Aa; F2; Satellites
        n2 = np.loadtxt(self.filehead + "n2dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n3 = np.loadtxt(self.filehead + "n3dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n4 = np.loadtxt(self.filehead + "n4dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        n5 = np.loadtxt(self.filehead + "n5dielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        lin2 = np.loadtxt(self.filehead + "n2lidielsat.dat", skiprows=1, usecols=(0, 1, 2, 3, 4, 5))
        # Use line labels from file
        lines_n2 = np.genfromtxt(self.filehead + "n2dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n3 = np.genfromtxt(self.filehead + "n3dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n4 = np.genfromtxt(self.filehead + "n4dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_n5 = np.genfromtxt(self.filehead + "n5dielsat.dat", skip_header=1, usecols=(6), dtype="str")
        lines_lin2 = np.genfromtxt(self.filehead + "n2lidielsat.dat", skip_header=1, usecols=(6), dtype="str")

        # replace duplicate line "name" with "name"+ str(n)
        line_names = lines_n2.tolist()
        lines_n2 = [v + str(line_names[:i].count(v) + 1) if line_names.count(v) > 1 else v for i, v in
                    enumerate(line_names)]
        line_names = lines_n3.tolist()
        lines_n3 = [v + str(line_names[:i].count(v) + 1) if line_names.count(v) > 1 else v for i, v in
                    enumerate(line_names)]
        line_names = lines_n4.tolist()
        lines_n4 = [v + str(line_names[:i].count(v) + 1) if line_names.count(v) > 1 else v for i, v in
                    enumerate(line_names)]
        line_names = lines_n5.tolist()
        lines_n5 = [v + str(line_names[:i].count(v) + 1) if line_names.count(v) > 1 else v for i, v in
                    enumerate(line_names)]
        line_names = lines_lin2.tolist()
        lines_lin2 = [v + str(line_names[:i].count(v) + 1) if line_names.count(v) > 1 else v for i, v in
                    enumerate(line_names)]

        rates_n2 = diel_calc(n2, Te)
        rates_n3 = diel_calc(n3, Te)
        rates_n4 = diel_calc(n4, Te)
        rates_n5 = diel_calc(n5, Te)
        rates_lin2 = diel_calc(lin2, Te, label="li")

        # Convert data to DataArrays
        exc_array = xr.DataArray(
            data=exc[:, 1:, np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": exc[:, 0] * 1e3,
                "line_name": lines_main,
                "type": ["excit"],
                "wavelength": (("line_name",), wavelengths_main,),
            },
            dims=["electron_temperature", "line_name", "type"],
        )
        recom_array = xr.DataArray(
            data=recom[:, 1:, np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": recom[:, 0] * 1e3,
                "line_name": lines_main,
                "type": ["recom"],
                "wavelength": ("line_name", wavelengths_main,),
            },
            dims=["electron_temperature", "line_name", "type"],
        )
        cxr_array = xr.DataArray(data=cxr[:, 1:5, np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": cxr[:, 0] * 1e3,
                "line_name": lines_main,
                "type": ["cxr"],
                "wavelength": ("line_name", wavelengths_main),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        ise_array = xr.DataArray(
            data=ise[:, 1:, np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": ise[:, 0] * 1e3,
                "line_name": lines_ise,
                "type": ["ise"],
                "wavelength": ("line_name", wavelengths_ise),
            },
            dims=["electron_temperature", "line_name", "type"],
        )
        isi_array = xr.DataArray(
            data=isi[:, 1:, np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": isi[:, 0] * 1e3,
                "line_name": lines_isi,
                "type": ["isi"],
                "wavelength": ("line_name", wavelengths_isi),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        casc_factor_array = xr.DataArray(
            data=casc[:, 1:, np.newaxis],
            coords={
                "electron_temperature": casc[:, 0, ] * 1e3,
                "line_name": lines_casc,
                "type": ["diel"],
                "wavelength": ("line_name", wavelengths_casc),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        n2_array = xr.DataArray(
            data=rates_n2[:,:,np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": Te,
                "line_name": lines_n2,
                "type": ["diel"],
                "wavelength": ("line_name", n2[:, 0] * 0.1),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        n3_array = xr.DataArray(
            data=rates_n3[:,:,np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": Te,
                "line_name": lines_n3,
                "type": ["diel"],
                "wavelength": ("line_name", n3[:, 0] * 0.1 ),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        n4_array = xr.DataArray(
            data=rates_n4[:,:,np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": Te,
                "line_name": lines_n4,
                "type": ["diel"],
                "wavelength": ("line_name", n4[:, 0] * 0.1),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        n5_array = xr.DataArray(
            data=rates_n5[:,:,np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": Te,
                "line_name": lines_n5,
                "type": ["diel"],
                "wavelength": ("line_name", n5[:, 0] * 0.1),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        lin2_array = xr.DataArray(
            data=rates_lin2[:,:,np.newaxis] * conversion_factor,
            coords={
                "electron_temperature": Te,
                "line_name": lines_lin2,
                "type": ["li_diel"],
                "wavelength": ("line_name",lin2[:, 0] * 0.1),
            },
            dims=["electron_temperature", "line_name", "type"],
        )

        # Cascade functions
        casc_factor = casc_factor_array.interp(
            electron_temperature=Te, method="quadratic"
        )

        q_idx = n2_array.line_name.str.contains("q!").values
        r_idx = n2_array.line_name.str.contains("r!").values
        s_idx = n2_array.line_name.str.contains("s!").values
        t_idx = n2_array.line_name.str.contains("t!").values
        casc_idx = q_idx | r_idx | s_idx | t_idx
        n2_array.loc[dict(line_name=casc_idx)] = n2_array.sel(line_name=casc_idx) * (1 + casc_factor.values)

        # Atomic data
        pec_database = dict(
            excit=exc_array,
            recom=recom_array,
            cxr=cxr_array,
            ise=ise_array,
            isi=isi_array,
            n2=n2_array,
            n3=n3_array,
            n4=n4_array,
            n5=n5_array,
            li_n2=lin2_array,
        )
        return pec_database


    def _interp_pecs(self, Te: np.typing.ArrayLike = np.linspace(200, 10000, 10000)):
        _interp_pec = {}
        for _pec_name, _pec in self.pec_rawdata.items():
            _interp_pec[_pec_name] = _pec.interp(electron_temperature=Te, kwargs={"fill_value": "extrapolate"})
        return _interp_pec

    def make_dataset(self):
        """
        TODO: add caching of dataset
        Returns
        -------
        """
        _pecs = self._interp_pecs()
        _dataset = {}
        for _pec_name, _pec in _pecs.items():
            _dataset[_pec_name] = _pec.to_dataset(dim="type")
        _dataset = xr.merge([*_dataset.values()])
        _dataset = _dataset.to_array(dim="type")
        _dataset = _dataset.transpose(*["electron_temperature", "line_name", "type"])
        dataset = {
                "element": self.element,
                "file": self.filehead,
                "charge": self.charge,
                "emiss_coeff":_dataset,
        }
        return dataset
