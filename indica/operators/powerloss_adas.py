from typing import cast
from typing import List
import warnings

import matplotlib.pylab as plt
import numpy as np
from numpy.core.numeric import zeros_like
from pandas import DataFrame
import xarray as xr
from xarray import DataArray

from indica.configs.readers.adasconf import ADF11
from indica.operators import FractionalAbundanceAdas
from indica.readers.adas import ADASReader
from indica.utilities import DATA_PATH
from indica.utilities import format_dataarray
from indica.utilities import get_element_info
from indica.utilities import set_plot_colors


class PowerLoss:
    """Calculate the total power loss associated with a given impurity element

    plt
        line emission emissivity coefficients
    prb
        recombination and bremsstrahlung emissivity coefficients
    prc
        charge exchange emissivity coefficients
    """

    def __init__(
        self,
        element: str,
        plt_year: str = None,
        prb_year: str = None,
        prc_year: str = None,
        full_run: bool = False,
    ):
        warnings.warn("Interpolating on Te only!!!")

        self.full_run = full_run
        self.adas_reader = ADASReader()

        _element_info = get_element_info(element)
        self.element = element
        self.element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }

        self.plt_year = plt_year
        self.prb_year = prb_year
        self.prc_year = prc_year
        for key in ["plt", "prb", "prc"]:
            # Year
            if getattr(self, f"{key}_year") is None:
                _year = ADF11[element][key]
                setattr(self, f"{key}_year", _year)
            year = getattr(self, f"{key}_year")

            # Filename
            filename = f"{key}{year}_{element}.dat"
            setattr(self, f"{key}_file", filename)

            # Data
            data = self.adas_reader.get_adf11(key, self.element, year)
            setattr(self, key, data)

    def interpolate_power(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Interpolates the various powers based on inputted Ne and Te.

        Ne
            electron density profile
        Te
            electron temperature profile
        """

        self.Ne, self.Te = Ne, Te  # type: ignore
        plt_spec = self.plt.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        prb_spec = self.prb.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        if self.prc is not None:
            prc_spec = self.prc.interp(electron_temperature=Te, method="cubic").interp(
                electron_density=Ne, method="linear"
            )
        else:
            prc_spec = xr.full_like(plt_spec, 0.0)

        self.plt_spec, self.prc_spec, self.prb_spec = plt_spec, prc_spec, prb_spec
        self.nq = len(self.plt_spec.ion_charge) + 1
        self.ion_charge = np.linspace(0, self.nq - 1, self.nq)

        return plt_spec, prc_spec, prb_spec, self.nq

    def calculate_power_loss(
        self,
        Ne: DataArray,
        F_z_t: DataArray,
        Nn: DataArray = None,
    ):
        """Calculates total radiated power of all ionisation charges of a given
        impurity element.

        Ne
            electron density profile
        F_z_t
            fractional abundance of all ionisation charges of given element.
        Nn
            thermal neutral hydrogen density profile

        """
        if Nn is not None:
            if self.prc is None:
                raise ValueError(
                    "Nn (Thermal hydrogen density) cannot be given when \
                    prc (effective charge exchange power) at initialisation \
                    is None."
                )
        elif self.prc is not None:
            Nn = cast(DataArray, zeros_like(Ne))

        self.Ne, self.Nn = Ne, Nn  # type: ignore

        if F_z_t is not None:
            try:
                assert not np.iscomplexobj(F_z_t)
            except AssertionError:
                raise ValueError(
                    "Inputted F_z_t is a complex type or array of complex numbers, \
                        must be real"
                )
            self.F_z_t = F_z_t  # type: ignore
        elif self.F_z_t is None:
            raise ValueError("Please provide a valid F_z_t (Fractional Abundance).")

        self.coord = self.plt_spec.coords[
            [k for k in self.plt_spec.dims if k != "ion_charge"][0]
        ]
        self.dim = self.coord.dims[0]
        self.ncoord = len(self.coord)

        plt, prb, prc = self.plt_spec, self.prb_spec, self.prc_spec

        # cooling_factor = xr.full_like(self.plt_spec)

        cooling_factor = np.zeros((self.nq, self.ncoord))
        for icoord, coord_val in enumerate(self.coord.data):
            q = 0
            cooling_factor[q, icoord] = (plt * self.F_z_t).loc[
                {"ion_charge": q, self.dim: coord_val}
            ]

            for q in range(1, self.nq - 1):
                cooling_factor[q, icoord] = (
                    plt.loc[{"ion_charge": q, self.dim: coord_val}]
                    + (Nn / Ne * prc.sel(ion_charge=q - 1)).loc[{self.dim: coord_val}]
                    + prb.loc[{"ion_charge": q - 1, self.dim: coord_val}]
                ) * self.F_z_t.loc[{"ion_charge": q, self.dim: coord_val}]

            q = self.nq - 1
            cooling_factor[q, icoord] = (
                (Nn / Ne * prc.sel(ion_charge=q - 1)).loc[{self.dim: coord_val}]
                + prb.loc[{"ion_charge": q - 1, self.dim: coord_val}]
            ) * self.F_z_t.loc[{"ion_charge": q, self.dim: coord_val}]

        coords = {"ion_charge": self.ion_charge, self.dim: self.coord}
        self.cooling_factor = format_dataarray(
            cooling_factor, "total_radiation_loss_parameter", coords
        )

        return self.cooling_factor

    def __call__(  # type: ignore
        self,
        Te: DataArray,
        F_z_t: DataArray,
        Ne: DataArray = None,
        Nn: DataArray = None,
    ):
        """Executes all functions in correct order to calculate the total radiated
        power.

        Ne
            electron density profile
        Te
            electron temperature profile
        Nn
            thermal neutral hydrogen density profile
        F_z_t
            fractional abundance of all ionisation charges of given element
        """

        if self.full_run or not hasattr(self, "cooling_factor"):
            self.interpolate_power(Ne, Te)
            cooling_factor = self.calculate_power_loss(Ne, F_z_t, Nn)  # type: ignore
            self.cooling_factor = cooling_factor
        else:
            cooling_factor = interpolate_results(self.cooling_factor, self.Te, Te)

        return cooling_factor


def cooling_factor_corona(
    elements: List[str],
    write_to_file: bool = False,
    plot: bool = False,
    new_figure: bool = True,
    include_neutrals: bool = False,
):
    """
    Initialises atomic data classes with default ADAS files and runs the
    __call__ with default plasma parameters
    """
    tau = None
    Ne_const = 5.0e19
    Nn1 = 1.0e17
    Nn0 = 1.0e12

    fract_abu: dict = {}
    power_loss_tot: dict = {}
    atomic_data_files: dict = {}
    cooling_factor: dict = {}
    filenames = ""
    files_to_read = ["scd", "acd", "ccd", "plt", "prb", "prc"]
    Te_files = []

    print("Read atomic data")
    adas_reader = ADASReader()
    for elem in elements:
        atomic_data_files[elem] = {}
        for file_type in files_to_read:
            _atomic_data = adas_reader.get_adf11(
                file_type, elem, ADF11[elem][file_type]
            )
            filenames += f"{_atomic_data.filename}"
            Te_files.append(_atomic_data.electron_temperature)
            atomic_data_files[elem][file_type] = _atomic_data

    # Set Te so that max(Te) doesn't exceed the value available in all atomic-data files
    _indx = np.argmin(np.array([np.max(_Te) for _Te in Te_files]))
    _Te = Te_files[_indx]
    nTe = np.size(_Te)
    Te = DataArray(_Te.data, coords=[("index", np.arange(nTe))])
    Ne = xr.full_like(Te, Ne_const)
    Nn = xr.full_like(Te, 0.0)
    if include_neutrals:
        _Nn = np.array([Te.values[i] for i in np.arange(Te.size - 1, -1, -1)])
        _Nn -= np.min(_Nn)
        _Nn /= np.max(_Nn)
        _Nn *= Nn1
        _Nn += Nn0
        Nn.values = _Nn

    _to_write = {"Te": np.array(Te), "Ne": np.array(Ne), "Nn": np.array(Nn)}

    print("Calculate fractional abundance and cooling factors")
    for elem in elements:
        print(f"  {elem}")
        fract_abu[elem] = FractionalAbundanceAdas(
            atomic_data_files[elem]["scd"],
            atomic_data_files[elem]["acd"],
            ccd=atomic_data_files[elem]["ccd"],
        )
        _fz = fract_abu[elem](Ne=Ne, Te=Te, Nn=Nn, tau=tau)

        power_loss_tot[elem] = PowerLoss(
            atomic_data_files[elem]["plt"],
            atomic_data_files[elem]["prb"],
            prc=atomic_data_files[elem]["prc"],
        )
        _power_loss = power_loss_tot[elem](Te, _fz, Ne=Ne, Nn=Nn)

        _cooling_factor: DataArray = _power_loss.sum("ion_charge")
        _cooling_factor = (
            _cooling_factor.assign_coords(electron_temperature=("index", Te.data))
            .swap_dims({"index": "electron_temperature"})
            .drop_vars("index")
        )

        cooling_factor[elem] = _cooling_factor
        _to_write[elem] = np.array(_cooling_factor)

    _to_write["atomic_data_files"] = filenames

    if write_to_file:
        if include_neutrals:
            file_name = f"{DATA_PATH}corona_cooling_factors_Nn.csv"
        else:
            file_name = f"{DATA_PATH}corona_cooling_factors.csv"
        print(f"Writing data to {file_name}")
        df = DataFrame(_to_write)
        df.to_csv(file_name)

    if plot:
        if new_figure:
            plt.figure()
        cmap, _ = set_plot_colors()
        cols = cmap(np.linspace(0.75, 0.1, len(cooling_factor), dtype=float))

        label = ""
        marker = "o"
        linestyle = "solid"
        if include_neutrals:
            marker = ""
            linestyle = "dashed"

        for i, elem in enumerate(elements):
            if new_figure:
                label = elem
            cooling_factor[elem].plot(
                label=label,
                alpha=0.8,
                marker=marker,
                color=cols[i],
                linestyle=linestyle,
            )
        if new_figure:
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()

    return cooling_factor, _to_write, fract_abu


def interpolate_results(
    data: DataArray, Te_data: DataArray, Te_interp: DataArray, method="cubic"
):
    """
    Interpolate fractional abundance or cooling factor on electron
    temperature for fast processing

    atomic_data
        Fractional abundance or cooling factor DataArrays
    Te
        Electron temperature on which interpolation is to be performed
    """
    dim_old = [d for d in data.dims if d != "ion_charge"][0]
    _data = data.assign_coords(electron_temperature=(dim_old, Te_data.data))
    _data = _data.swap_dims({dim_old: "electron_temperature"}).drop_vars(dim_old)
    result = _data.interp(electron_temperature=Te_interp).drop_vars(
        ("electron_temperature",)
    )
    return result
