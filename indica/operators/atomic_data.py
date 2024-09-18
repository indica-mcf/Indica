import copy
from typing import cast
from typing import List
from typing import Tuple
from typing import Union

import matplotlib.pylab as plt
import numpy as np
from numpy.core.numeric import zeros_like
from pandas import DataFrame
import scipy
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.readers.adas import ADASReader
from indica.readers.adas import ADF11
from indica.utilities import DATA_PATH
from indica.utilities import set_plot_colors
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from ..datatypes import DataType

np.set_printoptions(edgeitems=10, linewidth=100)


class FractionalAbundance(Operator):
    """Calculate fractional abundance for all ionisation charges of a given element.

    Parameters
    ----------
    scd
        xarray.DataArray of effective ionisation rate coefficients of all relevant
        ionisation charges of given impurity element.
    acd
        xarray.DataArray of effective recombination rate coefficients of all relevant
        ionisation charges of given impurity element.
    ccd
        xarray.DataArray of charge exchange cross coupling coefficients of all relevant
        ionisation charges of given impurity element. (Optional)

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.

    Returns
    -------
    F_z_t
        xarray.DataArray of fractional abundance of all ionisation charges of given
        impurity element.

    Methods
    -------
    interpolate_rates(Ne, Te)
        Interpolates rates based on inputted Ne and Te, also determines the number
        of ionisation charges for a given element.
    calc_ionisation_balance_matrix(Ne, Nh)
        Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation charges.
    calc_F_z_tinf()
        Calculates the equilibrium fractional abundance of all ionisation charges,
        F_z(t=infinity) used for the final time evolution equation.
    calc_eigen_vals_and_vecs()
        Calculates the eigenvalues and eigenvectors of the ionisation balance matrix.
    calc_eigen_coeffs(F_z_t0)
        Calculates the coefficients from the eigenvalues and eigenvectors for the time
        evolution equation.
    calculate_abundance(tau)
        Calculates the fractional abundance of all ionisation charges at time tau.
    __call__(Ne, Te, Nh, tau, F_z_t0, full_run)
        Executes all functions in correct order to calculate the fractional abundance.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("ionisation_rate", "impurity_element"),
        ("recombination_rate", "impurity_element"),
        ("charge-exchange_rate", "impurity_element"),
        ("number_density", "electrons"),
        ("temperature", "electrons"),
        ("number_density", "thermal_hydrogen"),
        ("initial_fractional_abundance", "impurity_element"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("fractional_abundance", "impurity_element"),
    ]

    def __init__(
        self,
        scd: DataArray,
        acd: DataArray,
        ccd: DataArray = None,
    ):
        """Initialises FractionalAbundance class"""
        self.Ne = None
        self.Te = None
        self.Nh = None
        self.tau = None
        self.F_z_t0 = None
        self.scd = scd
        self.acd = acd
        self.ccd = ccd

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        """Indicates the datatypes of the results when calling the operator
        with arguments of the given types. It is assumed that the
        argument types are valid.

        Parameters
        ----------
        args
            The datatypes of the parameters which the operator is to be called with.

        Returns
        -------
        :
            The datatype of each result that will be returned if the operator is
            called with these arguments.

        """
        return (("fractional abundance", "impurity_element"),)

    def interpolate_rates(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Interpolates rates based on inputted Ne and Te, also determines the number
        of ionisation charges for a given element.

        Parameters
        ----------
        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        Te
            xarray.DataArray of electron temperature as a profile of a user-chosen
            coordinate.

        Returns
        -------
        scd_spec
            Interpolated effective ionisation rate coefficients.
        acd_spec
            Interpolated effective recombination rate coefficients.
        ccd_spec
            Interpolated charge exchange cross coupling coefficients.
        num_of_ion_charge
            Number of ionisation charges(stages) for the given impurity element.
        """

        self.Ne, self.Te = Ne, Te  # type: ignore

        scd_spec = self.scd.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        if self.ccd is not None:
            ccd_spec = self.ccd.indica.interp2d(
                electron_temperature=Te,
                electron_density=Ne,
                method="cubic",
                assume_sorted=True,
            )
        else:
            ccd_spec = None

        acd_spec = self.acd.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        self.scd_spec, self.acd_spec, self.ccd_spec = scd_spec, acd_spec, ccd_spec
        self.num_of_ion_charge = self.scd_spec.shape[0] + 1

        return scd_spec, acd_spec, ccd_spec, self.num_of_ion_charge

    def calc_ionisation_balance_matrix(
        self,
        Ne: DataArray,
        Nh: DataArray = None,
    ):
        """Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation charges.

        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        Nh
            xarray.DataArray of thermal hydrogen as a profile of a user-chosen
            coordinate. (Optional)

        Returns
        -------
        ionisation_balance_matrix
            Matrix representing coefficients of the differential equation governing
            the time evolution of the ionisation balance.
        """
        if Nh is not None:
            if self.ccd is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    ccd (effective charge exchange recombination) at initialisation \
                    is None."
                )
        elif self.ccd is not None:
            Nh = cast(DataArray, zeros_like(Ne))

        self.Ne, self.Nh = Ne, Nh  # type: ignore

        num_of_ion_charge = self.num_of_ion_charge
        scd, acd, ccd = self.scd_spec, self.acd_spec, self.ccd_spec

        x1_coord = scd.coords[[k for k in scd.dims if k != "ion_charge"][0]]
        self.x1_coord = x1_coord

        dims = (
            num_of_ion_charge,
            num_of_ion_charge,
            *x1_coord.shape,
        )

        ionisation_balance_matrix = np.zeros(dims)

        icharge = 0
        ionisation_balance_matrix[icharge, icharge : icharge + 2] = np.array(
            [
                -Ne * scd[icharge],  # type: ignore
                Ne * acd[icharge]
                + (Nh * ccd[icharge] if Nh is not None and ccd is not None else 0.0),
            ]
        )
        for icharge in range(1, num_of_ion_charge - 1):
            ionisation_balance_matrix[icharge, icharge - 1 : icharge + 2] = np.array(
                [
                    Ne * scd[icharge - 1],
                    -Ne * (scd[icharge] + acd[icharge - 1])  # type: ignore
                    - (
                        Nh * ccd[icharge - 1]
                        if Nh is not None and ccd is not None
                        else 0.0
                    ),
                    Ne * acd[icharge]
                    + (
                        Nh * ccd[icharge] if Nh is not None and ccd is not None else 0.0
                    ),
                ]
            )
        icharge = num_of_ion_charge - 1
        ionisation_balance_matrix[icharge, icharge - 1 : icharge + 1] = np.array(
            [
                Ne * scd[icharge - 1],
                -Ne * (acd[icharge - 1])  # type: ignore
                - (
                    Nh * ccd[icharge - 1] if Nh is not None and ccd is not None else 0.0
                ),
            ]
        )

        ionisation_balance_matrix = np.squeeze(ionisation_balance_matrix)
        self.ionisation_balance_matrix = ionisation_balance_matrix

        return ionisation_balance_matrix

    def calc_F_z_tinf(
        self,
    ):
        """Calculates the equilibrium fractional abundance of all ionisation charges,
        F_z(t=infinity) used for the final time evolution equation.

        Returns
        -------
        F_z_tinf
            Fractional abundance at equilibrium.
        """
        x1_coord = self.x1_coord
        ionisation_balance_matrix = self.ionisation_balance_matrix

        null_space = np.zeros((self.num_of_ion_charge, x1_coord.size))
        F_z_tinf = np.zeros((self.num_of_ion_charge, x1_coord.size))

        for ix1 in range(x1_coord.size):
            null_space[:, ix1, np.newaxis] = scipy.linalg.null_space(
                ionisation_balance_matrix[:, :, ix1]
            )

        # Complex type casting for compatibility with eigen calculation results later.
        F_z_tinf = np.abs(null_space).astype(dtype=np.complex128)

        # normalization needed for high-z elements
        F_z_tinf = F_z_tinf / np.sum(F_z_tinf, axis=0)

        F_z_tinf = DataArray(
            data=F_z_tinf,
            coords=[
                (
                    "ion_charge",
                    np.linspace(0, self.num_of_ion_charge - 1, self.num_of_ion_charge),
                ),
                x1_coord,
            ],
            dims=["ion_charge", x1_coord.dims[0]],
        )

        self.F_z_tinf = F_z_tinf

        return np.real(F_z_tinf)

    def calc_eigen_vals_and_vecs(
        self,
    ):
        """Calculates the eigenvalues and eigenvectors of the ionisation balance
        matrix.

        Returns
        -------
        eig_vals
            Eigenvalues of the ionisation balance matrix.
        eig_vecs
            Eigenvectors of the ionisation balance matrix.
        """
        x1_coord = self.x1_coord
        eig_vals = np.zeros(
            (self.num_of_ion_charge, x1_coord.size), dtype=np.complex128
        )
        eig_vecs = np.zeros(
            (self.num_of_ion_charge, self.num_of_ion_charge, x1_coord.size),
            dtype=np.complex128,
        )

        for ix1 in range(x1_coord.size):
            eig_vals[:, ix1], eig_vecs[:, :, ix1] = scipy.linalg.eig(
                self.ionisation_balance_matrix[:, :, ix1],
            )

        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs

        return eig_vals, eig_vecs

    def calc_eigen_coeffs(
        self,
        F_z_t0: DataArray = None,
    ):
        """Calculates the coefficients from the eigenvalues and eigenvectors for the
        time evolution equation.

        Parameters
        ----------
        F_z_t0
            Initial fractional abundance for given impurity element. (Optional)

        Returns
        -------
        eig_coeffs
            Coefficients calculated from the eigenvalues and eigenvectors needed
            for the time evolution equation.
        F_z_t0
            Initial fractional abundance, either user-provided or fully neutral
            eg. [1.0, 0.0, 0.0, 0.0, 0.0] for Beryllium.
        """
        x1_coord = self.x1_coord

        if F_z_t0 is None:
            # mypy doesn't understand contionals or reassignments either.
            F_z_t0 = np.zeros(self.F_z_tinf.shape, dtype=np.complex128)  # type: ignore
            F_z_t0[0, :] = np.array(  # type: ignore
                [1.0 + 0.0j for i in range(x1_coord.size)]
            )

            F_z_t0 = DataArray(
                data=F_z_t0,
                coords=[
                    (
                        "ion_charge",
                        np.linspace(
                            0, self.num_of_ion_charge - 1, self.num_of_ion_charge
                        ),
                    ),
                    x1_coord,  # type: ignore
                ],
                dims=["ion_charge", x1_coord.dims[0]],
            )
        else:
            try:
                assert F_z_t0.ndim < 3
            except AssertionError:
                raise ValueError("F_z_t0 must be at most 2-dimensional.")

            F_z_t0 = F_z_t0 / np.sum(F_z_t0, axis=0)
            F_z_t0 = F_z_t0.as_type(dtype=np.complex128)  # type: ignore

            F_z_t0 = DataArray(
                data=F_z_t0.values,  # type: ignore
                coords=[
                    (
                        "ion_charge",
                        np.linspace(
                            0, self.num_of_ion_charge - 1, self.num_of_ion_charge
                        ),
                    ),
                    x1_coord,  # type: ignore
                ],
                dims=["ion_charge", x1_coord.dims[0]],
            )

        eig_vals = self.eig_vals
        eig_vecs_inv = np.zeros(self.eig_vecs.shape, dtype=np.complex128)
        for ix1 in range(x1_coord.size):
            eig_vecs_inv[:, :, ix1] = np.linalg.pinv(
                np.transpose(self.eig_vecs[:, :, ix1])
            )

        boundary_conds = F_z_t0 - self.F_z_tinf

        eig_coeffs = np.zeros(eig_vals.shape, dtype=np.complex128)
        for ix1 in range(x1_coord.size):
            eig_coeffs[:, ix1] = np.dot(boundary_conds[:, ix1], eig_vecs_inv[:, :, ix1])

        self.eig_coeffs = eig_coeffs

        # If argument to numpy functions is of type DataArray then output is of
        # type DataArray
        F_z_t0 = np.abs(np.real(F_z_t0))  # type: ignore

        self.F_z_t0 = F_z_t0  # type: ignore

        return eig_coeffs, F_z_t0

    def calculate_abundance(self, tau: LabeledArray):
        """Calculates the fractional abundance of all ionisation charges at time tau.

        Parameters
        ----------
        tau
            Time after t0 (t0 is defined as the time at which F_z_t0 is taken).

        Returns
        -------
        F_z_t
            Fractional abundance at tau.
        """
        x1_coord = self.x1_coord
        F_z_t = copy.deepcopy(self.F_z_tinf)
        for ix1 in range(x1_coord.size):
            if isinstance(tau, (DataArray, np.ndarray)):
                itau = tau[ix1].values if isinstance(tau, DataArray) else tau[ix1]
            else:
                itau = tau

            for icharge in range(self.num_of_ion_charge):
                F_z_t[:, ix1] += (
                    self.eig_coeffs[icharge, ix1]
                    * np.exp(self.eig_vals[icharge, ix1] * itau)
                    * self.eig_vecs[:, icharge, ix1]
                )

        F_z_t = np.abs(np.real(F_z_t))

        self.F_z_t = F_z_t
        # Mypy complains about assigning a LabeledArray to an object that was type None.
        # Can't really fix incompatibility without eliminating LabeledArray since mypy
        # seems to have an issue with type aliases.
        self.tau = tau  # type: ignore

        return F_z_t

    def __call__(  # type: ignore
        self,
        Te: DataArray,
        Ne: DataArray = None,
        Nh: DataArray = None,
        tau: LabeledArray = None,
        F_z_t0: DataArray = None,
        full_run: bool = False,
    ) -> DataArray:
        """Executes all functions in correct order to calculate the fractional
        abundance.

        Parameters
        ----------
        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        Te
            xarray.DataArray of electron temperature as a profile of a user-chosen
            coordinate.
        Nh
            xarray.DataArray of thermal hydrogen as a profile of a user-chosen
            coordinate. (Optional)
        tau
            Time after t0 (t0 is defined as the time at which F_z_t0 is taken).
            (Optional)
        F_z_t0
            Initial fractional abundance for given impurity element. (Optional)
        full_run
            Boolean specifying whether to run the entire ordered workflow(True)
            for calculating abundance from the start. If (False), fractional abundance
            will be interpolated on input electron temperature
            (Optional)

        Returns
        -------
        F_z_t
            Fractional abundance at tau.
        """
        if full_run or not hasattr(self, "F_z_t"):
            self.interpolate_rates(Ne, Te)

            self.calc_ionisation_balance_matrix(Ne, Nh)

            self.calc_F_z_tinf()

            if tau is None:
                F_z_t = np.real(self.F_z_tinf)
                self.F_z_t = F_z_t
                return F_z_t

            self.calc_eigen_vals_and_vecs()

            self.calc_eigen_coeffs(F_z_t0)

            F_z_t = self.calculate_abundance(tau)

            self.F_z_t = F_z_t
        else:

            F_z_t = interpolate_results(self.F_z_t, self.Te, Te)

        return F_z_t


class PowerLoss(Operator):
    """Calculate the total power loss associated with a given impurity element

    Parameters
    ----------
    plt
        xarray.DataArray of radiated power of line emission from excitation of all
        relevant ionisation charges of given impurity element.
    prb
        xarray.DataArray of radiated power from recombination and bremsstrahlung of
        given impurity element.
    prc
        xarray.DataArray of radiated power of charge exchange emission of all relevant
        ionisation charges of given impurity element. (Optional)

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.

    Returns
    -------
    cooling_factor
        xarray.DataArray of total radiated power loss of all ionisation charges of given
        impurity element.

    Methods
    -------
    interpolate_power(Ne, Te)
        Interpolates the various powers based on inputted Ne and Te.
    calculate_power_loss(Ne, F_z_t, Nh)
        Calculates total radiated power of all ionisation charges of a given
        impurity element.
    __call__(Ne, Te, Nh, F_z_t, full_run)
        Executes all functions in correct order to calculate the total radiated power.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("line_power_coefficient", "impurity_element"),
        ("recombination_power_coefficient", "impurity_element"),
        ("charge-exchange_power_coefficient", "impurity_element"),
        ("number_density", "electrons"),
        ("temperature", "electrons"),
        ("fractional_abundance", "impurity_element"),
        ("number_density", "thermal_hydrogen"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("total_radiated power loss", "impurity_element"),
    ]

    def __init__(
        self,
        plt: DataArray,
        prb: DataArray,
        prc: DataArray = None,
    ):
        self.plt = plt
        self.prc = prc
        self.prb = prb
        self.Ne = None
        self.Nh = None
        self.Te = None
        self.F_z_t = None

        imported_data = {}
        imported_data["plt"] = self.plt
        imported_data["prb"] = self.prb
        if self.prc is not None:
            imported_data["prc"] = self.prc

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        """Indicates the datatypes of the results when calling the operator
        with arguments of the given types. It is assumed that the
        argument types are valid.

        Parameters
        ----------
        args
            The datatypes of the parameters which the operator is to be called with.

        Returns
        -------
        :
            The datatype of each result that will be returned if the operator is
            called with these arguments.

        """
        return (("total_radiated power loss", "impurity_element"),)

    def interpolate_power(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Interpolates the various powers based on inputted Ne and Te.

        Parameters
        ----------
        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        Te
            xarray.DataArray of electron temperature as a profile of a user-chosen
            coordinate.

        Returns
        -------
        plt_spec
            Interpolated radiated power of line emission from excitation of all
            relevant ionisation charges.
        prc_spec
            Interpolated radiated power of charge exchange emission of all relevant
            ionisation charges.
        prb_spec
            Interpolated radiated power from recombination and bremsstrahlung.
        num_of_ion_charge
            Number of ionisation charges(stages) for the given impurity element.
        """

        self.Ne, self.Te = Ne, Te  # type: ignore
        plt_spec = self.plt.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        if self.prc is not None:
            prc_spec = self.prc.interp(electron_temperature=Te, method="cubic").interp(
                electron_density=Ne, method="linear"
            )

        else:
            prc_spec = None

        prb_spec = self.prb.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        self.plt_spec, self.prc_spec, self.prb_spec = plt_spec, prc_spec, prb_spec
        self.num_of_ion_charge = self.plt_spec.shape[0] + 1

        return plt_spec, prc_spec, prb_spec, self.num_of_ion_charge

    def calculate_power_loss(
        self,
        Ne: DataArray,
        F_z_t: DataArray,
        Nh: DataArray = None,
    ):
        """Calculates total radiated power of all ionisation charges of a given
        impurity element.

        Parameters
        ----------
        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        F_z_t
            xarray.DataArray of fractional abundance of all ionisation charges of given
            impurity element.
        Nh
            xarray.DataArray of thermal hydrogen number density as a profile of a
            user-chosen coordinate. (Optional)

        Returns
        -------
        cooling_factor
            Total radiated power of all ionisation charges.
        """
        if Nh is not None:
            if self.prc is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    prc (effective charge exchange power) at initialisation \
                    is None."
                )
        elif self.prc is not None:
            Nh = cast(DataArray, zeros_like(Ne))

        self.Ne, self.Nh = Ne, Nh  # type: ignore

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

        self.x1_coord = self.plt_spec.coords[
            [k for k in self.plt_spec.dims if k != "ion_charge"][0]
        ]

        x1_coord = self.x1_coord

        plt, prb, prc = self.plt_spec, self.prb_spec, self.prc_spec

        # TODO: make this faster by using DataArray methods
        cooling_factor = xr.zeros_like(self.F_z_t)
        for ix1 in range(x1_coord.size):
            icharge = 0
            cooling_factor[icharge, ix1] = (
                plt[icharge, ix1] * self.F_z_t[icharge, ix1]  # type: ignore
            )
            for icharge in range(1, self.num_of_ion_charge - 1):
                cooling_factor[icharge, ix1] = (
                    plt[icharge, ix1]
                    + (
                        (Nh[ix1] / Ne[ix1]) * prc[icharge - 1, ix1]
                        if (prc is not None) and (Nh is not None)
                        else 0.0
                    )
                    + prb[icharge - 1, ix1]
                ) * self.F_z_t[
                    icharge, ix1
                ]  # type: ignore

            icharge = self.num_of_ion_charge - 1
            cooling_factor[icharge, ix1] = (
                (
                    (Nh[ix1] / Ne[ix1]) * prc[icharge - 1, ix1]
                    if (prc is not None) and (Nh is not None)
                    else 0.0
                )
                + prb[icharge - 1, ix1]
            ) * self.F_z_t[
                icharge, ix1
            ]  # type: ignore

        self.cooling_factor = cooling_factor

        return cooling_factor

    def __call__(  # type: ignore
        self,
        Te: DataArray,
        F_z_t: DataArray,
        Ne: DataArray = None,
        Nh: DataArray = None,
        full_run: bool = False,
    ):
        """Executes all functions in correct order to calculate the total radiated
        power.

        Parameters
        ----------
        Ne
            xarray.DataArray of electron density as a profile of a user-chosen
            coordinate.
        Te
            xarray.DataArray of electron temperature as a profile of a user-chosen
            coordinate.
        Nh
            xarray.DataArray of thermal hydrogen number density as a profile of a
            user-chosen coordinate. (Optional)
        F_z_t
            xarray.DataArray of fractional abundance of all ionisation charges of given
            impurity element. (Optional)
        full_run
            Boolean specifying whether to only run calculate_power_loss(False) or to
            run the entire ordered workflow(True) for calculating power loss from the
            start. This is mostly only useful for unit testing and is set to True by
            default. (Optional)

        Returns
        -------
        cooling_factor
            Total radiated power of all ionisation charges.
        """

        if full_run or not hasattr(self, "cooling_factor"):
            self.interpolate_power(Ne, Te)
            cooling_factor = self.calculate_power_loss(Ne, F_z_t, Nh)  # type: ignore
            self.cooling_factor = cooling_factor
        else:
            cooling_factor = interpolate_results(self.cooling_factor, self.Te, Te)

        return cooling_factor


def interpolate_results(
    data: DataArray, Te_data: DataArray, Te_interp: DataArray, method="cubic"
):
    """
    Interpolate fractional abundance or cooling factor on electron
    temperature for fast processing

    Parameters
    ----------
    atomic_data
        Fractional abundance or cooling factor DataArrays
    Te
        Electron temperature on which interpolation is to be performed

    Returns
    -------
    Interpolated values
    """
    dim_old = [d for d in data.dims if d != "ion_charge"][0]
    _data = data.assign_coords(electron_temperature=(dim_old, Te_data.data))
    _data = _data.swap_dims({dim_old: "electron_temperature"}).drop_vars(dim_old)
    result = _data.interp(electron_temperature=Te_interp).drop_vars(
        ("electron_temperature",)
    )
    return result


def default_atomic_data(
    elements: Tuple[str, ...],
    Te: DataArray = None,
    Ne: DataArray = None,
    Nh: DataArray = None,
    tau: DataArray = None,
):
    """
    Initialises atomic data classes with default ADAS files and runs the
    __call__ with default plasma parameters
    """
    if Te is None or Ne is None:
        Te, Ne, Nh, tau = default_profiles()

    # print_like("Initialize fractional abundance and power loss objects")
    fract_abu, power_loss_tot, power_loss_sxr = {}, {}, {}
    adas_reader = ADASReader()
    for elem in elements:
        scd = adas_reader.get_adf11("scd", elem, ADF11[elem]["scd"])
        acd = adas_reader.get_adf11("acd", elem, ADF11[elem]["acd"])
        ccd = adas_reader.get_adf11("ccd", elem, ADF11[elem]["ccd"])
        fract_abu[elem] = FractionalAbundance(scd, acd, ccd=ccd)
        F_z_t = fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau)
        F_z_t = fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau)
        plt = adas_reader.get_adf11("plt", elem, ADF11[elem]["plt"])
        prb = adas_reader.get_adf11("prb", elem, ADF11[elem]["prb"])
        prc = adas_reader.get_adf11("prc", elem, ADF11[elem]["prc"])
        power_loss_tot[elem] = PowerLoss(plt, prb, prc=prc)
        power_loss_tot[elem](Te, F_z_t, Ne=Ne, Nh=Nh)

        try:
            pls = adas_reader.get_adf11("pls", elem, ADF11[elem]["pls"])
            prs = adas_reader.get_adf11("prs", elem, ADF11[elem]["prs"])
            power_loss_sxr[elem] = PowerLoss(pls, prs)
            power_loss_sxr[elem](Te, F_z_t, Ne=Ne, Nh=Nh)
        except Exception:
            print(f"No SXR-filtered data available for element {elem}")

    return fract_abu, power_loss_tot, power_loss_sxr


def default_profiles(n_rad: int = 20):
    """
    Set default plasma profiles to calculate atomic data
    """
    xend = 1.02
    rho_end = 1.01
    rho = np.abs(np.linspace(rho_end, 0, n_rad) ** 1.8 - rho_end - 0.01)
    rho_coord = xr.DataArray(
        rho, coords={"rho_poloidal": rho}, dims="rho_poloidal"
    ).coords
    Te = xr.DataArray(np.linspace(50, 10e3, n_rad), coords=rho_coord)
    Ne = xr.DataArray(np.logspace(18, 21, n_rad), coords=rho_coord)

    # TODO: fix FractionalAbundance so that it does 2d interp of Nh and Te
    params = {
        "y0": 1e14,
        "y1": 5e15,
        "yend": 5e15,
        "wcenter": 0.01,
        "wped": 18,
        "peaking": 1,
    }
    Nh_prof = ProfilerGauss(
        datatype="neutral_density", xspl=rho, xend=xend, parameters=params
    )
    Nh = Nh_prof()
    tau = None
    return Te, Ne, Nh, tau


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
    Nh1 = 1.0e17
    Nh0 = 1.0e12

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
    Nh = xr.full_like(Te, 0.0)
    if include_neutrals:
        _Nh = np.array([Te.values[i] for i in np.arange(Te.size - 1, -1, -1)])
        _Nh -= np.min(_Nh)
        _Nh /= np.max(_Nh)
        _Nh *= Nh1
        _Nh += Nh0
        Nh.values = _Nh

    _to_write = {"Te": np.array(Te), "Ne": np.array(Ne), "Nh": np.array(Nh)}

    print("Calculate fractional abundance and cooling factors")
    for elem in elements:
        print(f"  {elem}")
        fract_abu[elem] = FractionalAbundance(
            atomic_data_files[elem]["scd"],
            atomic_data_files[elem]["acd"],
            ccd=atomic_data_files[elem]["ccd"],
        )
        _fz = fract_abu[elem](Ne=Ne, Te=Te, Nh=Nh, tau=tau)

        power_loss_tot[elem] = PowerLoss(
            atomic_data_files[elem]["plt"],
            atomic_data_files[elem]["prb"],
            prc=atomic_data_files[elem]["prc"],
        )
        _power_loss = power_loss_tot[elem](Te, _fz, Ne=Ne, Nh=Nh)

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
            file_name = f"{DATA_PATH}corona_cooling_factors_Nh.csv"
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
