import copy
from typing import cast
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from numpy.core.numeric import zeros_like
import scipy
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType

np.set_printoptions(edgeitems=10, linewidth=100)


class FractionalAbundance(Operator):
    """Calculate fractional abundance for all ionisation charges of a given element.

    Parameters
    ----------
    SCD
        xarray.DataArray of effective ionisation rate coefficients of all relevant
        ionisation charges of given impurity element.
    ACD
        xarray.DataArray of effective recombination rate coefficients of all relevant
        ionisation charges of given impurity element.
    CCD
        xarray.DataArray of charge exchange cross coupling coefficients of all relevant
        ionisation charges of given impurity element. (Optional)
    sess
        Object representing this session of calculations with the library.
        Holds and communicates provenance information. (Optional)

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
        SCD: DataArray,
        ACD: DataArray,
        CCD: DataArray = None,
        sess: session.Session = session.global_session,
        set_default: bool = False,
        main_ion: str = "h",
    ):
        """Initialises FractionalAbundance class"""
        super().__init__(sess)
        self.Ne = None
        self.Te = None
        self.Nh = None
        self.tau = None
        self.F_z_t0 = None
        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD

        if set_default:
            Te, Ne, Nh = default_profiles(main_ion=main_ion)
            self.__call__(Ne=Ne, Te=Te, Nh=Nh)

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
        SCD_spec
            Interpolated effective ionisation rate coefficients.
        ACD_spec
            Interpolated effective recombination rate coefficients.
        CCD_spec
            Interpolated charge exchange cross coupling coefficients.
        num_of_ion_charges
            Number of ionisation charges(stages) for the given impurity element.
        """

        self.Ne, self.Te = Ne, Te  # type: ignore

        SCD_spec = self.SCD.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        if self.CCD is not None:
            CCD_spec = self.CCD.indica.interp2d(
                electron_temperature=Te,
                electron_density=Ne,
                method="cubic",
                assume_sorted=True,
            )
        else:
            CCD_spec = None

        ACD_spec = self.ACD.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )

        self.SCD_spec, self.ACD_spec, self.CCD_spec = SCD_spec, ACD_spec, CCD_spec
        self.num_of_ion_charges = self.SCD_spec.shape[0] + 1

        return SCD_spec, ACD_spec, CCD_spec, self.num_of_ion_charges

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
            if self.CCD is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    CCD (effective charge exchange recombination) at initialisation \
                    is None."
                )
        elif self.CCD is not None:
            Nh = cast(DataArray, zeros_like(Ne))

        self.Ne, self.Nh = Ne, Nh  # type: ignore

        num_of_ion_charges = self.num_of_ion_charges
        SCD, ACD, CCD = self.SCD_spec, self.ACD_spec, self.CCD_spec

        x1_coord = SCD.coords[[k for k in SCD.dims if k != "ion_charges"][0]]
        self.x1_coord = x1_coord

        dims = (
            num_of_ion_charges,
            num_of_ion_charges,
            *x1_coord.shape,
        )

        ionisation_balance_matrix = np.zeros(dims)

        icharge = 0
        ionisation_balance_matrix[icharge, icharge : icharge + 2] = np.array(
            [
                -Ne * SCD[icharge],  # type: ignore
                Ne * ACD[icharge]
                + (Nh * CCD[icharge] if Nh is not None and CCD is not None else 0.0),
            ]
        )
        for icharge in range(1, num_of_ion_charges - 1):
            ionisation_balance_matrix[icharge, icharge - 1 : icharge + 2] = np.array(
                [
                    Ne * SCD[icharge - 1],
                    -Ne * (SCD[icharge] + ACD[icharge - 1])  # type: ignore
                    - (
                        Nh * CCD[icharge - 1]
                        if Nh is not None and CCD is not None
                        else 0.0
                    ),
                    Ne * ACD[icharge]
                    + (
                        Nh * CCD[icharge] if Nh is not None and CCD is not None else 0.0
                    ),
                ]
            )
        icharge = num_of_ion_charges - 1
        ionisation_balance_matrix[icharge, icharge - 1 : icharge + 1] = np.array(
            [
                Ne * SCD[icharge - 1],
                -Ne * (ACD[icharge - 1])  # type: ignore
                - (
                    Nh * CCD[icharge - 1] if Nh is not None and CCD is not None else 0.0
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

        null_space = np.zeros((self.num_of_ion_charges, x1_coord.size))
        F_z_tinf = np.zeros((self.num_of_ion_charges, x1_coord.size))

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
                    "ion_charges",
                    np.linspace(
                        0, self.num_of_ion_charges - 1, self.num_of_ion_charges
                    ),
                ),
                x1_coord,
            ],
            dims=["ion_charges", x1_coord.dims[0]],
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
            (self.num_of_ion_charges, x1_coord.size), dtype=np.complex128
        )
        eig_vecs = np.zeros(
            (self.num_of_ion_charges, self.num_of_ion_charges, x1_coord.size),
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
                        "ion_charges",
                        np.linspace(
                            0, self.num_of_ion_charges - 1, self.num_of_ion_charges
                        ),
                    ),
                    x1_coord,  # type: ignore
                ],
                dims=["ion_charges", x1_coord.dims[0]],
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
                        "ion_charges",
                        np.linspace(
                            0, self.num_of_ion_charges - 1, self.num_of_ion_charges
                        ),
                    ),
                    x1_coord,  # type: ignore
                ],
                dims=["ion_charges", x1_coord.dims[0]],
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

            for icharge in range(self.num_of_ion_charges):
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
    PLT
        xarray.DataArray of radiated power of line emission from excitation of all
        relevant ionisation charges of given impurity element.
    PRB
        xarray.DataArray of radiated power from recombination and bremsstrahlung of
        given impurity element.
    PRC
        xarray.DataArray of radiated power of charge exchange emission of all relevant
        ionisation charges of given impurity element. (Optional)
    sess
        Object representing this session of calculations with the library.
        Holds and communicates provenance information. (Optional)

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
        PLT: DataArray,
        PRB: DataArray,
        PRC: DataArray = None,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess)
        self.PLT = PLT
        self.PRC = PRC
        self.PRB = PRB
        self.Ne = None
        self.Nh = None
        self.Te = None
        self.F_z_t = None

        imported_data = {}
        imported_data["PLT"] = self.PLT
        imported_data["PRB"] = self.PRB
        if self.PRC is not None:
            imported_data["PRC"] = self.PRC

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
        PLT_spec
            Interpolated radiated power of line emission from excitation of all
            relevant ionisation charges.
        PRC_spec
            Interpolated radiated power of charge exchange emission of all relevant
            ionisation charges.
        PRB_spec
            Interpolated radiated power from recombination and bremsstrahlung.
        num_of_ion_charges
            Number of ionisation charges(stages) for the given impurity element.
        """

        self.Ne, self.Te = Ne, Te  # type: ignore
        # TODO: why errors using interp2d for cubic density interpolation?
        # try:
        #     PLT_spec = self.PLT.indica.interp2d(
        #         electron_temperature=Te,
        #         electron_density=Ne,
        #         method="cubic",
        #         assume_sorted=True,
        #     )
        # except:
        # print("PowerLoss: error in indica.interp2d")
        PLT_spec = self.PLT.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        if self.PRC is not None:
            # try:
            #     PRC_spec = self.PRC.indica.interp2d(
            #         electron_temperature=Te,
            #         electron_density=Ne,
            #         method="cubic",
            #         assume_sorted=True,
            #     )
            # except:
            # print("PowerLoss: error in indica.interp2d")
            PRC_spec = self.PRC.interp(electron_temperature=Te, method="cubic").interp(
                electron_density=Ne, method="linear"
            )

        else:
            PRC_spec = None

        # try:
        #     PRB_spec = self.PRB.indica.interp2d(
        #         electron_temperature=Te,
        #         electron_density=Ne,
        #         method="cubic",
        #         assume_sorted=True,
        #     )
        # except:
        # print("PowerLoss: error in indica.interp2d")
        PRB_spec = self.PRB.interp(electron_temperature=Te, method="cubic").interp(
            electron_density=Ne, method="linear"
        )

        self.PLT_spec, self.PRC_spec, self.PRB_spec = PLT_spec, PRC_spec, PRB_spec
        self.num_of_ion_charges = self.PLT_spec.shape[0] + 1

        return PLT_spec, PRC_spec, PRB_spec, self.num_of_ion_charges

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
            if self.PRC is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    PRC (effective charge exchange power) at initialisation \
                    is None."
                )
        elif self.PRC is not None:
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

        self.x1_coord = self.PLT_spec.coords[
            [k for k in self.PLT_spec.dims if k != "ion_charges"][0]
        ]

        x1_coord = self.x1_coord

        PLT, PRB, PRC = self.PLT_spec, self.PRB_spec, self.PRC_spec

        # TODO: make this faster by using DataArray methods
        cooling_factor = xr.zeros_like(self.F_z_t)
        for ix1 in range(x1_coord.size):
            icharge = 0
            cooling_factor[icharge, ix1] = (
                PLT[icharge, ix1] * self.F_z_t[icharge, ix1]  # type: ignore
            )
            for icharge in range(1, self.num_of_ion_charges - 1):
                cooling_factor[icharge, ix1] = (
                    PLT[icharge, ix1]
                    + (
                        (Nh[ix1] / Ne[ix1]) * PRC[icharge - 1, ix1]
                        if (PRC is not None) and (Nh is not None)
                        else 0.0
                    )
                    + PRB[icharge - 1, ix1]
                ) * self.F_z_t[
                    icharge, ix1
                ]  # type: ignore

            icharge = self.num_of_ion_charges - 1
            cooling_factor[icharge, ix1] = (
                (
                    (Nh[ix1] / Ne[ix1]) * PRC[icharge - 1, ix1]
                    if (PRC is not None) and (Nh is not None)
                    else 0.0
                )
                + PRB[icharge - 1, ix1]
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
    dim_old = [d for d in data.dims if d != "ion_charges"][0]
    _data = data.assign_coords(electron_temperature=(dim_old, Te_data))
    _data = _data.swap_dims({dim_old: "electron_temperature"}).drop_vars(dim_old)
    result = _data.interp(electron_temperature=Te_interp).drop_vars(
        ("electron_temperature",)
    )
    return result


def default_profiles(main_ion: str = "h"):
    from indica.readers.adas import ADF11, ADASReader

    _Te = DataArray(np.append(np.arange(5, 500, 10), np.arange(600, 10.0e3, 100)))
    x = np.linspace(0, 1, np.size(_Te))
    Te = _Te.assign_coords(dim_0=x)
    Ne = xr.full_like(Te, 5.0e19)

    adas_reader = ADASReader()
    scd = adas_reader.get_adf11("scd", main_ion, ADF11[main_ion]["scd"])
    acd = adas_reader.get_adf11("acd", main_ion, ADF11[main_ion]["acd"])
    ccd = adas_reader.get_adf11("ccd", main_ion, ADF11[main_ion]["ccd"])
    Fz_main_ion = FractionalAbundance(scd, acd, CCD=ccd)
    fz = Fz_main_ion(Te, Ne=Ne)
    Nh = xr.full_like(Te, 0.0)
    _Nh = fz.sel(ion_charges=0) - fz.sel(ion_charges=0).min()
    _Nh /= _Nh.max()
    Nh.values = _Nh * 1.0e16 + 1.0e12

    return Te, Ne, Nh
