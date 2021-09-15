import copy
from typing import get_args
from typing import List
from typing import Tuple
from typing import Union
import warnings

import numpy as np
import scipy
import xarray as xr
from xarray import DataArray

from indica.numpy_typing import LabeledArray
from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..utilities import input_check


np.set_printoptions(edgeitems=10, linewidth=100)


def shape_check(
    data_to_check: dict,
):
    """Check to make sure all items in a given dictionary
    have the same dimensions as each other.
    Parameters
    ----------
    data_to_check
        Dictionary containing data to check.
    """
    try:
        for key1, val1 in data_to_check.items():
            for key2, val2 in data_to_check.items():
                assert val1.shape == val2.shape
    except AssertionError:
        raise ValueError(f"{key1} and {key2} are not the same shape")


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
    Ne
        xarray.DataArray of electron density as a profile of a user-chosen coordinate.
    Te
        xarray.DataArray of electron temperature as a profile of a user-chosen
        coordinate.
    Nh
        xarray.DataArray of thermal hydrogen as a profile of a user-chosen coordinate.
        (Optional)
    CCD
        xarray.DataArray of charge exchange cross coupling coefficients of all relevant
        ionisation charges of given impurity element. (Optional)
    F_z_t0
        Optional initial fractional abundance for given impurity element. (Optional)
    unit_testing
        Boolean for unit testing purposes
        (whether to call ordered_setup in __init__ or not) (Optional)
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
    interpolation_bounds_check(imported_data, inputted_data)
        Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(SCD,CCD,ACD,PLT,PRC,PRB).
    interpolate_rates()
        Interpolates rates based on inputted Ne and Te, also determines the number
        of ionisation charges for a given element.
    calc_ionisation_balance_matrix()
        Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation charges.
    calc_F_z_tinf()
        Calculates the equilibrium fractional abundance of all ionisation charges,
        F_z(t=infinity) used for the final time evolution equation.
    calc_eigen_vals_and_vecs()
        Calculates the eigenvalues and eigenvectors of the ionisation balance matrix.
    calc_eigen_coeffs()
        Calculates the coefficients from the eigenvalues and eigenvectors for the time
        evolution equation.
    ordered_setup()
        Sets up data for calculation in correct order.
    __call__(tau)
        Calculates the fractional abundance of all ionisation charges at time tau.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("ionisation_rate", "impurity_element"),
        ("recombination_rate", "impurity_element"),
        ("number_density", "electrons"),
        ("temperature", "electrons"),
        ("initial_fractional_abundance", "impurity_element"),
        ("number_density", "thermal_hydrogen"),
        ("charge-exchange_rate", "impurity_element"),
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
    ):
        """Initialises FractionalAbundance class and additionally performs error
        checking on inputs.
        """
        super().__init__(sess)
        self.Ne = None
        self.Te = None
        self.Nh = None
        self.tau = None
        self.F_z_t0 = None
        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD

        imported_data = {}
        imported_data["SCD"] = self.SCD
        imported_data["ACD"] = self.ACD
        if self.CCD is not None:
            imported_data["CCD"] = self.CCD

        for ikey, ival in imported_data.items():
            input_check(var_name=ikey, var_to_check=ival, var_type=DataArray)

        shape_check(imported_data)

    def interpolation_bounds_check(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(SCD,CCD,ACD,PLT,PRC,PRB).

        Parameters
        ----------
        imported_data
            Imported data (SCD, ACD, CCD, PLT, PRC, PRB)
        inputted_data
            Inputted data (Ne, Nh, Te) (Nh is tested in self.__init__())
        """
        imported_data = {}
        imported_data["SCD"] = self.SCD
        imported_data["ACD"] = self.ACD
        if self.CCD is not None:
            imported_data["CCD"] = self.CCD

        inputted_data = {}

        input_check("Ne", Ne, DataArray, greater_than_or_equal_zero=True)
        inputted_data["Ne"] = Ne

        input_check("Te", Te, DataArray, greater_than_or_equal_zero=False)
        inputted_data["Te"] = Te

        shape_check(inputted_data)

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] <= np.max(val.coords["electron_density"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] >= np.min(val.coords["electron_density"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] <= np.max(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] >= np.min(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron temperature is smaller than the \
                    minimum interpolation range in {key}"
            )

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

        self.interpolation_bounds_check(Ne, Te)

        self.Ne, self.Te = Ne, Te  # type: ignore

        SCD_spec = self.SCD.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        SCD_spec = SCD_spec

        if self.CCD is not None:
            CCD_spec = self.CCD.indica.interp2d(
                electron_temperature=Te,
                electron_density=Ne,
                method="cubic",
                assume_sorted=True,
            )
            CCD_spec = CCD_spec
        else:
            CCD_spec = None

        ACD_spec = self.ACD.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        ACD_spec = ACD_spec

        self.SCD, self.ACD, self.CCD = SCD_spec, ACD_spec, CCD_spec
        self.num_of_ion_charges = self.SCD.shape[0] + 1

        return SCD_spec, ACD_spec, CCD_spec, self.num_of_ion_charges

    def calc_ionisation_balance_matrix(
        self,
        Ne: DataArray,
        Nh: DataArray = None,
    ):
        """Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation charges.

        Returns
        -------
        ionisation_balance_matrix
            Matrix representing coefficients of the differential equation governing
            the time evolution of the ionisation balance.
        """
        inputted_data = {}

        input_check("Ne", Ne, DataArray, greater_than_or_equal_zero=True)
        inputted_data["Ne"] = Ne

        if Nh is not None:
            if self.CCD is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    CCD (effective charge exchange recombination) at initialisation \
                    is None."
                )
            input_check("Nh", Nh, DataArray, greater_than_or_equal_zero=True)
            inputted_data["Nh"] = Nh
        elif self.CCD is not None:
            raise ValueError(
                "Nh (Thermal hydrogen density) cannot be None when\
                CCD (effective charge exchange recombination) at initialisation \
                is not None."
            )

        shape_check(inputted_data)

        if self.Ne is not None:
            if np.logical_not(np.all(Ne == self.Ne)):
                warnings.warn(
                    "Ne given to calc_ionisation_balance_matrix is different from \
                        the internal Ne known to FractionalAbundance object."
                )

        self.Ne, self.Nh = Ne, Nh  # type: ignore

        num_of_ion_charges = self.num_of_ion_charges
        SCD, ACD, CCD = self.SCD, self.ACD, self.CCD

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
            F_z_t0 = np.zeros(self.F_z_tinf.shape, dtype=np.complex128)
            F_z_t0[0, :] = np.array([1.0 + 0.0j for i in range(x1_coord.size)])

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
            input_check("F_z_t0", F_z_t0, DataArray, greater_than_or_equal_zero=True)

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

        F_z_t0 = np.abs(np.real(F_z_t0))

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

        input_check(
            "tau",
            tau,
            get_args(LabeledArray),
            greater_than_or_equal_zero=True,
        )

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
        Ne: DataArray,
        Te: DataArray,
        Nh: DataArray = None,
        tau: LabeledArray = 1e3,
        F_z_t0: DataArray = None,
        full_run: bool = True,
    ) -> DataArray:
        """
        Sets up data for calculation in correct order. Allows changing of the inputted
        """
        if full_run:
            self.interpolate_rates(Ne, Te)

            self.calc_ionisation_balance_matrix(Ne, Nh)

            self.calc_F_z_tinf()

            self.calc_eigen_vals_and_vecs()

            self.calc_eigen_coeffs(F_z_t0)

        F_z_t = self.calculate_abundance(tau)

        self.F_z_t = F_z_t

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
    Ne
        xarray.DataArray of electron density as a profile of a user-chosen coordinate.
    Te
        xarray.DataArray of electron temperature as a profile of a user-chosen
        coordinate.
    F_z_t
        xarray.DataArray of fractional abundance of all ionisation charges of given
        impurity element.
    PRC
        xarray.DataArray of radiated power of charge exchange emission of all relevant
        ionisation charges of given impurity element. (Optional)
    unit_testing
        Boolean for unit testing purposes
        (whether to call ordered_setup in __init__ or not) (Optional)
    Nh
        xarray.DataArray of thermal hydrogen number density as a profile of a
        user-chosen coordinate.
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
    interpolation_bounds_check(inputted_data, imported_data)
        Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(PLT,PRC,PRB).
    interpolate_power()
        Interpolates the various powers based on inputted Ne and Te.
    __call__()
        Calculates total radiated power of all ionisation charges of a given
        impurity element.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("line_power_coeffecient", "impurity_element"),
        ("recombination_power_coeffecient", "impurity_element"),
        ("number_density", "electrons"),
        ("number_density", "thermal_hydrogen"),
        ("temperature", "electrons"),
        ("charge-exchange_power_coeffecient", "impurity_element"),
        ("fractional_abundance", "impurity_element"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("total_radiated power loss", "impurity_element"),
    ]

    def __init__(
        self,
        PLT: DataArray,
        PRB: DataArray,
        # Ne: DataArray,
        # Te: DataArray,
        # F_z_t: DataArray,
        PRC: DataArray = None,
        # Nh: Optional[DataArray] = None,
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

        for ikey, ival in imported_data.items():
            input_check(var_name=ikey, var_to_check=ival, var_type=DataArray)

        shape_check(imported_data)

    def interpolation_bounds_check(
        self,
        Ne: DataArray,
        Te: DataArray,
    ):
        """Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(PLT,PRC,PRB).

        Parameters
        ----------
        imported_data
            Imported data (PLT, PRC, PRB)
        inputted_data
            Inputted data (Ne, Nh, Te) (Nh is tested in self.__init__())
        """

        imported_data = {}
        imported_data["PLT"] = self.PLT
        imported_data["PRB"] = self.PRB
        if self.PRC is not None:
            imported_data["PRC"] = self.PRC

        inputted_data = {}

        input_check("Ne", Ne, DataArray, greater_than_or_equal_zero=True)
        inputted_data["Ne"] = Ne

        input_check("Te", Te, DataArray, greater_than_or_equal_zero=False)
        inputted_data["Te"] = Te

        shape_check(inputted_data)

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] <= np.max(val.coords["electron_density"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] >= np.min(val.coords["electron_density"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] <= np.max(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] >= np.min(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise ValueError(
                f"Inputted electron temperature is smaller than the \
                    minimum interpolation range in {key}"
            )

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

        self.interpolation_bounds_check(Ne, Te)

        self.Ne, self.Te = Ne, Te  # type: ignore

        PLT_spec = self.PLT.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        PLT_spec = PLT_spec

        if self.PRC is not None:
            PRC_spec = self.PRC.indica.interp2d(
                electron_temperature=Te,
                electron_density=Ne,
                method="cubic",
                assume_sorted=True,
            )
            PRC_spec = PRC_spec
        else:
            PRC_spec = None

        PRB_spec = self.PRB.indica.interp2d(
            electron_temperature=Te,
            electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        PRB_spec = PRB_spec

        self.PLT, self.PRC, self.PRB = PLT_spec, PRC_spec, PRB_spec
        self.num_of_ion_charges = self.PLT.shape[0] + 1

        return PLT_spec, PRC_spec, PRB_spec, self.num_of_ion_charges

    def calculate_power_loss(self, Ne: DataArray, Nh: DataArray, F_z_t: DataArray):
        inputted_data = {}
        inputted_data["Ne"] = Ne

        if Nh is not None:
            if self.PRC is None:
                raise ValueError(
                    "Nh (Thermal hydrogen density) cannot be given when \
                    PRC (effective charge exchange power) at initialisation \
                    is None."
                )
            input_check("Nh", Nh, DataArray, greater_than_or_equal_zero=True)
            inputted_data["Nh"] = Nh
            self.Nh = Nh  # type: ignore
        elif self.PRC is not None:
            raise ValueError(
                "Nh (Thermal hydrogen density) cannot be None when\
                CCD (effective charge exchange power) at initialisation \
                is not None."
            )

        if len(inputted_data) > 1:
            shape_check(inputted_data)

        input_check("F_z_t", F_z_t, DataArray, greater_than_or_equal_zero=True)
        try:
            assert not np.iscomplexobj(F_z_t)
        except AssertionError:
            raise ValueError(
                "Inputted F_z_t is a complex type or array of complex numbers, \
                    must be real"
            )
        self.F_z_t = F_z_t  # type: ignore

        self.x1_coord = self.PLT.coords[
            [k for k in self.PLT.dims if k != "ion_charges"][0]
        ]

        x1_coord = self.x1_coord

        # Mypy complaints about F_z_t not being subscriptable since it thinks
        # it's a NoneType have been suppresed. This is because F_z_t is tested
        # to be a DataArray with elements greater than zero.
        # (in the input_check() above)

        cooling_factor = xr.zeros_like(self.F_z_t)
        for ix1 in range(x1_coord.size):
            icharge = 0
            cooling_factor[icharge, ix1] = (
                self.PLT[icharge, ix1] * self.F_z_t[icharge, ix1]  # type: ignore
            )
            for icharge in range(1, self.num_of_ion_charges - 1):
                cooling_factor[icharge, ix1] += (
                    self.PLT[icharge, ix1]
                    + (
                        (Nh[ix1] / Ne[ix1]) * self.PRC[icharge - 1, ix1]
                        if (self.PRC is not None) and (Nh is not None)
                        else 0.0
                    )
                    + self.PRB[icharge - 1, ix1]
                ) * self.F_z_t[
                    icharge, ix1
                ]  # type: ignore

            icharge = self.num_of_ion_charges - 1
            cooling_factor[icharge, ix1] += (
                (
                    (Nh[ix1] / Ne[ix1]) * self.PRC[icharge - 1, ix1]
                    if (self.PRC is not None) and (Nh is not None)
                    else 0.0
                )
                + self.PRB[icharge - 1, ix1]
            ) * self.F_z_t[
                icharge, ix1
            ]  # type: ignore

        self.cooling_factor = cooling_factor

        return cooling_factor

    def __call__(  # type: ignore
        self,
        Ne: DataArray,
        Te: DataArray,
        Nh: DataArray = None,
        F_z_t: DataArray = None,
        full_run: bool = True,
    ):
        """Calculates total radiated power of all ionisation charges of a given
        impurity element.

        Returns
        -------
        cooling_factor
            Total radiated power of all ionisation charges.
        """

        if full_run:
            self.interpolate_power(Ne, Te)

        cooling_factor = self.calculate_power_loss(Ne, Nh, F_z_t)  # type: ignore

        return cooling_factor
