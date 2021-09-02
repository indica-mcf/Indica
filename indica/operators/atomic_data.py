import copy
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy
from xarray import DataArray

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
        raise AssertionError(f"{key1} and {key2} are not the same shape")


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
    __call__(tau)
        Calculates the fractional abundance of all ionisation charges at time tau.
    ordered_setup()
        Sets up data for calculation in correct order.
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
        Ne: DataArray,
        Te: DataArray,
        F_z_t0: DataArray = None,
        Nh: DataArray = None,
        CCD: DataArray = None,
        unit_testing: bool = False,
        sess: session.Session = session.global_session,
    ):
        """Initialises FractionalAbundance class and additionally performs error
        checking on inputs.
        """
        super().__init__(sess)
        self.num_of_ion_charges = 0
        self.ionisation_balance_matrix = None
        self.F_z_tinf = None
        self.F_z_t0 = None
        self.F_z_t = None
        self.eig_vals = None
        self.eig_vecs = None
        self.eig_coeffs = None

        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD

        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te

        imported_data = {}
        imported_data["SCD"] = self.SCD
        imported_data["ACD"] = self.ACD
        if self.CCD is not None:
            imported_data["CCD"] = self.CCD

        inputted_data = {}
        inputted_data["Ne"] = self.Ne
        if self.Nh is not None:
            inputted_data["Nh"] = self.Nh
        inputted_data["Te"] = self.Te

        for ikey, ival in dict(inputted_data, **imported_data).items():
            input_check(var_name=ikey, var_to_check=ival, var_type=DataArray)

        if F_z_t0 is not None:
            input_check("F_z_t0", F_z_t0, DataArray, greater_than_or_equal_zero=True)

            try:
                assert F_z_t0.ndim < 3
            except AssertionError:
                raise AssertionError("F_z_t0 must be at most 2-dimensional.")
        self.F_z_t0 = F_z_t0

        self.interpolation_bounds_check(imported_data, inputted_data)

        shape_check(inputted_data)
        shape_check(imported_data)

        if not unit_testing:
            self.ordered_setup()

    def interpolation_bounds_check(self, imported_data, inputted_data):
        """Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(SCD,CCD,ACD,PLT,PRC,PRB).

        Parameters
        ----------
        imported_data
            Imported data (SCD, ACD, CCD, PLT, PRC, PRB)
        inputted_data
            Inputted data (Ne, Nh, Te) (Nh is tested in self.__init__())
        """
        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] <= np.max(val.coords["electron_density"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] >= np.min(val.coords["electron_density"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] <= np.max(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] >= np.min(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise AssertionError(
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
            Number of ionisation charges.
        """
        Ne, Te = self.Ne, self.Te

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
        Ne, Nh = self.Ne, self.Nh

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
                -Ne * SCD[icharge],
                Ne * ACD[icharge]
                + (Nh * CCD[icharge] if Nh is not None and CCD is not None else 0.0),
            ]
        )
        for icharge in range(1, num_of_ion_charges - 1):
            ionisation_balance_matrix[icharge, icharge - 1 : icharge + 2] = np.array(
                [
                    Ne * SCD[icharge - 1],
                    -Ne * (SCD[icharge] + ACD[icharge - 1])
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
                -Ne * (ACD[icharge - 1])
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

        if self.F_z_t0 is None:
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
                    x1_coord,
                ],
                dims=["ion_charges", x1_coord.dims[0]],
            )
        else:
            F_z_t0 = self.F_z_t0 / np.sum(self.F_z_t0, axis=0)
            F_z_t0 = F_z_t0.as_type(dtype=np.complex128)

            F_z_t0 = DataArray(
                data=F_z_t0.values,
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

        self.F_z_t0 = F_z_t0

        return eig_coeffs, F_z_t0

    def __call__(
        self,
        tau,
    ):
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
        try:
            assert np.any(np.abs(tau) != np.inf)
        except AssertionError:
            raise AssertionError("Given time value, tau, cannot be infinity")

        try:
            assert np.all(tau >= 0)
        except AssertionError:
            raise AssertionError("Given time value, tau, cannot be negative")

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

        return F_z_t

    def ordered_setup(self):
        """Sets up data for calculation in correct order."""
        self.interpolate_rates()

        self.calc_ionisation_balance_matrix()

        self.calc_F_z_tinf()

        self.calc_eigen_vals_and_vecs()

        self.calc_eigen_coeffs()


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
    Nh
        xarray.DataArray of thermal hydrogen number density as a profile of a
        user-chosen coordinate.
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
    F_z_t_check(self, F_z_t)
        Checks that inputted fractional abundance has valid values.
    Nh_check(Nh)
        Checks that the inputted thermal hydrogen number density has valid values.
    interpolation_bounds_check(inputted_data, imported_data)
        Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(PLT,PRC,PRB).
    interpolate_power()
        Interpolates the various powers based on inputted Ne and Te.
    __call__()
        Calculates total radiated power of all ionisation charges of a given
        impurity element.
    ordered_setup()
        Sets up data for calculation in correct order.
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
        Ne: DataArray,
        Te: DataArray,
        F_z_t: DataArray,
        PRC: Optional[DataArray] = None,
        unit_testing: Optional[bool] = False,
        Nh: Optional[DataArray] = None,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess)
        self.PLT = PLT
        self.PRC = PRC
        self.PRB = PRB
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te
        self.num_of_ion_charges = 0

        imported_data = {}
        imported_data["PLT"] = self.PLT
        imported_data["PRB"] = self.PRB
        if self.PRC is not None:
            imported_data["PRC"] = self.PRC
        inputted_data = {}
        inputted_data["Ne"] = self.Ne
        if self.Nh is not None:
            inputted_data["Nh"] = self.Nh
        inputted_data["Te"] = self.Te

        for ikey, ival in dict(inputted_data, **imported_data).items():
            input_check(var_name=ikey, var_to_check=ival, var_type=DataArray)

        input_check("F_z_t", F_z_t, DataArray, greater_than_or_equal_zero=True)
        try:
            assert not np.iscomplexobj(F_z_t)
        except AssertionError:
            raise AssertionError(
                "Inputted F_z_t is a complex type or array of complex numbers, \
                    must be real"
            )
        self.F_z_t = F_z_t

        self.interpolation_bounds_check(imported_data, inputted_data)

        shape_check(inputted_data)
        shape_check(imported_data)

        if not unit_testing:
            self.ordered_setup()

    def interpolation_bounds_check(self, imported_data, inputted_data):
        """Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(PLT,PRC,PRB).

        Parameters
        ----------
        imported_data
            Imported data (PLT, PRC, PRB)
        inputted_data
            Inputted data (Ne, Nh, Te) (Nh is tested in self.__init__())
        """

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] <= np.max(val.coords["electron_density"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"] >= np.min(val.coords["electron_density"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] <= np.max(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"] >= np.min(val.coords["electron_temperature"])
                )
        except AssertionError:
            raise AssertionError(
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

    def interpolate_power(self):
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
        """

        # Ne, Te = np.log10(self.Ne), np.log10(self.Te)
        Ne, Te = self.Ne, self.Te

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

    def __call__(self):
        """Calculates total radiated power of all ionisation charges of a given
        impurity element.

        Returns
        -------
        cooling_factor
            Total radiated power of all ionisation charges.
        F_z_t
            Fractional abundance, either user-provided or fully stripped
            eg. [0.0, 0.0, 0.0, 0.0, 1.0] for Beryllium.
        """

        Ne, Nh = self.Ne, self.Nh

        self.x1_coord = self.PLT.coords[
            [k for k in self.PLT.dims if k != "ion_charges"][0]
        ]

        x1_coord = self.x1_coord

        if self.F_z_t is None:
            F_z_t = np.zeros((self.num_of_ion_charges, x1_coord.size))
            F_z_t[-1, :] = np.array([1.0 for i in range(x1_coord.size)])
        else:
            F_z_t = self.F_z_t / np.linalg.norm(self.F_z_t)

        cooling_factor = np.zeros(x1_coord.size)
        for ix1 in range(x1_coord.size):
            icharge = 0
            cooling_factor[ix1] = (self.PLT[icharge, ix1]) * F_z_t[icharge, ix1]
            for icharge in range(1, self.num_of_ion_charges - 1):
                cooling_factor[ix1] += (
                    self.PLT[icharge, ix1]
                    + (
                        (Nh[ix1] / Ne[ix1]) * self.PRC[icharge - 1, ix1]
                        if (self.PRC is not None) and (Nh is not None)
                        else 0.0
                    )
                    + self.PRB[icharge - 1, ix1]
                ) * F_z_t[icharge, ix1]
            icharge = self.num_of_ion_charges - 1
            cooling_factor[ix1] += (
                (
                    (Nh[ix1] / Ne[ix1]) * self.PRC[icharge - 1, ix1]
                    if (self.PRC is not None) and (Nh is not None)
                    else 0.0
                )
                + self.PRB[icharge - 1, ix1]
            ) * F_z_t[icharge, ix1]

        self.cooling_factor = cooling_factor

        return cooling_factor

    def ordered_setup(self):
        """Sets up data for calculation in correct order."""
        self.interpolate_power()
