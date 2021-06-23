import copy
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import scipy
from xarray.core.dataarray import DataArray

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType

np.set_printoptions(edgeitems=10, linewidth=100)


class FractionalAbundance(Operator):
    """Calculate fractional abundance for all ionisation stages of a given element.

    Parameters
    ----------
    SCD
        xarray.DataArray of effective ionisation rate coefficients of all relevant
        ionisation stages of given impurity element.
    ACD
        xarray.DataArray of effective recombination rate coefficients of all relevant
        ionisation stages of given impurity element.
    CCD
        xarray.DataArray of charge exchange cross coupling coefficients of all relevant
        ionisation stages of given impurity element.
    Ne
        xarray.DataArray of electron density as a profile of rho
    Nh
        xarray.DataArray of thermal hydrogen as a profile of rho
    Te
        xarray.DataArray of electron temperature as a profile of rho
    N_z_t0
        Optional initial fractional abundance for given impurity element.
    sess
        Object representing this session of calculations with the library.
        Holds and communicates provenance information.

    Attributes
    ----------
    ARGUMENT_TYPES: List[DataType]
        Ordered list of the types of data expected for each argument of the
        operator.
    RESULT_TYPES: List[DataType]
        Ordered list of the types of data returned by the operator.

    Returns
    -------
    N_z_t
        xarray.DataArray of fractional abundance of all ionisation stages of given
        impurity element.

    Methods
    -------
    N_z_t0_check(N_z_t0)
        Checks that inputted initial fractional abundance has valid values.
    input_check(imported_data, inputted_data)
        Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(SCD,CCD,ACD,PLT,PRC,PRB).
    interpolate_rates()
        Interpolates rates based on inputted Ne and Te, also determines the number
        of ionisation stages for a given element.
    calc_ionisation_balance_matrix()
        Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation stages.
    calc_N_z_tinf()
        Calculates the equilibrium fractional abundance of all ionisation stages,
        N_z(t=infinity) used for the final time evolution equation.
    calc_eigen_vals_and_vecs()
        Calculates the eigenvalues and eigenvectors of the ionisation balance matrix.
    calc_eigen_coeffs()
        Calculates the coefficients from the eigenvalues and eigenvectors for the time
        evolution equation.
    ordered_setup()
        Sets up data for calculation in correct order.
    __call__(tau)
        Calculates the fractional abundance of all ionisation stages at time tau.
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
        CCD: DataArray,
        Ne: DataArray,
        Nh: DataArray,
        Te: DataArray,
        N_z_t0: np.ndarray = None,
        unit_testing: bool = False,
        sess: session.Session = session.global_session,
    ):
        """Initialises FractionalAbundance class and additionally performs error
        checking on inputs.
        """
        super().__init__(sess)
        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD
        self.num_of_stages = 0
        self.ionisation_balance_matrix = None
        self.N_z_tinf = None
        self.N_z_t0 = None
        self.N_z_t = None
        self.eig_vals = None
        self.eig_vecs = None
        self.eig_coeffs = None
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te

        imported_data = {}
        imported_data["SCD"] = self.SCD
        imported_data["ACD"] = self.ACD
        imported_data["CCD"] = self.CCD
        inputted_data = {}
        inputted_data["Ne"] = self.Ne
        inputted_data["Nh"] = self.Nh
        inputted_data["Te"] = self.Te

        self.N_z_t0_check(N_z_t0)
        self.N_z_t0 = N_z_t0

        try:
            for key, val in imported_data.items():
                assert isinstance(val, DataArray)
        except AssertionError:
            raise AssertionError(f"{key} is not an instance of xarray.DataArray")

        try:
            for key, val in inputted_data.items():
                assert isinstance(val, DataArray)
        except AssertionError:
            raise AssertionError(f"{key} is not an instance of xarray.DataArray")

        try:
            for key1, val1 in inputted_data.items():
                for key2, val2 in inputted_data.items():
                    assert val1.shape == val2.shape
        except AssertionError:
            raise AssertionError(f"{key1} and {key2} are not the same shape")

        try:
            assert np.all(inputted_data["Nh"] != np.nan)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be NaN"
            )

        try:
            assert np.all(np.abs(inputted_data["Nh"]) != np.inf)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be +inf or -inf"
            )

        try:
            assert np.all(inputted_data["Nh"] >= 0)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be negative"
            )

        self.input_check(imported_data, inputted_data)

        if not unit_testing:
            self.ordered_setup()

    def N_z_t0_check(self, N_z_t0):
        """Checks that inputted initial fractional abundance has valid values.

        Parameters
        ----------
        N_z_t0
            Initial fractional abundance to check.
        """
        if N_z_t0 is None:
            return

        try:
            assert isinstance(N_z_t0, np.ndarray)
        except AssertionError:
            raise AssertionError(
                "Initial fractional abundance must be a numpy array\
                (numpy.ndarray)."
            )

        try:
            assert np.all(N_z_t0 >= 0)
        except AssertionError:
            raise AssertionError(
                "Cannot have any negative values in the initial \
                fractional abundance data."
            )

        try:
            assert N_z_t0.ndim == 1
        except AssertionError:
            raise AssertionError("Initial fractional abundance must be 1-dimensional.")

        try:
            assert np.all(N_z_t0 != np.nan)
        except AssertionError:
            raise AssertionError(
                "Initial fractional abundance cannot contain any \
                NaNs."
            )

        try:
            assert np.all(np.abs(N_z_t0) != np.inf)
        except AssertionError:
            raise AssertionError(
                "Initial fractional abundance cannot contain any \
                infinities."
            )

    def input_check(self, imported_data, inputted_data):
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
                    inputted_data["Ne"]
                    <= np.max(np.power(10, val.coords["log10_electron_density"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"]
                    >= np.min(np.power(10, val.coords["log10_electron_density"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"]
                    <= np.max(np.power(10, val.coords["log10_electron_temperature"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"]
                    >= np.min(np.power(10, val.coords["log10_electron_temperature"]))
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
        of ionisation stages for a given element.

        Returns
        -------
        SCD_spec
            Interpolated effective ionisation rate coefficients.
        ACD_spec
            Interpolated effective recombination rate coefficients.
        CCD_spec
            Interpolated charge exchange cross coupling coefficients.
        num_of_stages
            Number of ionisation stages.
        """
        Ne, Te = np.log10(self.Ne), np.log10(self.Te)

        SCD_spec = self.SCD.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        SCD_spec = np.power(10.0, SCD_spec)

        CCD_spec = self.CCD.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        CCD_spec = np.power(10.0, CCD_spec)

        ACD_spec = self.ACD.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        ACD_spec = np.power(10.0, ACD_spec)

        self.SCD, self.ACD, self.CCD = SCD_spec, ACD_spec, CCD_spec
        self.num_of_stages = self.SCD.shape[0] + 1

        return SCD_spec, ACD_spec, CCD_spec, self.num_of_stages

    def calc_ionisation_balance_matrix(
        self,
    ):
        """Calculates the ionisation balance matrix that defines the differential equation
        that defines the time evolution of the fractional abundance of all of the
        ionisation stages.

        Returns
        -------
        ionisation_balance_matrix
            Matrix representing coefficients of the differential equation governing
            the time evolution of the ionisation balance.
        """
        Ne, Nh = self.Ne, self.Nh

        num_of_stages = self.num_of_stages
        SCD, ACD, CCD = self.SCD, self.ACD, self.CCD

        dims = (
            num_of_stages,
            num_of_stages,
            *SCD.coords["rho"].shape,
        )

        ionisation_balance_matrix = np.zeros(dims)

        istage = 0
        ionisation_balance_matrix[istage, istage : istage + 2] = np.array(
            [-Ne * SCD[istage], Ne * ACD[istage] + Nh * CCD[istage]]
        )
        for istage in range(1, num_of_stages - 1):
            ionisation_balance_matrix[istage, istage - 1 : istage + 2] = np.array(
                [
                    Ne * SCD[istage - 1],
                    -Ne * (SCD[istage] + ACD[istage - 1]) - Nh * CCD[istage - 1],
                    Ne * ACD[istage] + Nh * CCD[istage],
                ]
            )
        istage = num_of_stages - 1
        ionisation_balance_matrix[istage, istage - 1 : istage + 1] = np.array(
            [Ne * SCD[istage - 1], -Ne * (ACD[istage - 1]) - Nh * CCD[istage - 1]]
        )

        ionisation_balance_matrix = np.squeeze(ionisation_balance_matrix)
        self.ionisation_balance_matrix = ionisation_balance_matrix

        return ionisation_balance_matrix

    def calc_N_z_tinf(
        self,
    ):
        """Calculates the equilibrium fractional abundance of all ionisation stages,
        N_z(t=infinity) used for the final time evolution equation.

        Returns
        -------
        N_z_tinf
            Fractional abundance at equilibrium.
        """
        rho = self.Ne.coords["rho"]
        ionisation_balance_matrix = self.ionisation_balance_matrix

        null_space = np.zeros((self.num_of_stages, rho.size))
        N_z_tinf = np.zeros((self.num_of_stages, rho.size))

        for irho in range(rho.size):
            null_space[:, irho, np.newaxis] = scipy.linalg.null_space(
                ionisation_balance_matrix[:, :, irho]
            )

        # Complex type casting for compatibility with eigen calculation results later.
        N_z_tinf = np.abs(null_space).astype(dtype=np.complex128)
        self.N_z_tinf = N_z_tinf

        return N_z_tinf

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
        rho = self.Ne.coords["rho"]
        eig_vals = np.zeros((self.num_of_stages, rho.size), dtype=np.complex128)
        eig_vecs = np.zeros(
            (self.num_of_stages, self.num_of_stages, rho.size), dtype=np.complex128
        )

        for irho in range(rho.size):
            eig_vals[:, irho], eig_vecs[:, :, irho] = scipy.linalg.eig(
                self.ionisation_balance_matrix[:, :, irho],
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
        N_z_t0
            Initial fractional abundance, either user-provided or fully neutral
            eg. [1.0, 0.0, 0.0, 0.0, 0.0] for Beryllium.
        """
        rho = self.Ne.coords["rho"]

        if self.N_z_t0 is None:
            N_z_t0 = np.zeros(self.N_z_tinf.shape, dtype=np.complex128)
            N_z_t0[0, :] = np.array([1.0 + 0.0j for i in range(rho.size)])
        else:
            N_z_t0 = self.N_z_t0 / np.linalg.norm(self.N_z_t0)
            N_z_t0 = N_z_t0.as_type(dtype=np.complex128)

        eig_vals = self.eig_vals
        eig_vecs_inv = np.zeros(self.eig_vecs.shape, dtype=np.complex128)
        for irho in range(rho.size):
            eig_vecs_inv[:, :, irho] = np.linalg.pinv(
                np.transpose(self.eig_vecs[:, :, irho])
            )

        boundary_conds = N_z_t0 - self.N_z_tinf

        eig_coeffs = np.zeros(eig_vals.shape, dtype=np.complex128)
        for irho in range(rho.size):
            eig_coeffs[:, irho] = np.dot(
                boundary_conds[:, irho], eig_vecs_inv[:, :, irho]
            )

        self.eig_coeffs = eig_coeffs
        self.N_z_t0 = N_z_t0

        return eig_coeffs, N_z_t0

    def __call__(
        self,
        tau,
    ):
        """Calculates the fractional abundance of all ionisation stages at time tau.

        Parameters
        ----------
        tau
            Time after t0 (t0 is defined as the time at which N_z_t0 is taken).

        Returns
        -------
        N_z_t
            Fractional abundance at tau.
        """
        try:
            assert np.abs(tau) != np.inf
        except AssertionError:
            raise AssertionError("Given time value, tau, cannot be infinity")

        try:
            assert tau >= 0
        except AssertionError:
            raise AssertionError("Given time value, tau, cannot be negative")

        rho = self.Ne.coords["rho"]
        N_z_t = copy.deepcopy(self.N_z_tinf)
        for irho in range(rho.size):
            for istage in range(self.num_of_stages):
                N_z_t[:, irho] += (
                    self.eig_coeffs[istage, irho]
                    * np.exp(self.eig_vals[istage, irho] * tau)
                    * self.eig_vecs[:, istage, irho]
                )

        self.N_z_t = np.real(N_z_t)

        return np.real(N_z_t)

    def ordered_setup(self):
        """Sets up data for calculation in correct order."""
        self.interpolate_rates()

        self.calc_ionisation_balance_matrix()

        self.calc_N_z_tinf()

        self.calc_eigen_vals_and_vecs()

        self.calc_eigen_coeffs()

    # def __call__(self, tau):
    #     """Performs the calculations for fractional abundance.

    #     Parameters
    #     ----------
    #     tau
    #         Time after t0 (t0 is defined as the time at which N_z_t0 is taken).

    #     Returns
    #     -------
    #     N_z_t
    #         Fractional abundance at tau.
    #     """
    #     self.calc_N_z_t(tau)

    #     return self.N_z_t


class PowerLoss(Operator):
    """Calculate the total power loss associated with a given impurity element

    Parameters
    ----------
    PLT
        xarray.DataArray of radiated power of line emission from excitation of all
        relevant ionisation stages of given impurity element.
    PRC
        xarray.DataArray of radiated power of charge exchange emission of all relevant
        ionisation stages of given impurity element.
    PRB
        xarray.DataArray of radiated power from recombination and bremsstrahlung of
        given impurity element.
    Ne
        xarray.DataArray of electron density as a profile of rho
    Nh
        xarray.DataArray of thermal hydrogen as a profile of rho
    Te
        xarray.DataArray of electron temperature as a profile of rho
    N_z_t
        xarray.DataArray of fractional abundance of all ionisation stages of given
        impurity element.

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
        xarray.DataArray of total radiated power loss of all ionisation stages of given
        impurity element.

    Methods
    -------
    N_z_t_check(self, N_z_t)
        Checks that inputted fractional abundance has valid values.
    input_check(inputted_data, imported_data)
        Checks that inputted data (Ne and Te) has values that are within the
        interpolation ranges specified inside imported_data(PLT,PRC,PRB).
    interpolate_power()
        Interpolates the various powers based on inputted Ne and Te.
    __call__()
        Calculates total radiated power of all ionisation stages of a given
        impurity element.
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("line_power_coeffecient", "impurity_element"),
        ("recombination_power_coeffecient", "impurity_element"),
        ("charge-exchange_power_coeffecient", "impurity_element"),
        ("number_density", "electrons"),
        ("temperature", "electrons"),
        ("number_density", "thermal_hydrogen"),
        ("fractional_abundance", "impurity_element"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("total_radiated power loss", "impurity_element"),
    ]

    def __init__(
        self,
        PLT: DataArray,
        PRC: DataArray,
        PRB: DataArray,
        Ne: DataArray,
        Nh: DataArray,
        Te: DataArray,
        N_z_t: DataArray = None,
        unit_testing: bool = False,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess)
        self.PLT = PLT
        self.PRC = PRC
        self.PRB = PRB
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te
        self.num_of_stages = 0

        imported_data = {}
        imported_data["PLT"] = self.PLT
        imported_data["PRC"] = self.PRC
        imported_data["PRB"] = self.PRB
        inputted_data = {}
        inputted_data["Ne"] = self.Ne
        inputted_data["Nh"] = self.Nh
        inputted_data["Te"] = self.Te

        self.N_z_t_check(N_z_t)
        self.N_z_t = N_z_t

        external_calc_data = {}
        external_calc_data["N_z_t"] = self.N_z_t

        try:
            for key, val in imported_data.items():
                assert isinstance(val, DataArray)
        except AssertionError:
            raise AssertionError(f"{key} is not an instance of xarray.DataArray")

        try:
            for key, val in inputted_data.items():
                assert isinstance(val, DataArray)
        except AssertionError:
            raise AssertionError(f"{key} is not an instance of xarray.DataArray")

        try:
            for key1, val1 in inputted_data.items():
                for key2, val2 in inputted_data.items():
                    assert val1.shape == val2.shape
        except AssertionError:
            raise AssertionError(f"{key1} and {key2} are not the same shape")

        try:
            assert np.all(inputted_data["Nh"] != np.nan)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be NaN"
            )

        try:
            assert np.all(np.abs(inputted_data["Nh"]) != np.inf)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be +inf or -inf"
            )

        try:
            assert np.all(inputted_data["Nh"] >= 0)
        except AssertionError:
            raise AssertionError(
                "Inputted thermal hydrogen number density cannot be negative"
            )

        self.input_check(imported_data, inputted_data)

        if not unit_testing:
            self.ordered_setup()

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

    def N_z_t_check(self, N_z_t):
        """Checks that initial fractional abundance has valid values.

        Parameters
        ----------
        N_z_t
            Fractional abundance to check.
        """
        if N_z_t is None:
            return

        try:
            assert isinstance(N_z_t, DataArray)
        except AssertionError:
            raise AssertionError("Fractional abundance must be a xarray DataArray.")

        try:
            assert np.all(N_z_t >= 0)
        except AssertionError:
            raise AssertionError(
                "Cannot have any negative values in the fractional abundance data."
            )

        try:
            assert N_z_t.ndim == 1
        except AssertionError:
            raise AssertionError("Fractional abundance must be 1-dimensional.")

        try:
            assert np.all(N_z_t != np.nan)
        except AssertionError:
            raise AssertionError(
                "Fractional abundance cannot contain any NaNs (invalid numbers)."
            )

        try:
            assert np.all(np.abs(N_z_t) != np.inf)
        except AssertionError:
            raise AssertionError("Fractional abundance cannot contain any infinities.")

    def input_check(self, imported_data, inputted_data):
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
                    inputted_data["Ne"]
                    <= np.max(np.power(10, val.coords["log10_electron_density"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Ne"]
                    >= np.min(np.power(10, val.coords["log10_electron_density"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron number density is smaller than the \
                    minimum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"]
                    <= np.max(np.power(10, val.coords["log10_electron_temperature"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron temperature is larger than the \
                    maximum interpolation range in {key}"
            )

        try:
            for key, val in imported_data.items():
                assert np.all(
                    inputted_data["Te"]
                    >= np.min(np.power(10, val.coords["log10_electron_temperature"]))
                )
        except AssertionError:
            raise AssertionError(
                f"Inputted electron temperature is smaller than the \
                    minimum interpolation range in {key}"
            )

    def interpolate_power(self):
        """Interpolates the various powers based on inputted Ne and Te.

        Returns
        -------
        PLT_spec
            Interpolated radiated power of line emission from excitation of all
            relevant ionisation stages.
        PRC_spec
            Interpolated radiated power of charge exchange emission of all relevant
            ionisation stages.
        PRB_spec
            Interpolated radiated power from recombination and bremsstrahlung.
        """

        Ne, Te = np.log10(self.Ne), np.log10(self.Te)

        PLT_spec = self.PLT.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        PLT_spec = np.power(10, PLT_spec)

        PRC_spec = self.PRC.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        PRC_spec = np.power(10, PRC_spec)

        PRB_spec = self.PRB.indica.interp2d(
            log10_electron_temperature=Te,
            log10_electron_density=Ne,
            method="cubic",
            assume_sorted=True,
        )
        PRB_spec = np.power(10, PRB_spec)

        self.PLT, self.PRC, self.PRB = PLT_spec, PRC_spec, PRB_spec
        self.num_of_stages = self.PLT.shape[0] + 1

        return PLT_spec, PRC_spec, PRB_spec, self.num_of_stages

    def __call__(self):
        """Calculates total radiated power of all ionisation stages of a given
        impurity element.

        Returns
        -------
        cooling_factor
            Total radiated power of all ionisation stages.
        N_z_t
            Fractional abundance, either user-provided or fully stripped
            eg. [0.0, 0.0, 0.0, 0.0, 1.0] for Beryllium.
        """

        Ne, Nh = self.Ne, self.Nh

        rho = self.Ne.coords["rho"]

        if self.N_z_t is None:
            N_z_t = np.zeros((self.num_of_stages, rho.size))
            N_z_t[-1, :] = np.array([1.0 for i in range(rho.size)])
        else:
            N_z_t = self.N_z_t / np.linalg.norm(self.N_z_t)

        rho = Ne.coords["rho"]

        cooling_factor = np.zeros(rho.size)
        for irho in range(rho.size):
            istage = 0
            cooling_factor[irho] = (self.PLT[istage, irho]) * N_z_t[istage, irho]
            for istage in range(1, self.num_of_stages - 1):
                cooling_factor[irho] += (
                    self.PLT[istage, irho]
                    + self.PRC[istage - 1, irho]
                    + (Nh[irho] / Ne[irho]) * self.PRB[istage - 1, irho]
                ) * N_z_t[istage, irho]
            istage = self.num_of_stages - 1
            cooling_factor[irho] += (
                self.PRC[istage - 1, irho]
                + (Nh[irho] / Ne[irho]) * self.PRB[istage - 1, irho]
            ) * N_z_t[istage, irho]

        self.cooling_factor = cooling_factor

        return cooling_factor

    def ordered_setup(self):
        """Sets up data for calculation in correct order."""
        self.interpolate_power()

    # def __call__(self):
    #     """Performs the calculations for total radiated power loss.

    #     Returns
    #     -------
    #     cooling_factor
    #         Total radiated power loss of all ionisation stages of
    #         given impurity element.
    #     """
    #     self.calc_cooling_factor()

    #     return self.cooling_factor
