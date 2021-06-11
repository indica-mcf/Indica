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
    """Calculate fractional abundance for all ionisation stages of a given element

    Parameters
    ----------
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
    """

    ARGUMENT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("ionisation_rate", "impurity_element"),
        ("recombination_rate", "impurity_element"),
        ("charge-exchange_rate", "impurity_element"),
        ("line_power_coeffecient", "impurity_element"),
        ("recombination_power_coeffecient", "impurity_element"),
        ("charge-exchange_power_coeffecient", "impurity_element"),
        ("number_density", "electrons"),
        ("temperature", "electrons"),
        ("number_density", "thermal_hydrogen"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("fractional abundance", "impurity_element"),
        ("total radiated power loss", "impurity_element"),
    ]

    def __init__(
        self,
        SCD: DataArray,
        ACD: DataArray,
        CCD: DataArray,
        PLT: DataArray,
        PRC: DataArray,
        PRB: DataArray,
        Ne: DataArray,
        Nh: DataArray,
        Te: DataArray,
        sess: session.Session = session.global_session,
    ):
        super().__init__(sess)
        self.SCD = SCD
        self.ACD = ACD
        self.CCD = CCD
        self.PLT = PLT
        self.PRC = PRC
        self.PRB = PRB
        self.num_of_stages = 0
        self.ionisation_balance_matrix = None
        self.N_z_tinf = None
        self.N_z_t0 = None
        self.N_z_t = None
        self.eig_vals = None
        self.eig_vecs = None
        self.eig_coeffs = None
        self.cooling_factor = None
        self.Ne = Ne
        self.Nh = Nh
        self.Te = Te

        imported_data = {}
        imported_data["SCD"] = self.SCD
        imported_data["ACD"] = self.ACD
        imported_data["CCD"] = self.CCD
        imported_data["PLT"] = self.PLT
        imported_data["PRC"] = self.PRC
        imported_data["PRB"] = self.PRB
        inputted_data = {}
        inputted_data["Ne"] = self.Ne
        inputted_data["Nh"] = self.Nh
        inputted_data["Te"] = self.Te

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

    def input_check(self, imported_data, inputted_data):
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
        return (
            ("fractional abundance", "impurity_element"),
            ("total radiated power loss", "impurity_element"),
        )

    def interpolate_rates(
        self,
    ):
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
        rho = self.Ne.coords["rho"]
        ionisation_balance_matrix = self.ionisation_balance_matrix

        null_space = np.zeros((self.num_of_stages, rho.size))
        N_z_tinf = np.zeros((self.num_of_stages, rho.size))

        for irho in range(rho.size):
            null_space[:, irho, np.newaxis] = scipy.linalg.null_space(
                ionisation_balance_matrix[:, :, irho]
            )

        N_z_tinf = np.abs(null_space).astype(dtype=np.complex128)
        self.N_z_tinf = N_z_tinf

        return N_z_tinf

    def calc_eigen_vals_and_vecs(
        self,
    ):
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

        rho = self.Ne.coords["rho"]
        N_z_t0 = np.zeros(self.N_z_tinf.shape, dtype=np.complex128)
        N_z_t0[0, :] = np.array([1.0 + 0.0j for i in range(rho.size)])

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

    def calc_N_z_t(
        self,
        tau,
    ):
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

    def interpolate_power(
        self,
    ):
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

        return PLT_spec, PRC_spec, PRB_spec

    def calc_cooling_factor(
        self,
    ):
        Ne, Nh = self.Ne, self.Nh

        N_z_t = self.N_z_t
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
        self.interpolate_rates()

        self.calc_ionisation_balance_matrix()

        self.calc_N_z_tinf()

        self.calc_eigen_vals_and_vecs()

        self.calc_eigen_coeffs()

        self.interpolate_power()

    def __call__(self, tau):
        self.calc_N_z_t(tau)

        self.calc_cooling_factor()

        return self.N_z_t, self.cooling_factor
