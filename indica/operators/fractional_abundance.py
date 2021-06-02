from typing import List
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

from .abstractoperator import EllipsisType
from .abstractoperator import Operator
from .. import session
from ..datatypes import DataType
from ..readers import ADASReader

np.set_printoptions(edgeitems=10, linewidth=100)


class FractionalAbundance(Operator):
    """Calculate fractional abundance for a given ionisation stage of an ion

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
        ("ionisation_rate", "ion species"),
        ("recombination rate", "ion species"),
        ("charge-exchange rate", "ion species"),
        ("line power coeffecient", "ion species"),
        ("recombination power coeffecient", "ion species"),
        ("charge-exchange power coeffecient", "ion species"),
        ("temperature", "electron"),
        ("number density", "electron"),
        ("charge", "ion species"),
    ]
    RESULT_TYPES: List[Union[DataType, EllipsisType]] = [
        ("fractional abundance", "ion species"),
        ("total radiated power loss", "ion species"),
    ]

    def __init__(self, sess: session.Session = session.global_session):
        super().__init__(sess)

    def return_types(self, *args: DataType) -> Tuple[DataType, ...]:
        return (
            ("fractional abundance", "ion species"),
            ("total radiated power loss", "ion species"),
        )

    def __call__(self):
        ADAS_file = ADASReader("/mnt/c/Users/Sanket_Work/Documents/InDiCA_snippets/")

        element = "ne"

        Ne = 5.0e19
        Te = 1e3
        SCD = ADAS_file.get_adf11("scd", element, "89")

        SCD_spec = SCD
        SCD_spec = SCD_spec.interp(
            log10_electron_temperature=np.log10(Te), method="cubic", assume_sorted=True
        )
        SCD_spec = SCD_spec.interp(
            log10_electron_density=np.log10(Ne), method="cubic", assume_sorted=True
        )
        SCD_spec = np.power(10, SCD_spec)

        ACD = ADAS_file.get_adf11("acd", element, "89")

        ACD_spec = ACD
        ACD_spec = ACD_spec.interp(
            log10_electron_temperature=np.log10(Te), method="cubic", assume_sorted=True
        )
        ACD_spec = ACD_spec.interp(
            log10_electron_density=np.log10(Ne), method="cubic", assume_sorted=True
        )
        ACD_spec = np.power(10, ACD_spec)

        CCD = ADAS_file.get_adf11("ccd", element, "89")

        CCD_spec = CCD
        CCD_spec = CCD_spec.interp(
            log10_electron_temperature=np.log10(Te), method="cubic", assume_sorted=True
        )
        CCD_spec = CCD_spec.interp(
            log10_electron_density=np.log10(Ne), method="cubic", assume_sorted=True
        )
        CCD_spec = np.power(10, CCD_spec)

        ACD = ACD_spec
        CCD = CCD_spec
        SCD = SCD_spec

        NH = 1e-5 * Ne

        assert SCD.shape == ACD.shape == CCD.shape

        nz_shape = SCD.shape
        num_of_stages = nz_shape[0] + 1

        RHS_matr = np.zeros((num_of_stages, num_of_stages))

        istage = 0
        RHS_matr[istage, istage : istage + 2] = np.array(
            [-Ne * SCD[istage], Ne * ACD[istage] + NH * CCD[istage]]
        )
        for istage in range(1, num_of_stages - 1):
            RHS_matr[istage, istage - 1 : istage + 2] = np.array(
                [
                    Ne * SCD[istage - 1],
                    -Ne * (SCD[istage] + ACD[istage - 1]) - NH * CCD[istage - 1],
                    Ne * ACD[istage] + NH * CCD[istage],
                ]
            )
        istage = num_of_stages - 1
        RHS_matr[istage, istage - 1 : istage + 1] = np.array(
            [Ne * SCD[istage - 1], -Ne * (ACD[istage - 1]) - NH * CCD[istage - 1]]
        )

        # print(np.min(np.abs(RHS_matr[np.nonzero(RHS_matr)])))
        # print(".................................................................")

        RHS_null_space = scipy.linalg.null_space(RHS_matr)
        N_z_tinf = np.abs(RHS_null_space).astype(dtype=np.complex128)

        # test_null = np.dot(RHS_matr, N_z_tinf)
        # print(np.allclose(test_null, np.zeros(nz_shape)))
        # print(".................................................................")

        num_eig_vals = min(75, num_of_stages)
        vals, vecs = scipy.sparse.linalg.eigs(RHS_matr, k=num_eig_vals, which="LM")
        # vals, vecs = np.abs(vals), np.abs(vecs)

        print(vals)
        print(".................................................................")

        print(vecs)
        print(".................................................................")

        for i in range(num_eig_vals):
            test_ARPack = np.dot(RHS_matr, vecs[:, i]) - np.dot(vals[i], vecs[:, i])
            print(np.allclose(test_ARPack, np.zeros(num_of_stages), atol=1e-7))

        # vals, vecs = np.abs(vals), np.abs(vecs)

        Nz_t0 = np.zeros(N_z_tinf.shape, dtype=np.complex128)
        Nz_t0[0, 0] = 1.0 + 0.0j

        eig_vecs = np.transpose(vecs)
        eig_vecs_inv = np.linalg.pinv(eig_vecs)

        boundary_conds = np.transpose(Nz_t0 - N_z_tinf)

        eig_coeffs = np.dot(boundary_conds, eig_vecs_inv)
        eig_coeffs = eig_coeffs[0]

        eig_vals = vals

        def N_z_t(N_z_tinf, eig_coeffs, eig_vals, eig_vecs, tau, num_eig_vals):
            result = np.tile(np.transpose(N_z_tinf)[0], (len(tau), 1)).T
            for it, itau in enumerate(tau):
                for i in range(num_eig_vals):
                    result[:, it] += (
                        eig_coeffs[i] * np.exp(eig_vals[i] * itau) * eig_vecs[i, :]
                    )

            return result

        tau = np.linspace(-16, 2, 1000)
        tau = np.power(10.0, tau)
        tau = np.insert(tau, 0, np.power(10.0, -np.inf))

        print(tau)
        print(".................................................................")

        data = N_z_t(N_z_tinf, eig_coeffs, eig_vals, eig_vecs, tau, num_eig_vals)
        data = np.around(data, decimals=16)

        print(np.linalg.norm(data[:, 0]))

        for istage in range(num_of_stages):
            plt.loglog(tau, data[istage], label=f"{istage}")
        plt.legend()
        # plt.ylim(1e-19, 2e0)
        plt.show()

        return


if __name__ == "__main__":
    FracAbund = FractionalAbundance()
    FracAbund()
