import numpy as np
import xarray as xr

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.operators.additional_high_z import AdditionalHighZ
from indica.utilities import coord_array
from ..fake_equilibrium import FakeEquilibrium


def make_dataarray(data, rho, t):
    return xr.DataArray(
        data=np.array(data, ndmin=2),
        dims=["t", "rho_poloidal"],
        coords=dict(
            rho_poloidal=rho,
            t=t,
        ),
    )


def test_calc_shape():
    """
    Regression test for calc_shape
    """
    rho = coord_array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rho_poloidal")
    t = coord_array([45], "t")

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    equilib = FakeEquilibrium(default_t=t)

    flux_surfs.set_equilibrium(equilib)

    n_high_z_midplane = make_dataarray(
        [1e-14, 9.5e-15, 8e-15, 8e-15, 7e-15, 3e-15, 0], rho, t
    )

    n_high_z_asymmetry_parameter = make_dataarray(
        [0.0, 0.1, 0.15, 0.2, 0.3, 0.8, 0.0], rho, t
    )
    q_high_z = make_dataarray([25, 23, 22, 15, 7, 3, 0], rho, t)
    q_additional_high_z = make_dataarray([20, 19, 17, 14, 8, 3, 0], rho, t)

    n_additional_high_z_shape = AdditionalHighZ._calc_shape(
        n_high_z_midplane,
        n_high_z_asymmetry_parameter,
        q_high_z,
        q_additional_high_z,
        flux_surfs,
    )

    n_additional_high_z_shape_expected = make_dataarray(
        np.array(
            [[1.0, 0.91161082, 0.77385276, 0.61048717, 0.38180397, 0.09532643, 0.0]]
        ),
        rho,
        t,
    )

    np.testing.assert_allclose(
        n_additional_high_z_shape, n_additional_high_z_shape_expected
    )

    # plt.figure()
    # plt.plot(rho, np.squeeze(n_high_z_midplane), "x", label="n_high_z")
    # plt.plot(
    #     rho,
    #     1e-14 * np.squeeze(n_additional_high_z_shape),
    #     "o",
    #     label="n_additional_high_z",
    # )
    # plt.legend()
    # plt.grid()
    # plt.show()


# TODO: analytic calc_shape test using known function
