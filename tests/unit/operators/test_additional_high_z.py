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


def test_calc_first_normalisation():
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

    n_additional_high_z_unnormalised_fsa = make_dataarray(
        np.array(
            [[1.0, 0.91161082, 0.77385276, 0.61048717, 0.38180397, 0.09532643, 0.0]]
        ),
        rho,
        t,
    )

    n_additional_high_z_seminormalised_fsa = AdditionalHighZ._calc_first_normalisation(
        n_additional_high_z_unnormalised_fsa,
        n_high_z_midplane,
        n_high_z_asymmetry_parameter,
        flux_surfs,
    )

    n_additional_high_z_seminormalised_fsa_expected = make_dataarray(
        np.array(
            [
                [
                    1.02114531e-14,
                    9.30887110e-15,
                    7.90216113e-15,
                    6.23396108e-15,
                    3.89877332e-15,
                    9.73421365e-16,
                    0.0,
                ]
            ]
        ),
        rho,
        t,
    )

    np.testing.assert_allclose(
        n_additional_high_z_seminormalised_fsa,
        n_additional_high_z_seminormalised_fsa_expected,
    )


def test_calc_unnormalised_additional_high_z_density():
    rho = coord_array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rho_poloidal")
    t = coord_array([45], "t")

    n_additional_high_z_unnormalised_fsa = make_dataarray(
        np.array(
            [[1.0, 0.91161082, 0.77385276, 0.61048717, 0.38180397, 0.09532643, 0.0]]
        ),
        rho,
        t,
    )

    flux_surfs = FluxSurfaceCoordinates("poloidal")
    equilib = FakeEquilibrium(default_t=t)
    flux_surfs.set_equilibrium(equilib)

    main_ion = "d"
    additional_high_z_ion = "ni"

    electron_temp = make_dataarray(
        np.array([3.0e3, 2.0e3, 1.7e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), rho, t
    )

    ion_temperature = xr.DataArray(
        data=np.array([[[2.0e3, 1.6e3, 1.4e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3]]]),
        dims=["element", "t", "rho_poloidal"],
        coords=dict(
            element=[additional_high_z_ion],
            rho_poloidal=rho,
            t=t,
        ),
    )

    zeff = 1.85 * xr.ones_like(n_additional_high_z_unnormalised_fsa)

    toroidal_rotations = xr.DataArray(
        data=np.array([[[200.0e3, 180.0e3, 170.0e3, 150.0e3, 100.0e3, 30.0e3, 5.0e3]]]),
        dims=["element", "t", "rho_poloidal"],
        coords=dict(
            element=[additional_high_z_ion],
            rho_poloidal=rho,
            t=t,
        ),
    )

    n_additional_high_z_unnormalised_midplane = (
        AdditionalHighZ._calc_unnormalised_additional_high_z_density(
            n_additional_high_z_unnormalised_fsa,
            toroidal_rotations,
            ion_temperature,
            main_ion,
            additional_high_z_ion,
            zeff,
            electron_temp,
            flux_surfs,
        )
    )

    n_additional_high_z_unnormalised_midplane_expected = xr.DataArray(
        data=np.array(
            [[1.0, 1.88312029, 2.01645164, 1.89535224, 1.37875282, 0.30532248, 0.0]]
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=t,
            rho_poloidal=rho,
        ),
    )

    np.testing.assert_allclose(
        n_additional_high_z_unnormalised_midplane,
        n_additional_high_z_unnormalised_midplane_expected,
    )


# TODO: analytic calc_unnormalised_additional_high_z_density test using known function
