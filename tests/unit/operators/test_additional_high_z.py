import numpy as np
from indica.operators.centrifugal_asymmetry import AsymmetryParameter
from tests.regression.operators.test_bolometry_derivation import bolometry_object_setup
from tests.regression.operators.test_bolometry_derivation import input_data_setup
import xarray as xr

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.operators.additional_high_z import AdditionalHighZ, calc_fsa_quantity
from indica.operators.extrapolate_impurity_density import (
    asymmetry_modifier_from_parameter,
)
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
        [1e16, 9.5e15, 8e15, 8e15, 7e15, 3e15, 0], rho, t
    )

    n_high_z_asymmetry_parameter = make_dataarray(
        [0.0, 0.1, 0.15, 0.2, 0.3, 0.8, 0.0], rho, t
    )

    n_high_z_fsa = calc_fsa_quantity(
        symmetric_component=n_high_z_midplane,
        asymmetry_parameter=n_high_z_asymmetry_parameter,
        flux_surfaces=flux_surfs,
    )

    q_high_z = make_dataarray([25, 23, 22, 15, 7, 3, 0], rho, t)
    q_additional_high_z = make_dataarray([20, 19, 17, 14, 8, 3, 0], rho, t)

    n_additional_high_z_shape = AdditionalHighZ._calc_shape(
        n_high_z_fsa,
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
        [1e14, 9.5e15, 8e15, 8e15, 7e15, 3e15, 0], rho, t
    )

    n_high_z_asymmetry_parameter = make_dataarray(
        [0.0, 0.1, 0.15, 0.2, 0.3, 0.8, 0.0], rho, t
    )

    n_high_z_fsa = calc_fsa_quantity(
        symmetric_component=n_high_z_midplane,
        asymmetry_parameter=n_high_z_asymmetry_parameter,
        flux_surfaces=flux_surfs,
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
        n_high_z_fsa,
        flux_surfs,
    )

    n_additional_high_z_seminormalised_fsa_expected = make_dataarray(
        np.array(
            [
                [
                    1.012692e16,
                    9.231809e15,
                    7.836744e15,
                    6.182354e15,
                    3.866498e15,
                    9.653631e14,
                    0.000000e00,
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


def test_calc_seminormalised_additional_high_z_density():
    rho = coord_array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rho_poloidal")
    t = coord_array([45], "t")

    n_additional_high_z_seminormalised_fsa = make_dataarray(
        np.array(
            [
                [
                    1.012692e16,
                    9.231809e15,
                    7.836744e15,
                    6.182354e15,
                    3.866498e15,
                    9.653631e14,
                    0.000000e00,
                ]
            ]
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

    zeff = 1.85 * xr.ones_like(n_additional_high_z_seminormalised_fsa)

    toroidal_rotations = xr.DataArray(
        data=np.array([[[200.0e3, 180.0e3, 170.0e3, 150.0e3, 100.0e3, 30.0e3, 5.0e3]]]),
        dims=["element", "t", "rho_poloidal"],
        coords=dict(
            element=[additional_high_z_ion],
            rho_poloidal=rho,
            t=t,
        ),
    )

    n_additional_high_z_asymmetry_parameter = AsymmetryParameter()(
        toroidal_rotations=toroidal_rotations,
        ion_temperature=ion_temperature,
        main_ion=main_ion,
        impurity=additional_high_z_ion,
        Zeff=zeff,
        electron_temp=electron_temp,
    )

    n_additional_high_z_seminormalised_midplane = (
        AdditionalHighZ._calc_seminormalised_additional_high_z_density(
            n_additional_high_z_seminormalised_fsa,
            n_additional_high_z_asymmetry_parameter,
            flux_surfs,
        )
    )

    n_additional_high_z_seminormalised_midplane_expected = xr.DataArray(
        data=np.array(
            [
                [
                    1.012692e16,
                    1.907021e16,
                    2.042044e16,
                    1.919408e16,
                    1.396252e16,
                    3.091976e15,
                    0.000000e00,
                ]
            ]
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=t,
            rho_poloidal=rho,
        ),
    )

    n_additional_high_z_asymmetry_parameter_expected = xr.DataArray(
        data=np.array(
            [
                [
                    1.84585255,
                    2.08597558,
                    2.16319933,
                    1.93145887,
                    2.34416936,
                    0.52743811,
                    0.02930212,
                ]
            ]
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=t,
            rho_poloidal=rho,
        ),
    )

    np.testing.assert_allclose(
        n_additional_high_z_seminormalised_midplane,
        n_additional_high_z_seminormalised_midplane_expected,
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        n_additional_high_z_asymmetry_parameter,
        n_additional_high_z_asymmetry_parameter_expected,
        rtol=1e-6,
    )


# TODO: analytic calc_unnormalised_additional_high_z_density test using known function


def test_calc_normalised_additional_high_z_density():
    """
    Create a bolometry test object.
    Pick an arbitrary normalisation factor for test.
    Replace the nickel density with our arbitrarily normalised nickel density.
    Calculate the observed bolometry.
    Set the nickel density in the bolometry object to 0.
    Call calc_unnormalised_additional_high_z_density.
    Check the normalisation factor agrees (it will not agree exactly due to
    problems with the method).
    """
    normalisation = 0.7
    rho = coord_array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rho_poloidal")
    additional_high_z_element = "ni"

    input_data = input_data_setup()
    base_t = input_data[3]
    bolometry_rho = input_data[5]
    bolometry_theta = input_data[6]
    bolometry_t = coord_array(base_t, "t")

    flux_surfaces = FluxSurfaceCoordinates("poloidal")
    equilib = FakeEquilibrium(default_t=bolometry_t)
    flux_surfaces.set_equilibrium(equilib)

    n_additional_high_z_seminormalised_midplane = xr.DataArray(
        data=np.tile(
            [
                1.012692e16,
                1.907021e16,
                2.042044e16,
                1.919408e16,
                1.396252e16,
                3.091976e15,
                0.000000e00,
            ],
            (len(bolometry_t), 1),
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=bolometry_t,
            rho_poloidal=rho,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    n_additional_high_z_asymmetry_parameter = xr.DataArray(
        data=np.tile(
            [
                1.84585255,
                2.08597558,
                2.16319933,
                1.93145887,
                2.34416936,
                0.52743811,
                0.02930212,
            ],
            (len(bolometry_t), 1),
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=bolometry_t,
            rho_poloidal=rho,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    R_mid, z_mid = flux_surfaces.convert_to_Rz(bolometry_rho, bolometry_theta)
    asymmetry_modifier = asymmetry_modifier_from_parameter(
        n_additional_high_z_asymmetry_parameter, R_mid
    )
    # construct density on a rho, theta grid
    n_additional_high_z_seminormalised = (
        n_additional_high_z_seminormalised_midplane * asymmetry_modifier
    )

    bolometry_obj = bolometry_object_setup(input_data)

    # insert density into bolometry obj
    n_additional_high_z_seminormalised = n_additional_high_z_seminormalised.transpose(
        "rho_poloidal", "theta", "t"
    )

    normalised_density = normalisation * n_additional_high_z_seminormalised
    bolometry_obj.impurity_densities.loc["ni", :, :, :] = normalised_density

    bolometry_observation = bolometry_obj(False, False, None)

    # zero nickel density in bolometry_obj
    bolometry_obj.impurity_densities.loc["ni", :, :, :] = xr.zeros_like(
        normalised_density
    )

    n_additional_high_z = AdditionalHighZ._calc_normalised_additional_high_z_density(
        n_additional_high_z_seminormalised_midplane,
        n_additional_high_z_asymmetry_parameter,
        additional_high_z_element,
        flux_surfaces,
        bolometry_observation,
        bolometry_obj,
    )

    normalisation_result = (
        n_additional_high_z / n_additional_high_z_seminormalised_midplane
    ).mean()

    # high rtol because the bolometry normalisation is inexact for small Ni densities
    np.testing.assert_allclose(normalisation_result, normalisation, rtol=0.5)


def test_call():
    normalisation = 0.7
    rho = coord_array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "rho_poloidal")

    input_data = input_data_setup()
    base_t = input_data[3]
    bolometry_rho = input_data[5]
    bolometry_theta = input_data[6]
    t = coord_array(base_t, "t")

    flux_surfaces = FluxSurfaceCoordinates("poloidal")
    equilib = FakeEquilibrium(default_t=t)
    flux_surfaces.set_equilibrium(equilib)

    n_additional_high_z_seminormalised_midplane = xr.DataArray(
        data=np.tile(
            [
                1.012692e16,
                1.907021e16,
                2.042044e16,
                1.919408e16,
                1.396252e16,
                3.091976e15,
                0.000000e00,
            ],
            (len(t), 1),
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=t,
            rho_poloidal=rho,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    n_additional_high_z_asymmetry_parameter = xr.DataArray(
        data=np.tile(
            [
                1.84585255,
                2.08597558,
                2.16319933,
                1.93145887,
                2.34416936,
                0.52743811,
                0.02930212,
            ],
            (len(t), 1),
        ),
        dims=["t", "rho_poloidal"],
        coords=dict(
            t=t,
            rho_poloidal=rho,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    R_mid, z_mid = flux_surfaces.convert_to_Rz(bolometry_rho, bolometry_theta)
    asymmetry_modifier = asymmetry_modifier_from_parameter(
        n_additional_high_z_asymmetry_parameter, R_mid
    )
    # construct density on a rho, theta grid
    n_additional_high_z_seminormalised = (
        n_additional_high_z_seminormalised_midplane * asymmetry_modifier
    )

    bolometry_obj = bolometry_object_setup(input_data)

    # insert density into bolometry obj
    n_additional_high_z_seminormalised = n_additional_high_z_seminormalised.transpose(
        "rho_poloidal", "theta", "t"
    )

    normalised_density = normalisation * n_additional_high_z_seminormalised
    bolometry_obj.impurity_densities.loc["ni", :, :, :] = normalised_density

    bolometry_observation = bolometry_obj(False, False, None)

    # zero nickel density in bolometry_obj
    bolometry_obj.impurity_densities.loc["ni", :, :, :] = xr.zeros_like(
        normalised_density
    )

    n_high_z_midplane = make_dataarray(
        np.tile([1e16, 9.5e15, 8e15, 8e15, 7e15, 3e15, 0], (len(t), 1)), rho, t
    ).interp(rho_poloidal=bolometry_rho)
    n_high_z_asymmetry_parameter = make_dataarray(
        np.tile([0.0, 0.1, 0.15, 0.2, 0.3, 0.8, 0.0], (len(t), 1)), rho, t
    ).interp(rho_poloidal=bolometry_rho)
    n_high_z_fsa = calc_fsa_quantity(
        symmetric_component=n_high_z_midplane,
        asymmetry_parameter=n_high_z_asymmetry_parameter,
        flux_surfaces=flux_surfaces,
    )
    q_high_z = make_dataarray(
        np.tile([25, 23, 22, 15, 7, 3, 0], (len(t), 1)), rho, t
    ).interp(rho_poloidal=bolometry_rho)
    q_additional_high_z = make_dataarray(
        np.tile([20, 19, 17, 14, 8, 3, 0], (len(t), 1)), rho, t
    ).interp(rho_poloidal=bolometry_rho)

    electron_temp = make_dataarray(
        np.tile([3.0e3, 2.0e3, 1.7e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3], (len(t), 1)), rho, t
    ).interp(rho_poloidal=bolometry_rho)

    main_ion = "d"
    additional_high_z_ion = "ni"

    ion_temperature = xr.DataArray(
        data=np.array(
            np.tile(
                [[[2.0e3, 1.6e3, 1.4e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3]]], ((len(t), 1))
            )
        ),
        dims=["element", "t", "rho_poloidal"],
        coords=dict(
            element=[additional_high_z_ion],
            rho_poloidal=rho,
            t=t,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    zeff = 1.85 * xr.ones_like(n_additional_high_z_seminormalised_midplane).interp(
        rho_poloidal=bolometry_rho
    )

    toroidal_rotations = xr.DataArray(
        data=np.array(
            np.tile(
                [[[200.0e3, 180.0e3, 170.0e3, 150.0e3, 100.0e3, 30.0e3, 5.0e3]]],
                (len(t), 1),
            )
        ),
        dims=["element", "t", "rho_poloidal"],
        coords=dict(
            element=[additional_high_z_ion],
            rho_poloidal=rho,
            t=t,
        ),
    ).interp(rho_poloidal=bolometry_rho)

    n_additional_high_z_asymmetry_parameter = AsymmetryParameter()(
        toroidal_rotations=toroidal_rotations,
        ion_temperature=ion_temperature,
        main_ion=main_ion,
        impurity=additional_high_z_ion,
        Zeff=zeff,
        electron_temp=electron_temp,
    )

    additional_high_z_operator = AdditionalHighZ()

    n_additional_high_z = additional_high_z_operator(
        n_high_z_fsa,
        n_additional_high_z_asymmetry_parameter,
        q_high_z,
        q_additional_high_z,
        additional_high_z_ion,
        bolometry_observation,
        bolometry_obj,
        flux_surfaces,
    )

    n_additional_high_z_expected = make_dataarray(
        np.tile(
            [
                9.29464917e15,
                1.23868853e16,
                1.51831619e16,
                1.73922681e16,
                1.88824070e16,
                1.94565777e16,
                1.95730797e16,
                1.94033973e16,
                1.92305549e16,
                1.93438217e16,
                1.95533395e16,
                1.96408333e16,
                1.96797485e16,
                1.96605359e16,
                1.95854241e16,
                1.94898682e16,
                1.92511961e16,
                1.87412884e16,
                1.81189062e16,
                1.75220158e16,
                1.69213984e16,
                1.63444845e16,
                1.57636684e16,
                1.53283441e16,
                1.45203991e16,
                1.22820703e16,
                1.01400253e16,
                8.50201547e15,
                7.11479242e15,
                5.94020408e15,
                4.92793122e15,
                4.02091949e15,
                3.30072371e15,
                2.90330883e15,
                2.57648090e15,
                2.22912593e15,
                1.88610019e15,
                1.53530693e15,
                1.17639771e15,
                6.82497864e14,
                0.00000000e00,
            ],
            (len(t), 1),
        ),
        bolometry_rho,
        t,
    )

    np.testing.assert_allclose(
        n_additional_high_z.isel(t=0), n_additional_high_z_expected.isel(t=0)
    )
