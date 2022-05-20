from unittest.mock import MagicMock

import numpy as np
from tests.unit.operators.KB5_Bolometry_data import example_bolometry_LoS
from tests.unit.test_equilibrium_single import equilibrium_dat_and_te
from xarray import DataArray

from indica.converters.flux_surfaces import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.operators.bolometry_derivation import BolometryDerivation
from indica.readers import ADASReader


def input_data_setup():
    """Initial set-up for the majority of the data needed for ExtrapolateImpurityDensity.

    Returns
    -------
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho_poloidal, t)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho_poloidal, t)
    flux_surfs
        FluxSurfaceCoordinates object representing polar coordinate systems
        using flux surfaces for the radial coordinate.
    base_t
        xarray.DataArray of time values. Dimensions (t)
    elements
        List of element symbols for all impurities.
    rho_arr
        xarray.DataArray of rho_poloidal values, np.linspace(0, 1, 41).
        Dimensions (rho_poloidal)
    theta_arr
        xarray.DataArray of theta values, np.linspace(-np.pi, np.pi, 21).
        Dimensions (theta)
    """
    base_rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])
    base_t = np.linspace(75.0, 80.0, 20)

    input_Te = np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3])
    input_Te = np.tile(input_Te, (len(base_t), 1)).T

    input_Te = DataArray(
        data=np.tile(np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]), (len(base_t), 1)).T,
        coords={"rho_poloidal": base_rho_profile, "t": base_t},
        dims=["rho_poloidal", "t"],
    )

    input_Ne = np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19])
    input_Ne = np.tile(input_Ne, (len(base_t), 1)).T

    input_Ne = DataArray(
        data=np.tile(
            np.array([5.0e19, 4.0e19, 3.0e19, 2.0e19, 1.0e19]), (len(base_t), 1)
        ).T,
        coords={"rho_poloidal": base_rho_profile, "t": base_t},
        dims=["rho_poloidal", "t"],
    )

    elements = ["be", "ne", "ni"]

    expanded_rho = np.linspace(base_rho_profile[0], base_rho_profile[-1], 41)

    rho_arr = expanded_rho
    theta_arr = np.linspace(-np.pi, np.pi, 21)

    rho_arr = DataArray(
        data=rho_arr, coords={"rho_poloidal": rho_arr}, dims=["rho_poloidal"]
    )
    theta_arr = DataArray(data=theta_arr, coords={"theta": theta_arr}, dims=["theta"])

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    input_Te = input_Te.interp(rho_poloidal=expanded_rho, method="linear")
    input_Ne = input_Ne.interp(rho_poloidal=expanded_rho, method="linear")

    return (
        input_Ne,
        input_Te,
        flux_surfs,
        base_t,
        elements,
        rho_arr,
        theta_arr,
    )


def fractional_abundance_setup(
    element: str,
    t: np.ndarray,
    input_Te: DataArray,
    input_Ne: DataArray,
) -> DataArray:
    """Calculate and output Fractional abundance at t=infinity for calculating
    the mean charge in test_impurity_concentration()

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Time np.ndarray (used for expanding the dimensions of the output of
        the function to ensure that time is a dimension of the output.)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)

    Returns
    -------
    F_z_tinf
        Fractional abundance of the ionisation stages of the element at t=infinity.
        xarray.DataArray with dimensions (ion_charges, rho, t)
    """
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    example_frac_abundance = FractionalAbundance(
        SCD,
        ACD,
    )

    example_frac_abundance.interpolate_rates(Ne=input_Ne, Te=input_Te)
    example_frac_abundance.calc_ionisation_balance_matrix(Ne=input_Ne)

    F_z_tinf = example_frac_abundance.calc_F_z_tinf()

    # ignore with mypy since this is testing and inputs are known
    F_z_tinf = F_z_tinf.expand_dims({"t": t.size}, axis=-1)  # type: ignore

    return F_z_tinf


def power_loss_setup(
    element: str,
    t: np.ndarray,
    input_Te: DataArray,
    input_Ne: DataArray,
) -> DataArray:
    """Calculate and output PowerLoss for calculating
    the extrapolated impurity density.

    Parameters
    ----------
    element
        String of the symbol of the element per ADAS notation
        e.g be for Beryllium
    t
        Time np.ndarray (used for expanding the dimensions of the output of
        the function to ensure that time is a dimension of the output.)
    input_Te
        xarray.DataArray of electron temperature. Dimensions (rho, t)
    input_Ne
        xarray.DataArray of electron density. Dimensions (rho, t)

    Returns
    -------
    power_loss
        Power loss of the element at t=infinity.
        xarray.DataArray with dimensions (ion_charges, rho, t).
    """
    ADAS_file = ADASReader()

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRB = ADAS_file.get_adf11("prc", element, "89")

    power_loss_obj = PowerLoss(PLT, PRB)

    frac_abund_inf = fractional_abundance_setup(
        element, t, input_Te.isel(t=0), input_Ne.isel(t=0)
    )

    power_loss = power_loss_obj(input_Ne.isel(t=0), input_Te.isel(t=0), frac_abund_inf)

    return power_loss


def bolometry_input_data_setup(input_data):
    """Set-up for input data used for calculating bolometry data.

    Parameters
    ----------
    input_data
        Output of input_data_setup().

    Returns
    -------
    example_frac_abunds
        List of fractional abundances (an xarray.DataArray for each impurity)
        dimensions of each element in list are (ion_charges, rho, t).
    main_ion_power_loss
        Power loss associated with the main ion (eg. deuterium),
        xarray.DataArray with dimensions (rho, t)
    impurity_power_losses
        Power loss associated with all of the impurity elements,
        xarray.DataArray with dimensions (elements, rho, t)
    """
    initial_data = input_data

    base_t = initial_data[3]
    input_Te = initial_data[1]
    input_Ne = initial_data[0]
    elements = initial_data[4]

    main_ion_power_loss = power_loss_setup("h", base_t, input_Te, input_Ne)

    main_ion_power_loss = main_ion_power_loss.assign_coords(t=("t", base_t))

    main_ion_power_loss = main_ion_power_loss.sum(dim="ion_charges")

    example_frac_abunds = []
    impurity_power_losses = []

    for ielement in elements:
        example_frac_abund = fractional_abundance_setup(
            ielement, base_t, input_Te.isel(t=0), input_Ne.isel(t=0)
        )
        example_frac_abunds.append(example_frac_abund)

        impurity_power_loss = power_loss_setup(ielement, base_t, input_Te, input_Ne)
        impurity_power_loss = impurity_power_loss.assign_coords(t=("t", base_t))
        impurity_power_loss = impurity_power_loss.sum(dim="ion_charges")
        impurity_power_losses.append(impurity_power_loss.data)

    impurity_power_losses = np.array(impurity_power_losses)

    impurity_power_losses = DataArray(
        data=impurity_power_losses,
        coords=dict(**{"element": elements}, **impurity_power_loss.coords),
        dims=["element", *impurity_power_loss.dims],
    )

    return (example_frac_abunds, main_ion_power_loss, impurity_power_losses)


def test_bolometry_derivation():
    """Regression test for bolometry derivation.
    Original bolometry data is hardcoded within this function.
    """
    initial_data = input_data_setup()
    bolometry_input_data = bolometry_input_data_setup(initial_data)

    flux_surfs = initial_data[2]
    LoS_bolometry_data = example_bolometry_LoS
    t_arr = initial_data[3]
    frac_abunds = bolometry_input_data[0]
    impurity_elements = initial_data[4]
    electron_density = initial_data[0]
    main_ion_power_loss = bolometry_input_data[1]
    impurities_power_loss = bolometry_input_data[2]
    rho_arr = initial_data[5]
    theta_arr = initial_data[6]

    beryllium_density = np.tile(
        0.03 * electron_density.data, (theta_arr.shape[0], 1, 1)
    )
    neon_density = np.tile(0.02 * electron_density.data, (theta_arr.shape[0], 1, 1))
    nickel_density = np.tile(0.0002 * electron_density.data, (theta_arr.shape[0], 1, 1))

    beryllium_density = np.transpose(beryllium_density, (1, 0, 2))
    neon_density = np.transpose(neon_density, (1, 0, 2))
    nickel_density = np.transpose(nickel_density, (1, 0, 2))

    impurity_densities = DataArray(
        data=np.ones(
            (len(impurity_elements), *rho_arr.shape, *theta_arr.shape, *t_arr.shape)
        ),
        coords=[
            ("element", impurity_elements),
            ("rho_poloidal", rho_arr),
            ("theta", theta_arr),
            ("t", t_arr),
        ],
        dims=["element", "rho_poloidal", "theta", "t"],
    )

    impurity_densities.data[0] = beryllium_density
    impurity_densities.data[1] = neon_density
    impurity_densities.data[2] = nickel_density

    deriv_only = False
    trim = True
    t_val = t_arr[0]

    t_arr = DataArray(data=t_arr, coords={"t": t_arr}, dims=["t"])

    original_bolometry_data = DataArray(
        data=np.array([0.00000000e00, 1.84150833e09, 4.65631669e09, 2.11855843e09]),
        coords={"channels": np.array([0.0, 1.0, 2.0, 3.0])},
        dims=["channels"],
    )

    example_bolometry_derivation = BolometryDerivation(
        flux_surfs,
        LoS_bolometry_data,
        t_arr,
        impurity_densities,
        frac_abunds,
        impurity_elements,
        electron_density,
        main_ion_power_loss,
        impurities_power_loss,
    )

    new_bolometry_data = example_bolometry_derivation(deriv_only, trim, t_val)

    regression_checks = np.isclose(
        original_bolometry_data.data, new_bolometry_data.data
    )
    regression_check = np.all(regression_checks)
    assert regression_check
