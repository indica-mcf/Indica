import numpy as np
from xarray.core.dataarray import DataArray

from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.toroidal_rotation_and_asymmetry import ToroidalRotation
from indica.readers.adas import ADASReader


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    t = np.linspace(77.5, 82.5, 5)
    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    input_Te = DataArray(
        data=np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]),
        coords={"rho": rho_profile},
        dims=["rho"],
    )

    input_Ne = DataArray(
        data=np.array([5.0e19, 4e19, 3.0e19, 2.0e19, 1.0e19]),
        coords={"rho": rho_profile},
        dims=["rho"],
    )

    example_frac_abundance = FractionalAbundance(
        SCD,
        ACD,
        input_Ne,
        input_Te,
    )

    F_z_tinf = example_frac_abundance.F_z_tinf

    # ignore with mypy since this is testing and inputs are known
    F_z_tinf = F_z_tinf.expand_dims({"t": t.size}, axis=-1)  # type: ignore

    return F_z_tinf


def test_toroidal_rotation():
    example_toroidal_rotation = ToroidalRotation()

    t = np.linspace(77.5, 82.5, 5)
    rho_profile = np.array([0.0, 0.4, 0.8, 0.95, 1.0])

    electron_temp = DataArray(
        data=np.array([3.0e3, 1.5e3, 0.5e3, 0.2e3, 0.1e3]),
        coords={"rho": rho_profile},
        dims=["rho"],
    )

    # be, c, ne, w
    element_atomic_numbers = [4, 10, 28, 74]
    elements = [ELEMENTS_BY_ATOMIC_NUMBER.get(i) for i in element_atomic_numbers]

    asymmetry_parameters = DataArray(
        data=np.ones((len(elements), *rho_profile.shape, *t.shape)),
        coords=[("elements", elements), ("rho", rho_profile), ("t", t)],
        dims=["elements", "rho", "t"],
    )

    ion_temperature = DataArray(
        data=np.array([2.0e3, 1.2e3, 0.5e3, 0.2e3, 0.1e3]),
        coords={"rho": rho_profile},
        dims=["rho"],
    )

    unified_atomic_mass_unit = 1.660539066e-27
    main_ion_mass = 2.014 * unified_atomic_mass_unit

    impurity_masses = DataArray(
        data=np.array([7.014, 20.1797, 58.6934, 183.84]) * unified_atomic_mass_unit,
        coords={"elements": elements},
        dims=["elements"],
    )

    mean_charges = DataArray(
        data=np.tile(np.array(element_atomic_numbers), (t.size, rho_profile.size, 1)).T,
        coords=[("elements", elements), ("rho", rho_profile), ("t", t)],
        dims=["elements", "rho", "t"],
    )

    Zeff_diag = DataArray(
        data=2.0 * np.ones((*rho_profile.shape, *t.shape)),
        coords=[("rho", rho_profile), ("t", t)],
        dims=["rho", "t"],
    )

    impurity_element = "beryllium"

    nominal_inputs = {
        "asymmetry_parameters": asymmetry_parameters,
        "ion_temperature": ion_temperature,
        "main_ion_mass": main_ion_mass,
        "impurity_masses": impurity_masses,
        "mean_charges": mean_charges,
        "Zeff_diag": Zeff_diag,
        "electron_temp": electron_temp,
        "impurity_element": impurity_element,
    }

    toroidal_rotation = example_toroidal_rotation(**nominal_inputs)

    assert toroidal_rotation is not None
