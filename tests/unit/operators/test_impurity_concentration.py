from unittest.mock import MagicMock

import numpy as np
from xarray import DataArray
from xarray.core.common import zeros_like

from indica.converters import FluxSurfaceCoordinates
from indica.converters.lines_of_sight import LinesOfSightTransform
from indica.datatypes import ELEMENTS_BY_ATOMIC_NUMBER
from indica.datatypes import ELEMENTS_BY_SYMBOL
from indica.equilibrium import Equilibrium
from indica.numpy_typing import LabeledArray
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.impurity_concentration import ImpurityConcentration
from indica.operators.mean_charge import MeanCharge
from indica.readers.adas import ADASReader
from ..test_equilibrium_single import equilibrium_dat_and_te


def fractional_abundance_setup(element: str, t: LabeledArray) -> DataArray:
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")

    input_Ne = np.logspace(19.0, 16.0, 10)
    input_Te = np.logspace(4.6, 2.0, 10)

    input_Te = DataArray(
        data=input_Te, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
    )
    input_Ne = DataArray(
        data=input_Ne, coords={"rho": np.linspace(0.0, 1.0, 10)}, dims=["rho"]
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


def test_impurity_concentration():
    example_ = ImpurityConcentration()

    R_arr = np.linspace(1.83, 3.9, 20)
    rho = np.linspace(0.0, 1.5, 10)
    t = np.linspace(77.5, 82.5, 5)
    # t = np.array([80.0])

    Zeff_diag = DataArray(
        data=np.ones(*t.shape) + 1.5,
        coords={"t": t},
        dims=["t"],
        attrs={
            "transform": LinesOfSightTransform(
                R_start=np.array([1.9]),
                z_start=np.array([0.2]),
                T_start=np.array([0.0]),
                R_end=np.array([3.8]),
                z_end=np.array([0.3]),
                T_end=np.array([0.0]),
                name="Zeff_diag",
            ),
            "Zeff_diag_coords": DataArray(
                data=np.array([0]), dims=["Zeff_diag_coords"]
            ),
        },
    )

    # be, c, ne, w
    elements = [4, 6, 10, 74]
    elements = [ELEMENTS_BY_ATOMIC_NUMBER.get(i) for i in elements]

    impurity_densities = DataArray(
        data=np.ones((len(elements), *R_arr.shape, *rho.shape, *t.shape)),
        coords=[("elements", elements), ("R", R_arr), ("rho", rho), ("t", t)],
        dims=["elements", "R", "rho", "t"],
    )

    electron_density = DataArray(
        data=np.ones((*rho.shape, *t.shape)) * 5e19,
        coords={"rho": rho, "t": t},
        dims=["rho", "t"],
    )

    mean_charge = zeros_like(impurity_densities)
    mean_charge = mean_charge.isel(R=0)
    mean_charge = mean_charge.drop_vars("R")

    F_z_tinf = fractional_abundance_setup("be", t)
    element_name = ELEMENTS_BY_SYMBOL.get("be")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[0] = result

    F_z_tinf = fractional_abundance_setup("c", t)
    element_name = ELEMENTS_BY_SYMBOL.get("c")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[1] = result

    F_z_tinf = fractional_abundance_setup("ne", t)
    element_name = ELEMENTS_BY_SYMBOL.get("ne")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[2] = result

    F_z_tinf = fractional_abundance_setup("w", t)
    element_name = ELEMENTS_BY_SYMBOL.get("w")

    mean_charge_obj = MeanCharge()
    result = mean_charge_obj(F_z_tinf, element_name)
    mean_charge.data[3] = result

    flux_surfs = FluxSurfaceCoordinates("poloidal")

    offset = MagicMock(return_value=0.02)
    equilib_dat, Te = equilibrium_dat_and_te()
    equilib = Equilibrium(equilib_dat, Te, sess=MagicMock(), offset_picker=offset)

    flux_surfs.set_equilibrium(equilib)

    concentration, t = example_(
        element="beryllium",
        Zeff_diag=Zeff_diag,
        impurity_densities=impurity_densities,
        electron_density=electron_density,
        mean_charge=mean_charge,
        flux_surfaces=flux_surfs,
        t=80.0,
    )
