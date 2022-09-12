"""
Base script for running InDiCA analysis tests.
Run with specific data source (e.g. JET JPF/PPF data)
"""
import json
from pathlib import Path
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import xarray as xr
from xarray import concat
from xarray import DataArray

from indica.utilities import coord_array
from .observers import Observable
from .observers import Observer
from .workflow_utilities import print_step_template


class BaseWorkflow:
    """
    Common components of running a benchmark InDiCA analysis for continuous
    testing.
    Subclass and specialise for specific use cases
    """

    cache_dir = Path(".").absolute().parent / "test_cache"
    cache_file = cache_dir / "cache.json"

    def __init__(
        self,
        config: Dict[str, Any] = None,
        config_file: Union[str, Path] = "input.json",
    ):
        """
        Get test parameters from json configuration file, sets defaults for values
        not present in file.

        :param config_key: Key to get parameters for in configuration file
        :type config_key: Hashable
        """
        # Config parameters
        if config is None:
            self.input = self._read_test_case(config_file=config_file)
        else:
            self.input = config
        self.comparison_data = self.input.get("comparison_data", {})

        self.rho: DataArray = coord_array(
            np.linspace(*self.input.get("rho", (0.0, 1.0, 25))),
            "rho_poloidal",
        )
        self.theta: DataArray = coord_array(
            np.linspace(*self.input.get("theta", (-np.pi, np.pi, 25))),
            "theta",
        )
        self.R: DataArray = coord_array(
            np.linspace(*self.input.get("R", (1.83, 3.9, 25))),
            "R",
        )
        self.z: DataArray = coord_array(
            np.linspace(*self.input.get("z", (-1.75, 2.0, 25))),
            "z",
        )
        time = self.input.get("t", (45, 50, 5))
        self.t: DataArray = coord_array(np.linspace(*time), "t")
        self.trange: Tuple[float, float] = (
            self.t.values[0] - (self.t.values[1] - self.t.values[0]),
            self.t.values[-1] + (self.t.values[1] - self.t.values[0]),
        )  # TEMP for inclusive time slice ranges

        self.high_z: str = self.input.get("high_z", "w")
        self.zeff_el: str = self.input.get("zeff_el", "be")
        self.zeff_el_extra: str = self.input.get("zeff_el_extra", "he")
        self.other_z: str = self.input.get("other_z", "ni")
        self.main_ion: str = self.input.get("main_ion", "d")
        self.ion_species: List[str] = [
            self.high_z,
            self.zeff_el,
            self.zeff_el_extra,
            self.other_z,
            self.main_ion,
        ]

        self.cxrs_instrument = self.input.get("cxrs_instrument", "cxg6").lower()

        # Fixed quantities
        self._power_loss = Observable()
        self._sxr_power_loss = Observable()
        self._sxr_emissivity = Observable()
        self._sxr_fitted_symmetric_emissivity = Observable()
        self._sxr_fitted_asymmetry_parameter = Observable()
        self._electron_density = Observable()
        self._electron_temperature = Observable()
        self._ion_temperature = Observable()
        self._toroidal_rotation = Observable()
        self._extra_zeff_element_concentration = Observable(
            initial_value=self.input.get("conc_zeff_el_2", 0.0)
        )
        self._sxr_calibration_factor = Observable(
            initial_value=self.input.get("initial_values", {}).get(
                "sxr_calibration_factor", None
            )
        )
        self._sxr_rescale_factor = Observable(
            initial_value=self.input.get("initial_values", {}).get(
                "sxr_rescale_factor", None
            )
        )
        self._n_high_z = Observable()

        # Convenience quantities
        self._power_loss_charge_averaged = Observer(
            operator=self._calculate_power_loss_charge_averaged,
            depends_on=[self._power_loss],
        )
        self._sxr_power_loss_charge_averaged = Observer(
            operator=self._calculate_sxr_power_loss_charge_averaged,
            depends_on=[self._sxr_power_loss],
        )

        # Observable quantities
        # self._n_high_z = Observer(
        #     operator=self._calculate_n_high_z,
        #     depends_on=[
        #         self._sxr_emissivity,
        #         self._sxr_power_loss,
        #         self._electron_density,
        #         self._sxr_calibration_factor,
        #         self._sxr_rescale_factor,
        #     ],
        # )
        self._asymmetry_parameter_high_z = Observer(
            operator=self._calculate_asymmetry_high_z,
            depends_on=[self._ion_temperature, self._toroidal_rotation],
        )
        self._asymmetry_parameter_other_z = Observer(
            operator=self._calculate_asymmetry_other_z,
            depends_on=[self._ion_temperature, self._toroidal_rotation],
        )

        # Depends on n_high_z
        self._n_zeff_el = Observer(
            operator=self._calculate_n_zeff_el,
            depends_on=[self._n_high_z, self._electron_density],
        )
        self._n_zeff_el_extra = Observer(
            operator=self._calculate_n_zeff_el_extra,
            depends_on=[self._n_high_z, self._electron_density],
        )
        self._n_other_z = Observer(
            operator=self._calculate_n_other_z,
            depends_on=[self._n_high_z, self._asymmetry_parameter_other_z],
        )
        self._derived_asymmetry_high_z = Observer(
            operator=self._calculate_derived_asymmetry_high_z,
            depends_on=[self._n_high_z],
        )

        # Depends on n_other_z
        self._derived_asymmetry_other_z = Observer(
            operator=self._calculate_derived_asymmetry_other_z,
            depends_on=[self._n_other_z],
        )

        # Depends on all available impurity densities
        self._n_main_ion = Observer(
            operator=self._calculate_n_main_ion,
            depends_on=[
                self._n_high_z,
                self._n_zeff_el,
                self._n_zeff_el_extra,
                self._n_other_z,
            ],
        )
        self._ion_densities = Observer(
            operator=self._combine_ion_density_arrays,
            depends_on=[
                self._n_high_z,
                self._n_zeff_el,
                self._n_zeff_el_extra,
                self._n_other_z,
                self._n_main_ion,
            ],
        )
        self._derived_bolometry = Observer(
            operator=self._calculate_derived_bolometry,
            depends_on=[self._ion_densities],
        )

        self.setup_check: Dict[str, bool] = {}

    def __call__(self, *args, **kwargs):
        for step_method, key_attr in self.setup_steps:
            if getattr(self, key_attr, None) is not None:
                continue
            print(
                print_step_template(fallback_size=(80, 24)).format(step_method.__name__)
            )
            step_method()

    def __str__(self) -> str:
        outp = ""
        formatter = "Step = {:<35s} | status = {:<8s}\n"

        for step_name, status in self._progress():
            outp += formatter.format(
                step_name, "Complete" if status is True else "Waiting"
            )
        return outp.rstrip("\n")

    @property
    def __external_properties__(self) -> List[str]:
        """
        List of properties that are externally obtained
        """
        return ["input"]

    def save(
        self,
        filename: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        if not overwrite:
            with open(filename, "rb") as f:
                current = pickle.load(f)
        else:
            current = {}
        current.update(
            {key: getattr(self, key, None) for key in self.__external_properties__}
        )
        with open(filename, "wb+") as f:
            pickle.dump(current, f)

    @classmethod
    def restore(cls, filename: Union[str, Path]) -> "BaseWorkflow":
        with open(filename, "rb") as f:
            properties: Dict[str, Any] = pickle.load(f)
        workflow = cls(config=properties.pop("input", None))
        for key, value in properties.items():
            setattr(workflow, key, value)
        return workflow

    @property
    def setup_steps(self) -> List[Tuple[Callable, str]]:
        raise NotImplementedError(
            f"{self.__class__} does not implemented setup_steps property"
        )

    def _read_test_case(
        self, config_file: Union[str, Path] = "input.json"
    ) -> Dict[str, Any]:
        with open(config_file, "r") as f:
            return json.load(f)

    def _progress(self) -> List[Tuple[str, bool]]:
        raise NotImplementedError(
            f"{self.__class__} does not implemented _progress method"
        )

    def clean_cache(self):
        """
        Empty test cache directory
        """
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except OSError as e:
            print(e)

    @property
    def power_loss(self) -> Dict[str, DataArray]:
        return self._power_loss.data

    @power_loss.setter
    def power_loss(self, data: Dict[str, DataArray]) -> None:
        self._power_loss.data = data

    @property
    def sxr_power_loss(self) -> Dict[str, DataArray]:
        return self._sxr_power_loss.data

    @sxr_power_loss.setter
    def sxr_power_loss(self, data: Dict[str, DataArray]) -> None:
        self._sxr_power_loss.data = data

    @property
    def sxr_emissivity(self) -> DataArray:
        return self._sxr_emissivity.data

    @sxr_emissivity.setter
    def sxr_emissivity(self, data: DataArray) -> None:
        self._sxr_emissivity.data = data

    @property
    def sxr_fitted_symmetric_emissivity(self) -> DataArray:
        return self._sxr_fitted_symmetric_emissivity.data

    @sxr_fitted_symmetric_emissivity.setter
    def sxr_fitted_symmetric_emissivity(self, data: DataArray) -> None:
        self._sxr_fitted_symmetric_emissivity.data = data

    @property
    def sxr_fitted_asymmetry_parameter(self) -> DataArray:
        return self._sxr_fitted_asymmetry_parameter.data

    @sxr_fitted_asymmetry_parameter.setter
    def sxr_fitted_asymmetry_parameter(self, data: DataArray) -> None:
        self._sxr_fitted_asymmetry_parameter.data = data

    @property
    def electron_density(self) -> DataArray:
        return self._electron_density.data

    @electron_density.setter
    def electron_density(self, data: DataArray) -> None:
        self._electron_density.data = data

    @property
    def electron_temperature(self) -> DataArray:
        return self._electron_temperature.data

    @electron_temperature.setter
    def electron_temperature(self, data: DataArray) -> None:
        self._electron_temperature.data = data

    @property
    def ion_temperature(self) -> DataArray:
        return self._ion_temperature.data

    @ion_temperature.setter
    def ion_temperature(self, data: DataArray) -> None:
        self._ion_temperature.data = data

    @property
    def toroidal_rotation(self) -> DataArray:
        return self._toroidal_rotation.data

    @toroidal_rotation.setter
    def toroidal_rotation(self, data: DataArray) -> None:
        self._toroidal_rotation.data = data

    @property
    def extra_zeff_element_concentration(self) -> DataArray:
        return self._extra_zeff_element_concentration.data

    @extra_zeff_element_concentration.setter
    def extra_zeff_element_concentration(self, data: DataArray) -> None:
        self._extra_zeff_element_concentration.data = data

    @property
    def sxr_calibration_factor(self) -> float:
        return self._sxr_calibration_factor.data

    @sxr_calibration_factor.setter
    def sxr_calibration_factor(self, data: float) -> None:
        self._sxr_calibration_factor.data = data

    def _calculate_sxr_calibration_factor(self) -> float:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_sxr_calibration_factor"
            " method"
        )

    def calculate_sxr_calibration_factor(self) -> float:
        self.sxr_calibration_factor = self._calculate_sxr_calibration_factor()
        return self.sxr_calibration_factor

    @property
    def sxr_rescale_factor(self) -> float:
        return self._sxr_rescale_factor.data

    @sxr_rescale_factor.setter
    def sxr_rescale_factor(self, data: float) -> None:
        self._sxr_rescale_factor.data = data

    def _calculate_sxr_rescale_factor(self) -> float:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_sxr_rescale_factor method"
        )

    def calculate_sxr_rescale_factor(self) -> float:
        self.sxr_rescale_factor = self._calculate_sxr_rescale_factor()
        return self.sxr_rescale_factor

    @property
    def power_loss_charge_averaged(self) -> DataArray:
        return self._power_loss_charge_averaged.data

    @power_loss_charge_averaged.setter
    def power_loss_charge_averaged(self, data: DataArray) -> None:
        self._power_loss_charge_averaged.data = data

    def _calculate_power_loss_charge_averaged(self) -> DataArray:
        if self.power_loss is None:
            raise UserWarning("Power loss not yet calculated")
        return xr.concat(
            [
                val.sum("ion_charges").assign_attrs(val.attrs)
                for val in self.power_loss.values()
            ],
            dim="element",
        ).assign_coords({"element": list(self.power_loss.keys())})

    def _calculate_sxr_power_loss_charge_averaged(self) -> DataArray:
        if self.sxr_power_loss is None:
            raise UserWarning("Power loss not yet calculated")
        return xr.concat(
            [
                val.sum("ion_charges").assign_attrs(val.attrs)
                for val in self.sxr_power_loss.values()
            ],
            dim="element",
        ).assign_coords({"element": list(self.sxr_power_loss.keys())})

    @property
    def sxr_power_loss_charge_averaged(self) -> DataArray:
        return self._sxr_power_loss_charge_averaged.data

    @sxr_power_loss_charge_averaged.setter
    def sxr_power_loss_charge_averaged(self, data: DataArray) -> None:
        self._sxr_power_loss_charge_averaged.data = data

    @property
    def asymmetry_parameter_high_z(self) -> DataArray:
        return self._asymmetry_parameter_high_z.data

    def _calculate_asymmetry_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_asymmetry_high_z method"
        )

    def calculate_asymmetry_high_z(self) -> DataArray:
        self._asymmetry_parameter_high_z.update()
        return self.asymmetry_parameter_high_z

    @property
    def asymmetry_parameter_other_z(self) -> DataArray:
        return self._asymmetry_parameter_other_z.data

    def _calculate_asymmetry_other_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement"
            " _calculate_asymmetry_other_z method"
        )

    def calculate_asymmetry_other_z(self) -> DataArray:
        self._asymmetry_parameter_other_z.update()
        return self.asymmetry_parameter_other_z

    def _derive_asymmetry_from_density(self, density: DataArray) -> DataArray:
        """
        Derive an asymmetry parameter from comparison of lfs (theta=0) and hfs (theta=1)
        impurity density data
        """
        lfs = self.n_high_z.sel({"theta": 0}, method="nearest", drop=True).copy()
        hfs = self.n_high_z.sel({"theta": np.pi}, method="nearest", drop=True).copy()
        asymmetry: DataArray = (lfs - hfs) / (lfs + hfs)
        return asymmetry

    @property
    def derived_asymmetry_high_z(self) -> DataArray:
        return self._derived_asymmetry_high_z.data

    def _calculate_derived_asymmetry_high_z(self) -> DataArray:
        return self._derive_asymmetry_from_density(density=self.n_high_z)

    def calculate_derived_asymmetry_high_z(self) -> DataArray:
        self._derived_asymmetry_high_z.update()
        return self.derived_asymmetry_high_z

    @property
    def derived_asymmetry_other_z(self) -> DataArray:
        return self._derived_asymmetry_other_z.data

    def _calculate_derived_asymmetry_other_z(self) -> DataArray:
        return self._derive_asymmetry_from_density(density=self.n_other_z)

    def calculate_derived_asymmetry_other_z(self) -> DataArray:
        self._derived_asymmetry_other_z.update()
        return self.derived_asymmetry_other_z

    @property
    def n_high_z(self) -> DataArray:
        return self._n_high_z.data

    @n_high_z.setter
    def n_high_z(self, data: DataArray) -> None:
        self._n_high_z.data = data

    def _calculate_n_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_high_z method"
        )

    def calculate_n_high_z(self) -> DataArray:
        self.n_high_z = self._calculate_n_high_z()
        return self.n_high_z

    def _extrapolate_n_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _extrapolate_n_high_z method"
        )

    def extrapolate_n_high_z(self) -> DataArray:
        self.n_high_z = self._extrapolate_n_high_z()
        return self.n_high_z

    def _rescale_n_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _rescale_n_high_z method"
        )

    def rescale_n_high_z(self) -> DataArray:
        self.n_high_z = self._rescale_n_high_z()
        return self.n_high_z

    def _optimise_n_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _optimise_n_high_z method"
        )

    def optimise_n_high_z(self) -> DataArray:
        self.n_high_z = self._optimise_n_high_z()
        return self.n_high_z

    @property
    def n_zeff_el(self) -> DataArray:
        return self._n_zeff_el.data

    def _calculate_n_zeff_el(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_zeff_el method"
        )

    def calculate_n_zeff_el(self) -> DataArray:
        self._n_zeff_el.update()
        return self.n_zeff_el

    @property
    def n_zeff_el_extra(self) -> DataArray:
        return self._n_zeff_el_extra.data

    def _calculate_n_zeff_el_extra(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_zeff_el_extra method"
        )

    def calculate_n_zeff_el_extra(self) -> DataArray:
        self._n_zeff_el_extra.update()
        return self.n_zeff_el_extra

    @property
    def n_other_z(self) -> DataArray:
        return self._n_other_z.data

    def _calculate_n_other_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_other_z method"
        )

    def calculate_n_other_z(self) -> DataArray:
        self._n_other_z.update()
        return self.n_other_z

    @property
    def n_main_ion(self) -> DataArray:
        return self._n_main_ion.data

    def _calculate_n_main_ion(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_main_ion method"
        )

    def calculate_n_main_ion(self) -> DataArray:
        self._n_main_ion.update()
        return self.n_main_ion

    @property
    def ion_densities(self) -> DataArray:
        return self._ion_densities.data

    def _combine_ion_density_arrays(self) -> DataArray:
        return concat(
            [
                self.n_high_z,
                self.n_zeff_el.expand_dims({"theta": self.theta}),  # type: ignore
                self.n_zeff_el_extra.expand_dims({"theta": self.theta}),  # type: ignore
                self.n_other_z,
                self.n_main_ion,
            ],
            dim="element",
        ).assign_coords({"element": self.ion_species})

    @property
    def derived_bolometry(self) -> DataArray:
        return self._derived_bolometry.data

    def _calculate_derived_bolometry(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_derived_bolometry method"
        )

    def calculate_derived_bolometry(self) -> DataArray:
        self._derived_bolometry.update()
        return self._derived_bolometry.data
