"""
Base script for running InDiCA analysis tests.
Run with specific data source (e.g. JET JPF/PPF data)
"""
import json
from pathlib import Path
from shutil import get_terminal_size
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import xarray as xr
from xarray import concat
from xarray import DataArray

from indica.utilities import coord_array
from indica.workflow.observers import Observable
from indica.workflow.observers import Observer


class BaseWorkflow:
    """
    Common components of running a benchmark InDiCA analysis for continuous
    testing.
    Subclass and specialise for specific use cases
    """

    cache_dir = Path(".").absolute().parent / "test_cache"
    cache_file = cache_dir / "cache.json"

    def __init__(self, config_file: Union[str, Path] = "input.json"):
        """
        Get test parameters from json configuration file, sets defaults for values
        not present in file.

        :param config_key: Key to get parameters for in configuration file
        :type config_key: Hashable
        """
        # Config parameters
        self.input = self._read_test_case(config_file=config_file)
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
        self.zeff_el_extra: str = self.input.get("zeff_el_2", "he")
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
        self._n_high_z = Observer(
            operator=self._calculate_n_high_z,
            depends_on=[
                self._sxr_emissivity,
                self._sxr_power_loss,
                self._electron_density,
                self._sxr_calibration_factor,
                self._sxr_rescale_factor,
            ],
        )
        self._asymmetry_parameter = Observer(
            operator=self._calculate_asymmetry_parameter,
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
            depends_on=[self._n_high_z, self._asymmetry_parameter],
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
        self._derived_bolometry = Observer(
            operator=self._calculate_derived_bolometry,
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

    def __call__(self, setup_only: bool = False, save_outputs: bool = True):
        term_width = get_terminal_size(fallback=(80, 24)).columns - 1
        step_template = (
            "\n" + "*" * term_width + "\n" + "\t{}\n" + "*" * term_width + "\n"
        )
        for step_name, step_method, key_attr in self.setup_steps:
            if hasattr(self, key_attr):
                continue
            print(step_template.format(step_name))
            step_method()
            if save_outputs is True:
                self.save(filename=self.cache_file, **self.__dict__)

        if setup_only is False:
            # TODO add some iterating and termination conditions here
            for step_name, step_method in self.analysis_steps:
                print(step_template.format(step_name))
                step_method()
                if save_outputs is True:
                    self.save(filename=self.cache_file, **self.__dict__)

    def __str__(self) -> str:
        outp = ""
        formatter = "Step = {:<35s} | status = {:<8s}\n"

        for step_name, status in self._progress():
            outp += formatter.format(
                step_name, "Complete" if status is True else "Waiting"
            )
        return outp.rstrip("\n")

    def save(
        self,
        filename: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        pass  # TODO

    @classmethod
    def restore(cls, filename: Optional[Union[str, Path]] = None) -> "BaseWorkflow":
        pass  # TODO

    @property
    def setup_steps(self) -> List[Tuple[str, Callable, str]]:
        raise NotImplementedError(
            f"{self.__class__} does not implemented setup_steps property"
        )

    @property
    def analysis_steps(self) -> List[Tuple[str, Callable]]:
        raise NotImplementedError(
            f"{self.__class__} does not implemented analysis_steps property"
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
    def sxr_calibration_factor(self) -> DataArray:
        return self._sxr_calibration_factor.data

    @sxr_calibration_factor.setter
    def sxr_calibration_factor(self, data: DataArray) -> None:
        self._sxr_calibration_factor.data = data

    @property
    def sxr_rescale_factor(self) -> DataArray:
        return self._sxr_rescale_factor.data

    @sxr_rescale_factor.setter
    def sxr_rescale_factor(self, data: DataArray) -> None:
        self._sxr_rescale_factor.data = data

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
                for val in self.power_loss.values()
            ],
            dim="element",
        ).assign_coords({"element": list(self.power_loss.keys())})

    @property
    def sxr_power_loss_charge_averaged(self) -> DataArray:
        return self._sxr_power_loss_charge_averaged.data

    @sxr_power_loss_charge_averaged.setter
    def sxr_power_loss_charge_averaged(self, data: DataArray) -> None:
        self._sxr_power_loss_charge_averaged.data = data

    @property
    def asymmetry_parameter(self) -> DataArray:
        return self._asymmetry_parameter.data

    def _calculate_asymmetry_parameter(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_asymmetry_parameter method"
        )

    def calculate_asymmetry_parameter(self) -> DataArray:
        self._asymmetry_parameter.update()
        return self.asymmetry_parameter

    @property
    def n_high_z(self) -> DataArray:
        return self._n_high_z.data

    def _calculate_n_high_z(self) -> DataArray:
        raise NotImplementedError(
            f"{self.__class__} does not implement _calculate_n_high_z method"
        )

    def calculate_n_high_z(self) -> DataArray:
        self._n_high_z.update()
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
