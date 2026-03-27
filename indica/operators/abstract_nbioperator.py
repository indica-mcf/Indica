from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

from xarray import DataArray

from indica import Equilibrium
from indica.converters import CoordinateTransform
from indica.converters import LineOfSightTransform
from indica.utilities import get_element_info


class NbiOperator(ABC):
    """Abstract class to run NBI modelling codes e.g. Fidasim, Nubeam"""

    equilibrium: Equilibrium
    transform: LineOfSightTransform

    def __init__(
        self,
        name: str,
        energy: float,
        power: float,
        nbi_element: str,
        current_fractions: Tuple[float, float, float],
    ):
        """
        General parameters of any NBI object

        name - NBI object string identifier
        element - NBI injected element symbol (e.g. "d" for deuterium)
        energy - NBI energy in eV
        power - NBI power in W
        current_fractions - NBI fractions for 1st, 2nd, and 3rd energy
        """
        self.name = name
        _element_info = get_element_info(nbi_element)
        self.nbi_element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }
        self.energy = energy
        self.power = power
        self.current_fractions = current_fractions

    def set_transform(self, transform: CoordinateTransform):
        """
        The coordinate transform (if any) applicable to the operator
        """
        self.transform = transform

    def set_parameters(self, **kwargs):
        """
        Set any model kwargs
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __call__(
        self,
        Ti: DataArray,
        Te: DataArray,
        Ne: DataArray,
        Nn: DataArray,
        Vtor: DataArray,
        Zeff: DataArray,
        MeanZ: DataArray,
        target_element: str,
        t: float,
        file_name: Optional[str] = "",
        pulse: int = 0,
        machine: str = "tokamak",
        prepare_kwargs: dict = {},
        run_kwargs: dict = {},
    ) -> dict:
        """
        Run NBI code for specified time-point (one only!)

        target_element - plasma main ion element symbol (e.g. "d" for deuterium)
        file_name - first part of the file name to save the NBI model data to

        # TODO: thermal neutral densities in indica to be changed from Nh to Nn
        """
        if not hasattr(self, "transform"):
            raise ValueError("transform is required (set it before calling)")

        if not hasattr(self.transform, "equilibrium"):
            raise ValueError("transform is missing equilibrium data")

        _element_info = get_element_info(target_element)
        self.target_element_info = {
            "Z": _element_info[0],
            "A": _element_info[1],
            "name": _element_info[2],
            "symbol": _element_info[3],
        }

        self.t = t
        self.file_name = file_name
        self.pulse = pulse
        self.machine = machine

        self.Ti = Ti.interp(t=t)
        self.Te = Te.interp(t=t)
        self.Ne = Ne.interp(t=t)
        self.Nn = Nn.interp(t=t)
        self.Vtor = Vtor.interp(t=t)
        self.Zeff = Zeff.sum("element").interp(t=t)
        self.MeanZ = MeanZ.mean("element").interp(t=t)

        """Prepare input data structure for NBI code"""
        self.prepare(**prepare_kwargs)

        """Run NBI code"""
        self.run(**run_kwargs)

        """Reorganise NBI code output to return Indica-native results"""
        result = self.refactor_output()

        return result

    @abstractmethod
    def prepare(self, **kwargs):
        raise NotImplementedError(
            "Implement this method to reorganise input data to feed to NBI code"
        )

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError(
            "Implement this method to run NBI code and return results"
        )

    @abstractmethod
    def refactor_output(self):
        """
        Result data structure must be identical for all NBI codes and match the
        attributes in the Plasma class for seamless mapping

        result = {
            "neutral_density":DataArray("t", "rhop"),
            "fast_ion_density":DataArray("t", "rhop"),
            "parallel_fast_ion_pressure":DataArray("t", "rhop"),
            "perpendicular_fast_ion_pressure":DataArray("t", "rhop")
        }

        """
        raise NotImplementedError("Implement this method to reorganise NBI code output")
