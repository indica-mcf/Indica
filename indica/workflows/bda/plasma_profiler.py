import os

import matplotlib.pyplot as plt
import xarray as xr
import yaml

from indica.defaults.load_defaults import load_default_objects
from indica.models.plasma import Plasma
from indica.profilers.profiler_base import ProfilerBase
from indica.profilers.profiler_gauss import ProfilerGauss, initialise_gauss_profilers

PLASMA_ATTRIBUTE_NAMES = [
    "electron_temperature",
    "electron_density",
    "ion_temperature",
    "ion_density",
    "impurity_density",
    "fast_density",
    "pressure_fast",
    "neutral_density",
    "zeff",
    "meanz",
    "wp",
    "wth",
    "pressure_tot",
    "pressure_th",
]


def map_plasma_profile_to_midplane(plasma: Plasma, profiles: dict):
    midplane_profiles: dict = {}

    R = plasma.R_midplane
    z = plasma.z_midplane
    _rho, _, _ = plasma.equilibrium.flux_coords(R, z, plasma.t)
    rho = _rho.swap_dims({"index": "R"}).drop_vars("index")

    for key, value in profiles.items():
        if "rho_poloidal" not in value.dims:
            continue
        if not hasattr(plasma, key):
            continue
        midplane_profiles[key] = value.interp(rho_poloidal=rho)
    return midplane_profiles


class PlasmaProfiler:
    def __init__(
        self,
        plasma: Plasma,
        profilers: dict[ProfilerBase],
        plasma_attribute_names=None,
    ):
        """
        Interface Profiler objects with Plasma object to generate plasma profiles
        and update them.

        Parameters
        ----------
        plasma
            Plasma object
        profilers
            dictionary of Profiler objects to generate profiles
        """

        if plasma_attribute_names is None:
            plasma_attribute_names = PLASMA_ATTRIBUTE_NAMES
        self.plasma = plasma
        self.profilers = profilers
        self.plasma_attribute_names = plasma_attribute_names
        self.phantom = None
        self.phantom_profiles = None

    def update_profilers(self, profilers: dict):
        for profile_name, profiler in profilers.items():
            self.profilers[profile_name] = profiler

    def set_profiles(self, profiles: dict[xr.DataArray], t: float = None):
        if t is None:
            t = self.plasma.time_to_calculate

        for profile_name, profile in profiles.items():
            _prof_identifiers = profile_name.split(
                ":"
            )  # impurities have ':' to identify elements
            if profile_name.__contains__(":"):
                if _prof_identifiers[1] in self.plasma.elements:
                    getattr(self.plasma, _prof_identifiers[0]).loc[
                        dict(t=t, element=_prof_identifiers[-1])
                    ] = profile
                else:
                    print(
                        f"profile {profile_name} can't be set because "
                        f"{_prof_identifiers[1]} not in plasma.elements"
                    )
            else:
                getattr(self.plasma, profile_name).loc[dict(t=t)] = profile

    def save_phantoms(self, phantom=False):
        #  if phantoms return profiles otherwise return empty arrays
        self.phantom = phantom
        phantom_profiles = self.plasma_attributes()
        if not phantom:
            for key, value in phantom_profiles.items():
                phantom_profiles[key] = value * 0
        self.phantom_profiles = phantom_profiles
        return phantom_profiles

    def plasma_attributes(self):
        plasma_attributes = {}
        for attribute in self.plasma_attribute_names:
            plasma_attributes[attribute] = getattr(self.plasma, attribute).sel(
                t=self.plasma.time_to_calculate
            )
        return plasma_attributes

    def __call__(self, parameters: dict = None, t=None):
        """
        Set parameters of desired profilers and assign to plasma class profiles

        Parameters
        ----------
        parameters
            Flat dictionary of {"profile_name.parameter":value}
            Special case for impurity density:
                {"profile_name:element.parameter":value}
        """
        if parameters is None:
            parameters = {}

        _profiles_to_update: list = []

        # set params for all profilers
        for parameter_name, parameter in parameters.items():
            profile_name, profile_param_name = parameter_name.split(".")
            if profile_name not in self.profilers.keys():
                continue

            if not hasattr(self.profilers[profile_name], profile_param_name):
                raise ValueError(
                    f"No parameter {profile_param_name} available for {profile_name}"
                )
            self.profilers[profile_name].set_parameters(
                **{profile_param_name: parameter}
            )
            _profiles_to_update.append(profile_name)

        # Update only desired profiles or if no parameters given update all
        if _profiles_to_update:
            profiles_to_update = list(set(_profiles_to_update))
        else:
            profiles_to_update = list(self.profilers.keys())

        updated_profiles = {
            profile_to_update: self.profilers[profile_to_update]()
            for profile_to_update in profiles_to_update
        }
        self.set_profiles(updated_profiles, t)
        return


if __name__ == "__main__":
    example_plasma = load_default_objects("st40", "plasma")
    profilers = initialise_gauss_profilers(
        example_plasma.rho, profile_names=["electron_density", "ion_temperature"]
    )
    plasma_profiler = PlasmaProfiler(
        plasma=example_plasma,
        profilers=profilers,
    )

    plasma_profiler(
        parameters={
            "electron_density.y0": 10e19,
            "ion_temperature.y0": 1e3,
        }
    )
    plasma_profiler.profilers["ion_temperature"].plot()
    plt.show()
