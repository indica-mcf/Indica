import flatdict

from indica.defaults.load_defaults import load_default_objects
from indica.profilers import ProfilerGauss, Profiler
from indica.models.plasma import Plasma
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_PROFILE_PARAMS = {
    "electron_density.y0": 5e19,
    "electron_density.y1": 2e18,
    "electron_density.yend": 1e18,
    "electron_density.wped": 3,
    "electron_density.wcenter": 0.3,
    "electron_density.peaking": 1.2,
    "impurity_density:ar.y0": 1e17,
    "impurity_density:ar.y1": 1e15,
    "impurity_density:ar.yend": 1e15,
    "impurity_density:ar.wcenter": 0.3,
    "impurity_density:ar.wped": 3,
    "impurity_density:ar.peaking": 2,
    "electron_temperature.y0": 3000,
    "electron_temperature.y1": 50,
    "electron_temperature.yend": 10,
    "electron_temperature.wcenter": 0.2,
    "electron_temperature.wped": 3,
    "electron_temperature.peaking": 1.5,
    "ion_temperature.y0": 5000,
    "ion_temperature.y1": 50,
    "ion_temperature.yend": 10,
    "ion_temperature.wcenter": 0.2,
    "ion_temperature.wped": 3,
    "ion_temperature.peaking": 1.5,
    "neutral_density.y0": 1e14,
    "neutral_density.y1": 5e15,
    "neutral_density.yend": 5e15,
    "neutral_density.wcenter": 0.01,
    "neutral_density.wped": 18,
    "neutral_density.peaking": 1,
    "toroidal_rotation.y0": 500.0e3,
    "toroidal_rotation.y1": 10.0e3,
    "toroidal_rotation.yend": 0.0,
    "toroidal_rotation.peaking": 1.5,
    "toroidal_rotation.wcenter": 0.35,
    "toroidal_rotation.wped": 3,
}
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

def initialise_gauss_profilers(profile_params: dict, xspl: np.ndarray):
    # considering whether profilers should be a dataclass or named tuple rather than bare dictionary

    flat_profile_params = flatdict.FlatDict(profile_params, ".")
    profile_names = flat_profile_params.as_dict().keys()

    profilers = {profile_name: ProfilerGauss(datatype=profile_name.split(":")[0],
                                        parameters=flat_profile_params[profile_name],
                                        xspl=xspl)
                 for profile_name in profile_names}

    return profilers



class PlasmaProfiler:
    def __init__(self, plasma: Plasma, profilers: dict[Profiler]):
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
        self.plasma = plasma
        self.profilers = profilers
        self.plasma_attribute_names = PLASMA_ATTRIBUTE_NAMES

    def set_profiles(self, profiles: dict[xr.DataArray], t: float = None):
        if t is None:
            t = self.plasma.time_to_calculate

        for profile_name, profile in profiles.items():
            _prof_identifiers = profile_name.split(":")  # impurities have ':' to identify elements
            if profile_name.__contains__(":"):
                getattr(self.plasma, _prof_identifiers[0]).loc[dict(t=t, element=_prof_identifiers[-1])] = profile
            else:
                getattr(self.plasma, profile_name).loc[dict(t=t)] = profile

    def save_phantoms(self, phantoms=False):
        #  if phantoms return profiles otherwise return empty arrays
        phantom_profiles = self.plasma_attributes()
        if not phantoms:
            for key, value in phantom_profiles.items():
                phantom_profiles[key] = value * 0
        return phantom_profiles


    def plasma_attributes(self):
        plasma_attributes = {}
        for attribute in self.plasma_attribute_names:
            plasma_attributes[attribute] = getattr(self.plasma, attribute)
        return plasma_attributes


    def __call__(self, parameters: dict):
        """
        Set parameters of desired profilers and assign to plasma class profiles

        Parameters
        ----------
        parameters
            Flat dictionary of {"profile_name.parameter":value}
            Special case for impurity density:
                {"profile_name:element.parameter":value}
        """

        _profiles_to_update: list = []

        # set params for all profilers
        for parameter_name, parameter in parameters.items():
            profile_name, profile_param_name = parameter_name.split(".")

            if not hasattr(self.profilers[profile_name], profile_param_name):
                raise ValueError(
                    f"No parameter {profile_param_name} available for {profile_name}"
                )
            self.profilers[profile_name].set_parameters(**{profile_param_name: parameter})
            _profiles_to_update.append(profile_name)

        # Update only desired profiles
        profiles_to_update = list(set(_profiles_to_update))
        updated_profiles = {profile_to_update: self.profilers[profile_to_update]()
                            for profile_to_update in profiles_to_update}
        self.set_profiles(updated_profiles)
        return


if __name__ == "__main__":

    plasma = load_default_objects("st40", "plasma")
    gauss_profilers = initialise_gauss_profilers(DEFAULT_PROFILE_PARAMS, xspl=plasma.rho)
    plasma_profiler = PlasmaProfiler(plasma=plasma, profilers=gauss_profilers)

    plasma_profiler(parameters={"electron_density.y0": 10e19, "electron_density.y1":1e19, "electron_density.yend":1e19})
    plasma_profiler.profilers["electron_density"].plot()
    plt.show()
