import flatdict
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.defaults.load_defaults import load_default_objects
from indica.models.plasma import Plasma
from indica.profilers import Profiler
from indica.profilers import ProfilerGauss

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
    "impurity_density:c.y0": 5e17,
    "impurity_density:c.y1": 2e17,
    "impurity_density:c.yend": 2e17,
    "impurity_density:c.wcenter": 0.3,
    "impurity_density:c.wped": 3,
    "impurity_density:c.peaking": 1.2,
    "impurity_density:he.y0": 5e17,
    "impurity_density:he.y1": 2e17,
    "impurity_density:he.yend": 2e17,
    "impurity_density:he.wcenter": 0.3,
    "impurity_density:he.wped": 3,
    "impurity_density:he.peaking": 1.2,
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
        # TODO: Changing NaN results to zeros messes with distributions
        # midplane_profiles[key] = xr.where(
        #     np.isfinite(_prof_midplane), _prof_midplane, 0.0
        # )
    return midplane_profiles


def initialise_gauss_profilers(
    xspl: np.ndarray, profile_params: dict = None, profiler_names: list = None
):
    # Should profilers be a dataclass or named tuple rather than bare dictionary
    if profile_params is None:
        profile_params = DEFAULT_PROFILE_PARAMS
    flat_profile_params = flatdict.FlatDict(profile_params, ".")

    if profiler_names is None:
        profile_names = flat_profile_params.as_dict().keys()
    else:
        profile_names = profiler_names

    profilers = {
        profile_name: ProfilerGauss(
            datatype=profile_name.split(":")[0],
            parameters=flat_profile_params[profile_name],
            xspl=xspl,
        )
        for profile_name in profile_names
    }

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
    gauss_profilers = initialise_gauss_profilers(
        profile_params=DEFAULT_PROFILE_PARAMS, xspl=example_plasma.rho
    )
    plasma_profiler = PlasmaProfiler(plasma=example_plasma, profilers=gauss_profilers)

    plasma_profiler(
        parameters={
            "electron_density.y0": 10e19,
            "electron_density.y1": 1e19,
            "electron_density.yend": 1e19,
            "ion_temperature.y0": 1e3,
        }
    )
    plasma_profiler.profilers["ion_temperature"].plot()
    plt.show()
