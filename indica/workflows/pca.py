from dataclasses import dataclass, field
from typing import Dict, Callable, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import flatdict

from sklearn.decomposition import PCA
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde

from indica.workflows.plasma_profiler import initialise_gauss_profilers
from indica.workflows.priors import PriorManager, PriorBasis, sample_from_priors
from indica.profilers import Profiler, ProfilerBasis


def pca_fit(profiles: np.ndarray, ncomps, verbose=True) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=ncomps)
    pca.fit(profiles)
    pca_profiles = pca.transform(profiles)
    if verbose:
        print(f"variance per component: {pca.explained_variance_ratio_ * 100}")
        print(f"summed variance: {np.sum(pca.explained_variance_ratio_ * 100)}")
    return pca_profiles, pca


def fit_priors(pca_profiles: np.ndarray, prior_values=None, ) -> LinearNDInterpolator:
    # TODO: add non 1 or 0 prior evaluations
    # each profile has a prior value
    if prior_values is None:
        prior_values = np.ones(shape=pca_profiles.shape[0])

    prior_fit = LinearNDInterpolator(pca_profiles, prior_values, fill_value=0, rescale=False)
    return prior_fit


def project_priors(prior_fit, pca_profiles) -> Tuple[np.ndarray, np.ndarray]:
    grid = [np.linspace(pca_profiles[:, dim].max(), pca_profiles[:, dim].min(), 100)
            for dim in range(pca_profiles.shape[1])]
    mesh_grid = np.meshgrid(*grid)
    Z = prior_fit(*mesh_grid)

    return Z, mesh_grid


def fit_KDE_prior(linear_fit: LinearNDInterpolator, size=100000) -> gaussian_kde:
    sampled_points = linear_fit.points
    min = sampled_points.min(axis=0)
    max = sampled_points.max(axis=0)
    ranges = [(min[idx], max[idx]) for idx in range(sampled_points.shape[1])]

    random_points = np.array([np.random.uniform(r[0], r[1], size=size) for r in ranges])
    accepted_points = random_points[:, linear_fit(*random_points).nonzero()[0]]
    kernel = gaussian_kde(accepted_points)

    marginalised_kernels = {f"weight_{i + 1}": kernel.marginal(i) for i in range(kernel.d)}
    return marginalised_kernels


def _reconstruct_profile(basis_function: np.ndarray, bias: np.ndarray, weights: np.ndarray, ):
    return np.dot(weights, basis_function) + bias


def _plot_principle_comps(pca, profile_name):
    plt.figure()
    plt.title(f"{profile_name}: principle components")
    for n in range(pca.components_.shape[0]):
        plt.plot(np.arange(pca.components_.shape[1]), pca.components_[n,], "-", label=f"component {n + 1}")
    plt.ylabel("A.U.")
    plt.xlabel("radial co-ordinate (-)")
    plt.legend()


def _plot_profile_fits(profiles: np.ndarray, projected_profiles: np.ndarray, profile_name, nsamples: int = 100):
    prof_array = xr.DataArray(profiles, dims=("index", "radial"),
                              coords=(np.arange(0, profiles.shape[0]), np.linspace(0, 1, profiles.shape[1])))
    proj_prof_array = xr.DataArray(projected_profiles, dims=("index", "radial"),
                                   coords=(
                                       np.arange(0, projected_profiles.shape[0]),
                                       np.linspace(0, 1, projected_profiles.shape[1])))

    plt.figure()
    plt.title(f"{profile_name}: Reconstructed Profiles")
    plt.fill_between(
        prof_array.radial,
        prof_array.quantile(0.0, dim="index"),
        prof_array.quantile(1.0, dim="index"),
        label="min-max of input profiles",
        zorder=3,
        color="red",
        alpha=0.5,
    )

    for sample in np.arange(0, nsamples):
        plt.plot(
            proj_prof_array.radial,
            proj_prof_array.sel(index=sample),
            zorder=3,
            # color="blue",
            # alpha=0.5,
        )

    plt.legend()
    return


def _plot_projected_priors(Z, meshgrid, pca_profiles, profile_name):
    if Z.ndim == 2:
        plt.figure()
        plt.title(f"{profile_name}: priors")
        plt.pcolormesh(meshgrid[0], meshgrid[1], Z, shading="auto")
        plt.plot(pca_profiles[:, 0], pca_profiles[:, 1], "ok", label="inputs")
        plt.legend()
        plt.colorbar()


def _plot_KDE_comparison(kernels: Dict[str, gaussian_kde], prior: LinearNDInterpolator, size=1000):
    plt.figure()
    sample = [kernel.resample(size) for name, kernel in kernels.items()]

    values = prior(*sample)
    accepted_mask = values != 0

    plt.title("KDE Samples")
    plt.plot(sample[0][accepted_mask], sample[1][accepted_mask], "gx", label="accepted")
    plt.plot(sample[0][np.invert(accepted_mask)], sample[1][np.invert(accepted_mask)], "rx", label="rejected")
    plt.legend()


@dataclass
class PCAProcess:
    gaussian_profiles: Dict[str, np.ndarray]

    ncomps: int = 2
    pca_weights: Dict[str, np.ndarray] = field(default_factory=lambda: {})  # (nsamples, ncomps)
    reconstructed_profiles: Dict[str, np.ndarray] = field(default_factory=lambda: {})  # (nsamples, xspl)
    pca_fits: Dict[str, PCA] = field(default_factory=lambda: {})
    prior_fits: Dict[str, LinearNDInterpolator] = field(default_factory=lambda: {})

    projected_priors: Dict[str, np.ndarray] = field(default_factory=lambda: {})  # (nsamples, ncomps)
    projected_prior_meshgrid: Dict[str, np.ndarray] = field(default_factory=lambda: {})  # (nsamples, nsamples, ...)
    KDE_priors: Dict[str, gaussian_kde] = field(default_factory=lambda: {})
    conditional_priors: Dict[str, Callable] = field(default_factory=lambda: {})

    def __post_init__(self):
        """
        DO EVERYTHING. TODO: Break this up
        """

        for profile_name in self.gaussian_profiles.keys():
            print(f"PCA Profile: {profile_name}")
            self.pca_weights[profile_name], self.pca_fits[profile_name] = pca_fit(
                self.gaussian_profiles[profile_name], self.ncomps)

            self.prior_fits[profile_name] = fit_priors(self.pca_weights[profile_name])
            name = "/".join([f"{profile_name}.basis_{i + 1}" for i in range(self.ncomps)])

            self.conditional_priors[name] = self.prior_fits[profile_name]
            self.reconstructed_profiles[profile_name] = _reconstruct_profile(
                self.pca_fits[profile_name].components_,
                self.pca_fits[profile_name].mean_,
                self.pca_weights[profile_name], )

            self.projected_priors[profile_name], self.projected_prior_meshgrid[profile_name] = \
                project_priors(self.prior_fits[profile_name], self.pca_weights[profile_name])
            self.KDE_priors[profile_name] = fit_KDE_prior(self.prior_fits[profile_name], )
        self.basis_priors = flatdict.FlatDict(self.KDE_priors, ".")

    def plot_profile(self, profile_name: str):

        _plot_profile_fits(self.gaussian_profiles[profile_name], self.reconstructed_profiles[profile_name],
                           profile_name)
        _plot_principle_comps(self.pca_fits[profile_name], profile_name)
        _plot_projected_priors(self.projected_priors[profile_name],
                               self.projected_prior_meshgrid[profile_name],
                               self.pca_weights[profile_name],
                               profile_name)
        _plot_KDE_comparison(self.KDE_priors[profile_name], self.prior_fits[profile_name])

    def plot_all(self):
        for name in self.gaussian_profiles.keys():
            self.plot_profile(name)
        plt.show(block=True)


def pca_workflow(prior_manager: PriorManager, opt_profiles: list, x_grid: xr.DataArray,
                 n_components=2, num_prof_samples: int = int(1e4)):
    prior_names = prior_manager.get_prior_names_from_profile_names(opt_profiles)
    param_samples: np.ndarray = sample_from_priors(prior_names, prior_manager.priors, size=num_prof_samples)
    param_samples: Dict[str, np.ndarray] = \
        {prior_name: param_samples[:, idx] for idx, prior_name in enumerate(prior_names)}

    profilers = initialise_gauss_profilers(xspl=x_grid, profiler_names=opt_profiles)
    profiles = sample_gauss_profiles(param_samples, profilers=profilers, size=num_prof_samples)

    pca_process = PCAProcess(gaussian_profiles=profiles, ncomps=n_components)
    pca_process.basis_priors = {key: PriorBasis(kernel=value) for key, value in pca_process.basis_priors.items()}

    new_profilers = {}
    for profile_name, _profiles in profiles.items():
        _basis_func = pca_process.pca_fits[profile_name].components_
        _bias = pca_process.pca_fits[profile_name].mean_
        new_profilers[profile_name] = ProfilerBasis(basis_functions=_basis_func, bias=_bias, ncomps=n_components,
                                                    radial_grid=x_grid)
    return pca_process, new_profilers


def sample_gauss_profiles(sample_params: Dict[str, np.ndarray], profilers: dict, size: int) -> Dict[str, np.ndarray]:
    # TODO: Vectorise profilers? (only if pca too slow)
    profiles = {}
    for _profile_name, _profiler in profilers.items():

        _params = {key.split(".")[1]: value for key, value in sample_params.items() if
                   _profile_name in key}
        _profiles = []
        for idx in range(size):
            profilers[_profile_name].set_parameters(
                **{param_name: param_value[idx] for param_name, param_value in
                   _params.items()})
            _profiles.append(profilers[_profile_name]().values)
        profiles[_profile_name] = np.vstack(_profiles)
    return profiles


if __name__ == "__main__":
    pca, pca_profilers = pca_workflow(PriorManager(), ["electron_temperature", "impurity_density:ar"],
                                      np.linspace(0, 1, 30), n_components=2, num_prof_samples=int(1e2))
    # pca.plot_all()
    # plt.show(block=True)

    print()
