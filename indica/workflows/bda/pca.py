from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Tuple

from hydra import compose
from hydra import initialize_config_module
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import xarray as xr

from indica.profilers.profiler_basis import ProfilerBasis
from indica.profilers.profiler_gauss import ProfilerGauss
from indica.workflows.bda.priors import PriorCompound
from indica.workflows.bda.priors import PriorManager
from indica.workflows.bda.priors import sample_from_priors


def fit_pca(profiles: np.ndarray, ncomps, verbose=True) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=ncomps)
    pca.fit(profiles)
    pca_profiles = pca.transform(profiles)
    if verbose:
        print(f"variance per component: {pca.explained_variance_ratio_ * 100}")
        print(f"summed variance: {np.sum(pca.explained_variance_ratio_ * 100)}")
    return pca_profiles, pca


def fit_linear_prior(
    pca_profiles: np.ndarray,
    prior_values=None,
) -> LinearNDInterpolator:
    if prior_values is None:
        prior_values = np.ones(shape=pca_profiles.shape[0])

    prior_fit = LinearNDInterpolator(
        pca_profiles, prior_values, fill_value=0, rescale=False
    )
    return prior_fit


def fit_kde_prior(
    linear_fit: LinearNDInterpolator, size=int(1e6), bw_method="scott"
) -> gaussian_kde:
    sampled_points = linear_fit.points
    min = sampled_points.min(axis=0)
    max = sampled_points.max(axis=0)
    ranges = [(min[idx], max[idx]) for idx in range(sampled_points.shape[1])]

    random_points = np.array([np.random.uniform(r[0], r[1], size=size) for r in ranges])
    accepted_points = random_points[:, linear_fit(*random_points).nonzero()[0]]
    kernel = gaussian_kde(accepted_points, bw_method=bw_method)
    return kernel


def _reconstruct_profile(
    basis_function: np.ndarray,
    bias: np.ndarray,
    weights: np.ndarray,
):
    return np.dot(weights, basis_function) + bias


def _plot_principle_comps(pca, profile_name):
    plt.figure()
    plt.title(f"{profile_name}: principle components")
    for n in range(pca.components_.shape[0]):
        plt.plot(
            np.arange(pca.components_.shape[1]),
            pca.components_[
                n,
            ],
            "-",
            label=f"component {n + 1}",
        )
    plt.ylabel("A.U.")
    plt.xlabel("radial co-ordinate (-)")
    plt.legend()


def _plot_profile_fits(
    profiles: np.ndarray,
    projected_profiles: np.ndarray,
    profile_name,
    nsamples: int = 100,
):
    prof_array = xr.DataArray(
        profiles,
        dims=("index", "radial"),
        coords=(np.arange(0, profiles.shape[0]), np.linspace(0, 1, profiles.shape[1])),
    )
    proj_prof_array = xr.DataArray(
        projected_profiles,
        dims=("index", "radial"),
        coords=(
            np.arange(0, projected_profiles.shape[0]),
            np.linspace(0, 1, projected_profiles.shape[1]),
        ),
    )

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


def _plot_KDE_comparison(kernel: gaussian_kde, prior: LinearNDInterpolator, size=1000):
    plt.figure()
    plt.title("KDE Samples")
    if kernel.d == 3:
        sample = kernel.resample(size=size).T
        axs = plt.axes(projection="3d")
        axs.scatter3D(
            sample[:, 0], sample[:, 1], sample[:, 2], color="r", label="KDE Samples"
        )
        axs.scatter3D(
            prior.points[:, 0],
            prior.points[:, 1],
            prior.points[:, 2],
            color="k",
            label="Prior Samples",
        )
        plt.legend()

    if kernel.d == 2:
        xmin = prior.points[:, 0].min()
        xmax = prior.points[:, 0].max()
        ymin = prior.points[:, 1].min()
        ymax = prior.points[:, 1].max()
        X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        plt.imshow(
            np.rot90(Z),
            extent=[xmin, xmax, ymin, ymax],
            cmap=plt.cm.gist_earth_r,
        )
        plt.colorbar()
        plt.plot(
            prior.points[:, 0],
            prior.points[:, 1],
            "x",
            color="k",
            label="Prior Samples",
        )
        plt.legend()


@dataclass
class PCAProcessor:
    gaussian_profiles: Dict[str, np.ndarray]

    ncomps: int = field(default=2)
    kde_size: int = field(default=int(1e6))
    pca_weights: Dict[str, np.ndarray] = field(
        default_factory=lambda: {}
    )  # (nsamples, ncomps)
    reconstructed_profiles: Dict[str, np.ndarray] = field(
        default_factory=lambda: {}
    )  # (nsamples, xspl)
    pca_fits: Dict[str, PCA] = field(default_factory=lambda: {})

    linear_prior: Dict[str, LinearNDInterpolator] = field(default_factory=lambda: {})
    kde_prior: Dict[str, gaussian_kde] = field(default_factory=lambda: {})

    def __post_init__(self):
        """
        DO EVERYTHING. TODO: Break this up
        """

        for profile_name in self.gaussian_profiles.keys():
            print(f"PCA Profile: {profile_name}")
            self.pca_weights[profile_name], self.pca_fits[profile_name] = fit_pca(
                self.gaussian_profiles[profile_name], self.ncomps
            )
            self.reconstructed_profiles[profile_name] = _reconstruct_profile(
                self.pca_fits[profile_name].components_,
                self.pca_fits[profile_name].mean_,
                self.pca_weights[profile_name],
            )

            self.linear_prior[profile_name] = fit_linear_prior(
                self.pca_weights[profile_name]
            )
            self.kde_prior[profile_name] = fit_kde_prior(
                self.linear_prior[profile_name], size=self.kde_size
            )

    def plot_profile(self, profile_name: str):

        _plot_profile_fits(
            self.gaussian_profiles[profile_name],
            self.reconstructed_profiles[profile_name],
            profile_name,
        )
        _plot_principle_comps(self.pca_fits[profile_name], profile_name)
        _plot_KDE_comparison(
            self.kde_prior[profile_name], self.linear_prior[profile_name]
        )

    def plot_all(self):
        for name in self.gaussian_profiles.keys():
            self.plot_profile(name)
        plt.show(block=True)


def pca_workflow(
    prior_manager: PriorManager,
    opt_profiles: list,
    x_grid: xr.DataArray,
    n_components=2,
    num_prof_samples: int = int(1e5),
    kde_samples: int = int(1e6),
):
    param_names = prior_manager.get_param_names_for_profiles(opt_profiles)
    param_samples: np.ndarray = sample_from_priors(
        param_names, prior_manager.priors, size=num_prof_samples
    )
    param_samples: Dict[str, np.ndarray] = {
        param_name: param_samples[:, idx] for idx, param_name in enumerate(param_names)
    }

    profilers = {
        profile_name: ProfilerGauss(
            datatype=profile_name.split(":")[0],
            xspl=x_grid,
        )
        for profile_name in opt_profiles
    }

    profiles = sample_gauss_profiles(
        param_samples, profilers=profilers, size=num_prof_samples
    )

    pca_processor = PCAProcessor(
        gaussian_profiles=profiles, ncomps=n_components, kde_size=kde_samples
    )
    pca_processor.compound_priors = {
        f"{key}.kde": PriorCompound(
            prior_func=kernel,
            labels=tuple(f"{key}.weight_{i + 1}" for i in range(kernel.d)),
        )
        for key, kernel in pca_processor.kde_prior.items()
    }

    new_profilers = {}
    for profile_name, _profiles in profiles.items():
        _basis_func = pca_processor.pca_fits[profile_name].components_
        _bias = pca_processor.pca_fits[profile_name].mean_
        new_profilers[profile_name] = ProfilerBasis(
            basis_functions=_basis_func,
            bias=_bias,
            ncomps=n_components,
            radial_grid=x_grid,
        )
    return pca_processor, new_profilers


def sample_gauss_profiles(
    sample_params: Dict[str, np.ndarray], profilers: dict, size: int
) -> Dict[str, np.ndarray]:
    # TODO: Vectorise profilers? (only if pca too slow)
    profiles = {}
    for _profile_name, _profiler in profilers.items():

        _params = {
            key.split(".")[1]: value
            for key, value in sample_params.items()
            if _profile_name in key
        }
        _profiles = []
        for idx in range(size):
            profilers[_profile_name].set_parameters(
                **{
                    param_name: param_value[idx]
                    for param_name, param_value in _params.items()
                }
            )
            _profiles.append(profilers[_profile_name]().values)
        profiles[_profile_name] = np.vstack(_profiles)
    return profiles


def main():
    with initialize_config_module(
        version_base=None, config_module="indica.configs.workflows.priors"
    ):
        cfg = compose(config_name="config")

    pca_processor, pca_profilers = pca_workflow(
        PriorManager(
            basic_prior_info=cfg.basic_prior_info,
            cond_prior_info=cfg.cond_prior_info,
        ),
        ["ion_temperature"],
        np.linspace(0, 1, 30),
        n_components=2,
        num_prof_samples=int(5e3),
        kde_samples=int(1e5),
    )

    pca_processor.plot_all()
    plt.show(block=True)


if __name__ == "__main__":
    main()
