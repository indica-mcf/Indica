import hydra
from omegaconf import DictConfig

from indica.workflows.priors import PriorManager
from indica.workflows.priors import sample_from_priors


@hydra.main(
    version_base=None,
    config_path="../configs/workflows/priors/",
    config_name="config",
)
def example_prior_manager(cfg: DictConfig):
    pm = PriorManager(**cfg)
    post = pm.ln_prior({"electron_density.y0": 1e20, "electron_density.y1": 1e19})
    samples = sample_from_priors(
        ["electron_density.y0", "electron_density.y1"], pm.priors
    )

    print(post)
    print(samples)
