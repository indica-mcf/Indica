# jussiphd Workflows

Practical map of the workflow stack in `workflows/jussiphd`, plus quick run instructions.

## Quick start

Run from the repository root (the directory that contains the `indica/` package):

```bash
python -m indica.workflows.jussiphd.experiments.multipulse_synthetic.flow
```

All `experiments/*/flow.py` files have a `__main__` entrypoint, so running as a module executes a default run.

To override parameters, call the flow directly in a short script:

```python
from indica.workflows.jussiphd.experiments.multipulse_synthetic.flow import (
    bolometry_inversion_multipulse_synthetic,
)

result = bolometry_inversion_multipulse_synthetic(
    generate_new_data=True,
    n_generations=1000,
    vae_n_epochs=10,
)
print(result)
```

## High-level hierarchy

```text
workflows/jussiphd/
├── components/        # Reusable building blocks (data, ML, evaluation, visualisation, filtering)
├── datasets/          # Stored CSV datasets used by multiple experiments
├── experiments/       # Flow entrypoints (`flow.py`) and per-experiment outputs/
├── flows/             # Legacy flow-export package (currently minimal)
├── plasma_profiler_init.py
├── los_bolometry_geometry.py
├── los_bolometry_radiation.py
└── visualise_losbolometry.py
```

## Components

- `components/data/`: dataset generation/loading/filtering (`data_generation.py`, `real_dataset_generation.py`, `noise_injection.py`, `quality_filtering.py`, `expanded_equilibria_generation.py`, `real_equilibrium.py`, `read_st40/`)
- `components/preprocessing/`: train/eval dataset assembly (`dataset_creation.py`)
- `components/ml/`: CVAE training/inference (`vae.py`) and saved model artifacts
- `components/evaluation/`: metrics, noise likelihood calibration, profile clustering
- `components/filtering/`: reusable dataset filter predicates
- `components/visualisations/`: training/eval/contextual comparison visualisations

## Experiment entrypoints

All entrypoints live at `experiments/<name>/flow.py`.

### Core train and eval

- `standard_multipulse`: baseline real multipulse inversion.
- `multipulse_real`: fuller real-data pipeline with quality filtering.
- `multipulse_synthetic`: baseline synthetic multipulse pipeline.
- `single_generated`: simplified single-generated synthetic variant.

### Synthetic variants and contextual studies

- `generate_synth_multipulse_splined`: generation-only splined synthetic dataset.
- `generate_synth_multipulse_clustered`: generation-only synthetic dataset sampled from real-data TE/NE anchor cluster families.
- `multipulse_synthetic_poisson_eps`: synthetic training with Poisson-like noise on `eps`.
- `multipulse_synthetic_expanded_equilibria`: expand synthetic profiles across real equilibrium contexts.
- `multipulse_synthetic_expanded_equilibria_train`: train/evaluate directly on expanded-equilibria data.
- `multipulse_synthetic_expanded_equilibria_constant_imp`: expanded-equilibria generation with fixed impurities (default C=5%, Ar=1%).
- `multipulse_synthetic_expanded_equilibria_constant_imp_compare`: VAE vs naive contextual inversion benchmark.
- `multipulse_synthetic_expanded_equilibria_constant_imp_noise_b_compare`: same contextual benchmark with noisy test brightness.
- `multipulse_synthetic_expanded_equilibria_constant_imp_noise_all_b_compare`: contextual benchmark with noise applied to the full `b` dataset.
- `multipulse_synthetic_noisy_test_b`: clean-train / noisy-test robustness flow.

### Analysis and benchmarking

- `comparison`: side-by-side real vs synthetic summary metrics.
- `inference_timing_synthetic`: inference-time benchmark.
- `vae_scaling_study`: model-size and data-size scaling study.
- `noise_matching`: synthetic-vs-real noise calibration (likelihood based).
- `noise_matching_brightness`: brightness-focused noise calibration.
- `testset_brightness_noise`: detailed clean-train vs noisy-test diagnostics.
- `eps_profile_clustering`: unsupervised profile clustering (`eps` and `b`).
- `te_ne_profile_comparison`: Te/Ne profile sampling comparison.
- `real_tene_clustering`: real TS `TE/NE` anchor-space workflow: plasma-gated read, middle-profile alignment, spline-anchor fitting, anchor clustering, per-cluster Gaussian estimation, and sampled-cluster-family visualisations.

## Useful paths

- `datasets/`: shared reference CSV datasets (including expanded-equilibria variants)
- `experiments/*/outputs/`: per-experiment figures, metrics, and summaries
- `experiments/real_tene_clustering/outputs/original_data/`: cached real `TE/NE` reads + aligned middle profiles
- `experiments/real_tene_clustering/outputs/cluster_info/`: spline-anchor vectors, fit summaries/spec, Gaussian-by-cluster summaries, sampled-family plots
- `experiments/real_tene_clustering/outputs/anchor_clusters/`: anchor cluster assignments and cluster visualisations
- `components/data/flow_data/`: generated/intermediate CSVs used by flows
- `components/ml/flow_data/`: trained model checkpoints and training metadata

## Notes

- Real-data flows (`multipulse_real`, `standard_multipulse`, `noise_matching*`) require access to ST40 data sources and compatible local configuration.
- Most flows support reuse by setting generation/training flags to `False` and pointing to existing CSV/model paths.
