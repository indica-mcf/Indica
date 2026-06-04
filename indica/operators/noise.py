from collections.abc import Callable

import numpy as np


def add_poisson_noise(data, typical_counts=100, reference=None, rng=None):
    """
    Add signal-dependent Poisson noise to continuous positive data.

    Parameters
    ----------
    data : np.ndarray or xarray.DataArray
        Positive clean signal.
    typical_counts : float
        Effective counts at the reference signal level. Higher
        values correspond to lower noise.
        At 100, the noise is approximately 10% of the signal at the reference level.
    reference : float or None
        Signal value corresponding to ``typical_counts``.
        If None, use mean(data).
    rng : np.random.Generator or None
        Optional random generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    input_data = data
    data = np.asarray(input_data)

    if np.any(data < 0):
        raise ValueError("Poisson noise requires non-negative data.")

    if reference is None:
        reference = np.mean(data)

    if reference <= 0:
        raise ValueError("reference must be positive.")

    lam = typical_counts * data / reference
    noisy_counts = rng.poisson(lam)
    noisy = noisy_counts * reference / typical_counts

    # Preserve xarray metadata when available.
    if (
        hasattr(input_data, "dims")
        and hasattr(input_data, "coords")
        and hasattr(input_data, "copy")
    ):
        return input_data.copy(data=noisy)

    return noisy


NOISE_MODELS: dict[str, Callable] = {
    "poisson": add_poisson_noise,
}


def get_noise_model(name: str) -> Callable:
    noise_name = name.lower()
    if noise_name not in NOISE_MODELS:
        available = ", ".join(sorted(NOISE_MODELS.keys()))
        raise ValueError(f"Unknown noise model '{name}'. Available models: {available}")
    return NOISE_MODELS[noise_name]
