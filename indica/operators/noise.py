from collections.abc import Callable

import numpy as np
import xarray as xr
from xarray import DataArray


def add_poisson_noise(
    data: DataArray,
    typical_counts: float = 100,
    reference: float | DataArray | None = None,
    rng: np.random.Generator | None = None,
) -> DataArray:
    """
    Add signal-dependent Poisson noise to positive DataArray values.

    Parameters
    ----------
    data : xarray.DataArray
        Clean signal. Noise is applied only where ``data > 0``.
    typical_counts : float
        Effective counts at the reference signal level. Higher
        values correspond to lower noise.
        At 100, the noise is approximately 10% of the signal at the reference level.
    reference : float, xarray.DataArray, or None
        Signal value corresponding to ``typical_counts``.
        If None, use mean(data).
    rng : np.random.Generator or None
        Optional random generator.
    """
    if not isinstance(data, DataArray):
        raise TypeError(
            "add_poisson_noise requires xarray.DataArray input. "
            f"Got {type(data)!r}."
        )

    if rng is None:
        rng = np.random.default_rng()

    if typical_counts <= 0:
        raise ValueError("typical_counts must be positive.")

    positive_mask = (data > 0).fillna(False)
    if not bool(positive_mask.any().item()):
        unchanged = data.copy()
        unchanged = unchanged.assign_attrs(data.attrs)
        unchanged.name = data.name
        return unchanged

    if reference is None:
        reference = float(data.where(positive_mask).mean(skipna=True).item())

    if isinstance(reference, DataArray):
        invalid_reference = ((reference <= 0) | ~np.isfinite(reference)).fillna(False)
        if bool((invalid_reference & positive_mask).any().item()):
            raise ValueError("reference must be positive where data > 0.")
        safe_reference = xr.where(positive_mask, reference, 1.0)
    else:
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("reference must be positive.")
        safe_reference = reference

    lam = xr.where(positive_mask, typical_counts * data / safe_reference, 0.0)
    noisy_counts = xr.apply_ufunc(rng.poisson, lam, keep_attrs=True)
    noisy = xr.where(
        positive_mask,
        noisy_counts * safe_reference / typical_counts,
        data,
    )

    # Explicitly preserve data variable attrs/name.
    noisy = noisy.assign_attrs(data.attrs)
    noisy.name = data.name
    return noisy


NOISE_MODELS: dict[str, Callable[..., DataArray]] = {
    "poisson": add_poisson_noise,
}


def get_noise_model(name: str) -> Callable[..., DataArray]:
    noise_name = name.lower()
    if noise_name not in NOISE_MODELS:
        available = ", ".join(sorted(NOISE_MODELS.keys()))
        raise ValueError(f"Unknown noise model '{name}'. Available models: {available}")
    return NOISE_MODELS[noise_name]
