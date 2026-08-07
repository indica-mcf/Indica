from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import xarray as xr
from xarray import DataArray

PoissonSampler: TypeAlias = Callable[[DataArray], DataArray | np.ndarray]
NormalSampler: TypeAlias = Callable[[DataArray], DataArray | np.ndarray]


def add_poisson_noise(
    data: DataArray,
    typical_counts: float = 100,
    background: float | DataArray = 0,
    rng: np.random.Generator | PoissonSampler | None = None,
) -> DataArray:
    """
    Add signal-dependent Poisson noise to positive DataArray values.

    Parameters
    ----------
    data : xarray.DataArray
        Clean signal. Noise is applied only where ``data > 0``.
    typical_counts : float
        Effective number of counts at the mean positive signal level after
        background subtraction. This controls relative noise magnitude:
        standard deviation is approximately ``1 / sqrt(typical_counts)``
        at that level.
    background : float or xarray.DataArray, default 0
        Background or offset level subtracted before count scaling.
        The same background is added back after sampling.
    rng : np.random.Generator or callable, optional
        Random source for Poisson sampling. If a callable is provided, it
        must accept a DataArray of Poisson rates and return sampled counts.
    """
    if not isinstance(data, DataArray):
        raise TypeError(
            "add_poisson_noise requires xarray.DataArray input. " f"Got {type(data)!r}."
        )

    if typical_counts <= 0:
        raise ValueError("typical_counts must be positive.")

    if isinstance(rng, np.random.Generator):
        poisson_sampler: PoissonSampler = rng.poisson
    elif callable(rng):
        poisson_sampler = rng
    elif rng is None:
        poisson_sampler = np.random.default_rng().poisson
    else:
        raise TypeError("rng must be a numpy Generator, callable, or None.")

    signal = data - background
    positive_mask = (signal > 0).fillna(False)
    if not bool(positive_mask.any().item()):
        unchanged = data.copy()
        unchanged = unchanged.assign_attrs(data.attrs)
        unchanged.name = data.name
        return unchanged

    if isinstance(background, DataArray):
        invalid_background = (~np.isfinite(background)).fillna(False)
        if bool((invalid_background & positive_mask).any().item()):
            raise ValueError("background must be finite where data - background > 0.")
    else:
        if not np.isfinite(background):
            raise ValueError("background must be finite.")

    signal_scale = float(signal.where(positive_mask).mean(skipna=True).item())
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        unchanged = data.copy()
        unchanged = unchanged.assign_attrs(data.attrs)
        unchanged.name = data.name
        return unchanged

    lam = xr.where(positive_mask, typical_counts * signal / signal_scale, 0.0)
    noisy_counts = xr.apply_ufunc(poisson_sampler, lam, keep_attrs=True)
    noisy_signal = xr.where(
        positive_mask,
        noisy_counts * signal_scale / typical_counts,
        signal,
    )
    noisy = noisy_signal + background

    # Explicitly preserve data variable attrs/name.
    noisy = noisy.assign_attrs(data.attrs)
    noisy.name = data.name
    return noisy


def add_channelwise_sqrt_noise(
    data: DataArray,
    noise_scale: float = 1.0,
    rng: np.random.Generator | NormalSampler | None = None,
) -> DataArray:
    """
    Add channel-wise Gaussian noise with magnitude proportional to sqrt(signal).

    For positive values, this applies:
      noisy = data + N(0, noise_scale * sqrt(data))
    Non-positive values are left unchanged.

    Parameters
    ----------
    data : xarray.DataArray
        Clean signal to perturb.
    noise_scale : float, default 1.0
        Multiplicative factor for the noise standard deviation.
    rng : np.random.Generator or callable, optional
        Random source used to draw Gaussian samples. If a callable is provided,
        it must accept a DataArray of per-element standard deviations and return
        sampled noise values with a broadcast-compatible shape.
    """
    if not isinstance(data, DataArray):
        raise TypeError(
            "add_channelwise_sqrt_noise requires xarray.DataArray input. "
            f"Got {type(data)!r}."
        )

    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")

    if isinstance(rng, np.random.Generator):
        normal_sampler: NormalSampler = lambda sigma: rng.normal(loc=0.0, scale=sigma)
    elif callable(rng):
        normal_sampler = rng
    elif rng is None:
        _rng = np.random.default_rng()
        normal_sampler = lambda sigma: _rng.normal(loc=0.0, scale=sigma)
    else:
        raise TypeError("rng must be a numpy Generator, callable, or None.")

    positive_mask = (data > 0).fillna(False)
    if not bool(positive_mask.any().item()):
        unchanged = data.copy()
        unchanged = unchanged.assign_attrs(data.attrs)
        unchanged.name = data.name
        return unchanged

    safe_positive = xr.where(positive_mask, data, 0.0)
    sigma = noise_scale * np.sqrt(safe_positive)
    noise = xr.apply_ufunc(normal_sampler, sigma, keep_attrs=True)
    noisy = xr.where(positive_mask, data + noise, data)

    noisy = noisy.assign_attrs(data.attrs)
    noisy.name = data.name
    return noisy


NOISE_MODELS: dict[str, Callable[..., DataArray]] = {
    "channelwise_sqrt": add_channelwise_sqrt_noise,
    "poisson": add_poisson_noise,
}


def get_noise_model(name: str) -> Callable[..., DataArray]:
    noise_name = name.lower()
    if noise_name not in NOISE_MODELS:
        available = ", ".join(sorted(NOISE_MODELS.keys()))
        raise ValueError(f"Unknown noise model '{name}'. Available models: {available}")
    return NOISE_MODELS[noise_name]
