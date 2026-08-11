"""Read ST40 data for jussiphd workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from indica.readers import ST40Reader


DEFAULT_SOURCE_SIGNAL = r"\ST40::TOP.BLOM_RZ1.BEST.PROFILES:EMISSION"
PLASMA_SUMMARY_NODE = r"\ST40::TOP.SUMMARY:PLASMA"


def read_st40_instrument_data(
    pulse: int = 13622,
    instrument: str = "blom_rz1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    revision: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Read one ST40 instrument payload and return the full reader dictionary."""
    reader = ST40Reader(
        pulse,
        tstart - dt,
        tend + dt,
        dt=dt,
        verbose=verbose,
    )
    return reader.get("", instrument, revision)


def read_st40_node(
    node: str = DEFAULT_SOURCE_SIGNAL,
    pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
) -> Any:
    """Read and return raw data from an arbitrary ST40 MDS node path."""
    reader = ST40Reader(
        pulse,
        tstart - dt,
        tend + dt,
        dt=dt,
        verbose=verbose,
    )
    return reader.reader_utils.conn.get(node).data()


def pulse_has_st40_plasma(
    pulse: int,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
    plasma_node: str = PLASMA_SUMMARY_NODE,
) -> bool:
    """Return True if the pulse appears to have plasma according to summary node."""
    try:
        raw = read_st40_node(
            node=plasma_node,
            pulse=pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=verbose,
        )
    except Exception:
        return False

    arr = np.asarray(raw)
    if arr.size == 0:
        return False

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False

    # Conventionally this signal is >0 when plasma exists.
    return bool(np.nanmax(finite) > 0.0)


def read_st40_emission_signal(
    instrument: str = "blom_rz1",
    pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    revision: int = 0,
    verbose: bool = False,
    node: str | None = None,
) -> Any:
    """Read and return emission, either by raw node or reader instrument mapping."""
    if node is not None:
        return read_st40_node(
            node=node,
            pulse=pulse,
            tstart=tstart,
            tend=tend,
            dt=dt,
            verbose=verbose,
        )

    instrument_data = read_st40_instrument_data(
        pulse=pulse,
        instrument=instrument,
        tstart=tstart,
        tend=tend,
        dt=dt,
        revision=revision,
        verbose=verbose,
    )
    # Prefer emissivity-like signal names, but fall back gracefully for legacy mappings.
    if "emission" in instrument_data:
        return instrument_data["emission"]
    if "brightness" in instrument_data:
        return instrument_data["brightness"]
    if len(instrument_data) == 1:
        return next(iter(instrument_data.values()))

    available = ", ".join(sorted(instrument_data.keys()))
    raise KeyError(
        f"Expected 'emission' (or fallback 'brightness') in ST40Reader output for "
        f"instrument '{instrument}', available keys: {available}"
    )
