"""Read ST40 data for jussiphd workflows."""

from __future__ import annotations

from typing import Any

from indica.readers import ST40Reader


DEFAULT_SOURCE_SIGNAL = r"\ST40::TOP.BLOM_RZ1.BEST.PROFILES:EMISSION"


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
    if "brightness" not in instrument_data:
        available = ", ".join(sorted(instrument_data.keys()))
        raise KeyError(
            f"Expected 'brightness' in ST40Reader output for instrument '{instrument}', "
            f"available keys: {available}"
        )
    return instrument_data["brightness"]
