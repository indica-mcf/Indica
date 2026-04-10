"""Real equilibrium loading component for jussiphd workflows."""

from __future__ import annotations

from typing import Any

from indica import Equilibrium
from indica.readers import ST40Reader


def load_real_equilibrium_from_pulse(
    pulse: int = 13622,
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
) -> Any:
    """Load equilibrium object from a real ST40 pulse EFIT signal."""
    reader = ST40Reader(
        pulse,
        tstart - dt,
        tend + dt,
        dt=dt,
        verbose=verbose,
    )
    equilibrium_data = reader.get("", "efit", 0)
    return Equilibrium(equilibrium_data)
