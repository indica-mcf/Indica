"""Sensor geometry visualisation helpers extracted from the surrogate notebook."""

from __future__ import annotations

from typing import Any

from indica import Equilibrium
from indica.readers import ST40Reader
from indica.workflows.jussiphd.los_bolometry_geometry import update_los


def preview_sensor_geometry(
    pulse: int = 13622,
    instrument: str = "blom_xy1",
    tstart: float = 0.04,
    tend: float = 0.15,
    dt: float = 0.01,
    verbose: bool = False,
) -> Any:
    """Load transform from a real pulse and plot current LOS geometry."""
    reader = ST40Reader(
        pulse,
        tstart - dt,
        tend + dt,
        dt=dt,
        verbose=verbose,
    )

    equilibrium_data = reader.get("", "efit", 0)
    equilibrium = Equilibrium(equilibrium_data)

    instrument_data = reader.get("", instrument, 0)
    quantity = next(iter(instrument_data))
    transform = instrument_data[quantity].attrs["transform"]
    transform.set_equilibrium(equilibrium, force=True)

    update_los(transform)
    transform.plot()
    print(f"Previewed LOS geometry for pulse {pulse} ({instrument})")
    return transform
