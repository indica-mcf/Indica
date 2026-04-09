"""Sensor geometry visualisation helpers extracted from the surrogate notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    _, figures = transform.plot(return_figures=True)
    if not figures:
        figures = [plt.gcf()]
    for idx, fig in enumerate(figures, start=1):
        output_path =f"/home/jussi.hakosalo/Indica/indica/workflows/jussiphd/components/visualisations/sensor_geometry_preview_{idx}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved geometry plot to {output_path}")
    print(f"Previewed LOS geometry for pulse {pulse} ({instrument})")
    return transform
