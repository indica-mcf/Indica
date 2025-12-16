from pathlib import Path
from typing import Any
from typing import Dict

from freeqdsk import geqdsk
import numpy as np
import xarray as xr
from xarray import DataArray


class EQDSKReader:
    def __init__(self):
        pass

    def read_geqdsk(self, filename: Path) -> Dict[str, Any]:
        with open(filename, "r") as f:
            gf = geqdsk.read(f)
        R = np.linspace(
            gf.rleft,
            gf.rleft + gf.rdim,
            gf.nr,
        )
        z = np.linspace(
            gf.zmid - (gf.zdim / 2),
            gf.zmid + (gf.zdim / 2),
            gf.nz,
        )
        psirz = DataArray(
            data=gf.psirz,
            coords={"R": R, "z": z},
            dims=("R", "z"),
        )
        xpsin = (psirz.interp(z=gf.zmaxis) - gf.psi_axis) / (
            gf.psi_boundary - gf.psi_axis
        )
        rmjo = DataArray(
            R[np.where(R >= gf.rmaxis)[0]],
            coords={"xpsin": xpsin.where(xpsin.R >= gf.rmaxis, drop=True).data},
            dims=("xpsin",),
        )
        rmji = DataArray(
            R[np.where(R <= gf.rmaxis)[0]],
            coords={"xpsin": xpsin.where(xpsin.R <= gf.rmaxis, drop=True).data},
            dims=("xpsin",),
        ).interp(psin=rmjo.psin)
        fpol = (
            DataArray(gf.fpol, coords={"R": R, "xpsin": xpsin}, dims=("R",))
            .interp(R=rmjo)
            .drop_vars("R")
        )
        return {
            "R": psirz.R,
            "z": psirz.z,
            "t": DataArray([0.0]),
            "rmjo": rmjo.expand_dims({"t": [0.0]}),
            "rmji": rmji.expand_dims({"t": [0.0]}),
            "f": fpol.expand_dims({"t": [0.0]}),
            "psi": psirz.expand_dims({"t": [0.0]}),
            "xpsin": xpsin.interp(R=rmjo).drop_vars("R"),
            "psi_boundary": DataArray(gf.psi_boundary),
            "psi_axis": DataArray(gf.psi_axis),
            "ftor": xr.zeros_like(fpol).expand_dims({"t": [0.0]}),
            "rbnd": DataArray(gf.rbdry, dims=("index",)),
            "zbnd": DataArray(gf.zbdry, dims=("index",)),
            "rmag": DataArray(gf.rmagx).expand_dims({"t": [0.0]}),
            "zmag": DataArray(gf.zmagx).expand_dims({"t": [0.0]}),
        }
