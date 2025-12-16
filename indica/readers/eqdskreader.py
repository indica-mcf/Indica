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

    @staticmethod
    def read_geqdsk(
        filename: Path,
        cocos: int = 1,
        t: float = 0.0,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        with open(filename, "r") as f:
            gf = geqdsk.read(f, cocos=cocos, *args, **kwargs)
        R = np.linspace(gf.rleft, gf.rleft + gf.rdim, gf.nr)
        z = np.linspace(gf.zmid - (gf.zdim / 2), gf.zmid + (gf.zdim / 2), gf.nz)
        psin = np.linspace(gf.psi_axis, gf.psi_boundary, gf.nx)
        psirz = DataArray(
            data=np.where(gf.psirz != gf.psirz[0, 0], gf.psirz, np.nan),
            coords={"R": R, "z": z},
            dims=("R", "z"),
        )
        _rmj = DataArray(
            R[np.where(~np.isnan(psirz.interp(z=gf.zmagx).data))],
            coords={"psin": psirz.interp(z=gf.zmagx).dropna("R").data},
            dims=("psin",),
        )
        rmjo = _rmj.where(_rmj >= gf.rmaxis, drop=True).interp(psin=psin)
        rmji = _rmj.where(_rmj < gf.rmaxis, drop=True).interp(psin=psin)
        fpol = DataArray(gf.fpol, coords={"psin": psin}, dims=("psin",))
        qpsi = gf.qpsi
        ftor = xr.zeros_like(fpol)
        for i, df in enumerate(np.diff(fpol)):
            ftor[i + 1] = ftor[i] + (0.5 * (qpsi[i] + qpsi[i + 1]) * df)
        return {
            "R": psirz.R,
            "z": psirz.z,
            "t": DataArray([t]),
            "rmjo": rmjo.interp(psin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "rmji": rmji.interp(psin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "f": fpol.interp(psin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "psi": psirz.T.expand_dims({"t": [t]}),
            "xpsin": DataArray(
                np.linspace(0, 1, gf.nx), coords={"psin": np.linspace(0, 1, gf.nx)}
            ),
            "psi_boundary": DataArray(gf.psi_boundary),
            "psi_axis": DataArray(gf.psi_axis),
            "ftor": ftor.interp(psin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "rbnd": DataArray(gf.rbdry, dims=("index",)).expand_dims({"t": [t]}),
            "zbnd": DataArray(gf.zbdry, dims=("index",)).expand_dims({"t": [t]}),
            "rmag": DataArray(gf.rmagx).expand_dims({"t": [t]}),
            "zmag": DataArray(gf.zmagx).expand_dims({"t": [t]}),
        }
