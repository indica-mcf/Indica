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
        psin = np.linspace(0, 1, gf.nx)
        psirz = DataArray(data=gf.psirz, coords={"R": R, "z": z}, dims=("R", "z"))
        xpsin = (psirz - gf.psi_axis) / (gf.psi_boundary - gf.psi_axis)
        _rmj = DataArray(
            R, coords={"xpsin": xpsin.interp(z=gf.zmagx).data}, dims=("xpsin",)
        ).drop_duplicates("xpsin", keep=False)
        rmjo = _rmj.where((_rmj >= gf.rmaxis), drop=True).interp(xpsin=psin)
        rmji = _rmj.where((_rmj < gf.rmaxis), drop=True).interp(xpsin=psin)
        fpol = DataArray(gf.fpol, coords={"xpsin": psin}, dims=("xpsin",))
        qpsi = gf.qpsi
        ftor = xr.zeros_like(fpol)
        for i, df in enumerate(np.diff(fpol)):
            ftor[i + 1] = ftor[i] + (0.5 * (qpsi[i] + qpsi[i + 1]) * df)
        return {
            "R": psirz.R,
            "z": psirz.z,
            "t": DataArray([t]),
            "rmjo": rmjo.interp(xpsin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "rmji": rmji.interp(xpsin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "f": fpol.interp(xpsin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "psi": psirz.T.expand_dims({"t": [t]}),
            "xpsin": DataArray(psin, coords={"xpsin": psin}),
            "psi_boundary": DataArray(gf.psi_boundary),
            "psi_axis": DataArray(gf.psi_axis),
            "ftor": ftor.interp(xpsin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "rbnd": DataArray(gf.rbdry, dims=("index",)).expand_dims({"t": [t]}),
            "zbnd": DataArray(gf.zbdry, dims=("index",)).expand_dims({"t": [t]}),
            "rmag": DataArray(gf.rmagx).expand_dims({"t": [t]}),
            "zmag": DataArray(gf.zmagx).expand_dims({"t": [t]}),
        }
