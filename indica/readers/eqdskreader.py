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
        r_psin = DataArray(
            R, coords={"psin": xpsin.interp(z=gf.zmagx).data}, dims=("psin",)
        )
        r_inboard = (
            r_psin.where((r_psin >= gf.rmaxis), drop=True)
            .drop_duplicates("psin", keep="first")
            .interp(psin=psin)
        )
        r_outboard = (
            r_psin.where((r_psin < gf.rmaxis), drop=True)
            .drop_duplicates("psin", keep="last")
            .interp(psin=psin)
        )
        fpol = DataArray(gf.fpol, coords={"psin": psin}, dims=("psin",))
        qpsi = gf.qpsi
        phi = xr.zeros_like(fpol)
        for i, df in enumerate(np.diff(fpol)):
            phi[i + 1] = phi[i] + (0.5 * (qpsi[i] + qpsi[i + 1]) * df)
        return {
            "R": psirz.R,
            "z": psirz.z,
            "t": DataArray([t]),
            "R_inboard": r_inboard.interp(psin=psin).expand_dims({"t": [t]}),
            "R_outboard": r_outboard.interp(psin=psin).expand_dims({"t": [t]}),
            "f": fpol.interp(psin=psin).expand_dims({"t": [t]}),
            "psi": psirz.T.expand_dims({"t": [t]}),
            "psi_norm": DataArray(psin, coords={"psin": psin}),
            "psi_boundary": DataArray(gf.psi_boundary),
            "psi_axis": DataArray(gf.psi_axis),
            "phi": phi.interp(psin=np.linspace(0, 1, gf.nx)).expand_dims({"t": [t]}),
            "R_boundary": DataArray(gf.rbdry, dims=("index",)).expand_dims({"t": [t]}),
            "z_boundary": DataArray(gf.zbdry, dims=("index",)).expand_dims({"t": [t]}),
            "R_mag": DataArray(gf.rmagx).expand_dims({"t": [t]}),
            "z_mag": DataArray(gf.zmagx).expand_dims({"t": [t]}),
        }
