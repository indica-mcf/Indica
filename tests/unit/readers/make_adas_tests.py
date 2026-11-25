from copy import deepcopy
from pathlib import Path

import adas
import numpy as np
import xarray as xr

from .test_adas import MockReader


def make_test_adf12(
    test_file: Path = Path("/home/adas/adas/adf12/qef93#h/qef93#h_ne10.dat"),
    get_adf12_args: tuple[str, str, str] = ("ne", "10", "93"),
):
    """Generate random data in the form of an ADF12 file.

    Uses a real ADF12 file for axes and official ADAS readers to ensure correctness in
    test file, so requires access to both of those to run.

    Parameters
    ----------
    test_file : Path
        Example (real) ADF12 file
    get_adf12_args : tuple[str, str, str]
        Arguments for `get_adf12` call

    """

    def read_reference(test_adf12: xr.DataArray, test_file: Path) -> xr.DataArray:
        ref_adf12 = xr.zeros_like(test_adf12)
        for block in ref_adf12.idx:
            print(block)
            for eb in ref_adf12.beam_energy:
                for ti in test_adf12.ion_temperature:
                    for ne in ref_adf12.electron_density:
                        for zf in test_adf12.effective_charge:
                            bt = test_adf12.total_magnetic_field.to_numpy()
                            ref_adf12.loc[block, eb, ti, ne, zf, :] = (
                                adas.read_adf12(
                                    str(test_file),
                                    block=block,
                                    ein=float(eb) + np.zeros_like(bt),
                                    tion=float(ti) + np.zeros_like(bt),
                                    dion=float(ne * 10**-6) + np.zeros_like(bt),
                                    zeff=float(zf) + np.zeros_like(bt),
                                    bmag=bt,
                                )
                                * 10**-6
                            )
        return ref_adf12

    reader = MockReader()
    reader.test_file = test_file

    # Don't need all ISEL! 3 is enough to test for a difference in reading
    test_adf12 = reader.get_adf12(*get_adf12_args).isel(idx=range(3))
    ref_adf12 = read_reference(test_adf12, reader.test_file)

    enref = 4.00e04
    teref = 5.00e03
    deref = 2.50e19
    zeref = 2.00e00
    bmref = 3.00e00

    dummy_adf12 = deepcopy(ref_adf12)
    dummy_adf12.loc[:] = np.random.random(dummy_adf12.shape) / 10**6
    nbsel = len(dummy_adf12.idx)

    fulldata = {
        "nbsel": nbsel,
        "csymb": ["NE"] * nbsel,
        "czion": [""] * nbsel,
        "cwavel": [""] * nbsel,
        "cdonor": [""] * nbsel,
        "crecvr": [""] * nbsel,
        "ctrans": [""] * nbsel,
        "cfile": [""] * nbsel,
        "ctype": [""] * nbsel,
        "cindm": [""] * nbsel,
        "qefref": dummy_adf12.sel(
            beam_energy=enref,
            ion_temperature=teref,
            electron_density=deref,
            effective_charge=zeref,
            total_magnetic_field=bmref,
        ).to_numpy(),
        "enref": [enref] * nbsel,
        "teref": [teref] * nbsel,
        "deref": [deref / 1e6] * nbsel,
        "zeref": [zeref] * nbsel,
        "bmref": [bmref] * nbsel,
        "nenera": [len(dummy_adf12.beam_energy)] * nbsel,
        "ntempa": [len(dummy_adf12.ion_temperature)] * nbsel,
        "ndensa": [len(dummy_adf12.electron_density)] * nbsel,
        "nzeffa": [len(dummy_adf12.effective_charge)] * nbsel,
        "nbmaga": [len(dummy_adf12.total_magnetic_field)] * nbsel,
        "enera": dummy_adf12.beam_energy.expand_dims({"idx": dummy_adf12.idx})
        .transpose("beam_energy", "idx")
        .to_numpy(),
        "tempa": dummy_adf12.ion_temperature.expand_dims({"idx": dummy_adf12.idx})
        .transpose("ion_temperature", "idx")
        .to_numpy(),
        "densa": dummy_adf12.electron_density.expand_dims({"idx": dummy_adf12.idx})
        .transpose("electron_density", "idx")
        .to_numpy()
        / 1e6,
        "zeffa": dummy_adf12.effective_charge.expand_dims({"idx": dummy_adf12.idx})
        .transpose("effective_charge", "idx")
        .to_numpy(),
        "bmaga": dummy_adf12.total_magnetic_field.expand_dims({"idx": dummy_adf12.idx})
        .transpose("total_magnetic_field", "idx")
        .to_numpy(),
        "qenera": dummy_adf12.sel(
            ion_temperature=teref,
            electron_density=deref,
            effective_charge=zeref,
            total_magnetic_field=bmref,
        )
        .transpose("beam_energy", "idx")
        .to_numpy(),
        "qtempa": dummy_adf12.sel(
            beam_energy=enref,
            electron_density=deref,
            effective_charge=zeref,
            total_magnetic_field=bmref,
        )
        .transpose("ion_temperature", "idx")
        .to_numpy(),
        "qdensa": dummy_adf12.sel(
            beam_energy=enref,
            ion_temperature=teref,
            effective_charge=zeref,
            total_magnetic_field=bmref,
        )
        .transpose("electron_density", "idx")
        .to_numpy(),
        "qzeffa": dummy_adf12.sel(
            beam_energy=enref,
            ion_temperature=teref,
            electron_density=deref,
            total_magnetic_field=bmref,
        )
        .transpose("effective_charge", "idx")
        .to_numpy(),
        "qbmaga": dummy_adf12.sel(
            beam_energy=enref,
            ion_temperature=teref,
            electron_density=deref,
            effective_charge=zeref,
        )
        .transpose("total_magnetic_field", "idx")
        .to_numpy(),
    }

    adas.write_adf12(
        "./test_adf12_oldstyle.dat",
        fulldata,
        comments=[
            "--------------------------",
            "For testing InDiCA readers",
            "--------------------------",
        ],
        oldstyle=True,
    )
    adas.write_adf12(
        "./test_adf12_newstyle.dat",
        fulldata,
        comments=[
            "--------------------------",
            "For testing InDiCA readers",
            "--------------------------",
        ],
        oldstyle=False,
    )

    dummy2_adf12_oldstyle = read_reference(
        test_adf12, Path("./test_adf12_oldstyle.dat")
    )
    dummy2_adf12_newstyle = read_reference(
        test_adf12, Path("./test_adf12_newstyle.dat")
    )

    np.savez(
        "./test_adf12_oldstyle.npz",
        data=dummy2_adf12_oldstyle.to_numpy().astype(float),
        **{
            str(key): val.to_numpy().astype(float)
            for key, val in dummy2_adf12_oldstyle.coords.items()
        },
    )
    np.savez(
        "./test_adf12_newstyle.npz",
        data=dummy2_adf12_newstyle.to_numpy().astype(float),
        **{
            str(key): val.to_numpy().astype(float)
            for key, val in dummy2_adf12_newstyle.coords.items()
        },
    )


def make_test_adf21(
    test_file: Path = Path("/home/adas/adas/adf21/bms97#h/bms97#h_ne10.dat"),
    get_adf21_args: tuple[str, str, str] = ("ne", "10", "97"),
) -> None:
    """Generate random data in the form of an ADF21 file.

    Uses a real ADF21 file for axes and official ADAS readers to ensure correctness in
    test file, so requires access to both of those to run.

    Parameters
    ----------
    test_file : Path
        Example (real) ADF21 file
    get_adf21_args : tuple[str, str, str]
        Arguments for `get_adf21` call

    """
    reader = MockReader()
    reader.test_file = test_file

    test_adf21 = reader.get_adf21(*get_adf21_args)
    ref_adf21 = xr.zeros_like(test_adf21)
    for ne in ref_adf21.target_density:
        for eb in ref_adf21.beam_energy:
            te = test_adf21.target_temperature.to_numpy()
            ref_adf21.loc[ne, eb, :] = (
                adas.read_adf21(
                    ["/home/adas/adas/adf21/bms97#h/bms97#h_ne10.dat"],
                    energy=float(eb) + np.zeros_like(te),
                    te=te,
                    dens=float(ne * 10**-6) + np.zeros_like(te),
                )
                * 10**-6
            )

    beref = 6.500e04
    tdref = 6.000e13
    ttref = 2.000e03

    dummy_adf21 = deepcopy(ref_adf21)
    dummy_adf21.loc[:, :, :] = np.random.random(dummy_adf21.shape) / 10**6

    reader.test_file = Path("./test_adf21.dat")

    fulldata = {
        "itz": 3,
        "tsym": "du",
        "beref": beref,
        "tdref": tdref,
        "ttref": ttref,
        "svref": float(
            dummy_adf21.sel(
                beam_energy=beref,
                target_density=tdref * 10**6,
                target_temperature=ttref,
            )
        )
        * 10**6,
        "be": dummy_adf21.beam_energy.to_numpy(),
        "tdens": dummy_adf21.target_density.to_numpy() * 10**-6,
        "ttemp": dummy_adf21.target_temperature.to_numpy(),
        "svt": dummy_adf21.sel(
            beam_energy=beref, target_density=tdref * 10**6
        ).to_numpy()
        * 10**6,
        "sved": dummy_adf21.sel(target_temperature=ttref)
        .transpose("beam_energy", "target_density")
        .to_numpy()
        * 10**6,
    }
    adas.write_adf21(
        "./test_adf21.dat",
        fulldata,
        comments=[
            "--------------------------",
            "For testing InDiCA readers",
            "--------------------------",
        ],
        project="InDiCA",
    )

    dummy2_adf21 = xr.zeros_like(test_adf21)
    for ne in dummy2_adf21.target_density:
        for eb in dummy2_adf21.beam_energy:
            for tt in dummy2_adf21.target_temperature:
                dummy2_adf21.loc[ne, eb, tt] = (
                    adas.read_adf21(
                        ["./test_adf21.dat"],
                        energy=eb,
                        te=tt,
                        dens=ne * 10**-6,
                    )[0]
                    * 10**-6
                )

    np.savez(
        "./test_adf21.npz",
        data=dummy2_adf21.to_numpy().astype(float),
        **{
            str(key): val.to_numpy().astype(float)
            for key, val in dummy2_adf21.coords.items()
        },
    )
