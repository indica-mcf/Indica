from getpass import getuser
from pathlib import Path
import sys
from typing import Union

import numpy as np

LL_TEMPLATE = """# @ executable = {executable}
# @ arguments = {arguments}
# @ input = /dev/null
# @ output = {output}
# @ error = {error}
# @ initialdir = {initialdir}
# @ shell = /bin/bash
# @ notify_user = {user}
# @ class = long
# @ notification = complete
# @ queue
"""


def run_inputs(rundir: Path) -> None:
    configfile = rundir / "config.yaml"
    with open(rundir / "run_inputs.ll", "w+") as f:
        f.writelines(
            LL_TEMPLATE.format(
                executable=Path(sys.executable).parent / "pywsxp",
                arguments=f"run -b {configfile}",
                output=rundir / "run_inputs.out",
                error=rundir / "run_inputs.err",
                initialdir=rundir,
                user=getuser(),
            )
        )


def run_all(rundir: Path) -> None:
    configfile = rundir / "config.yaml"
    with open(rundir / "run_all.ll", "w+") as f:
        f.writelines(
            LL_TEMPLATE.format(
                executable=Path(sys.executable).parent / "pywsxp",
                arguments=f"run -v {configfile}",
                output=rundir / "run_all.out",
                error=rundir / "run_all.err",
                initialdir=rundir,
                user=getuser(),
            )
        )


def _run_slice_i(rundir: Path, configfile: Path, i: int) -> None:
    with open(rundir / f"run_slice{i}.ll", "w+") as f:
        f.writelines(
            LL_TEMPLATE.format(
                executable=Path(sys.executable).parent / "pywsxp",
                arguments=f"run -t {i} -v {configfile}",
                output=rundir / f"run_slice{i}.out",
                error=rundir / f"run_slice{i}.err",
                initialdir=rundir,
                user=getuser(),
            )
        )


def run_slice(rundir: Path) -> None:
    from indica.workflows.pywsxp.jet import load

    configfile = rundir / "config.yaml"
    if not (rundir / "inputs.pkl").exists():
        _run_slice_i(rundir, configfile, 0)
        return
    plasma = load(configfile)[1]
    times: list[float] = np.array(plasma.time_to_calculate, ndmin=1).tolist()
    for i in range(len(times)):
        _run_slice_i(rundir, configfile, i)


def run_plots(rundir: Path) -> None:
    resultsfile = rundir / "results.nc"
    plotdir = rundir / "plots"
    with open(rundir / "run_plots.ll", "w+") as f:
        f.writelines(
            LL_TEMPLATE.format(
                executable=Path(sys.executable).parent / "pywsxp",
                arguments=f"plot -o {plotdir} {resultsfile}",
                output=rundir / "run_plots.out",
                error=rundir / "run_plots.err",
                initialdir=rundir,
                user=getuser(),
            )
        )


def main(rundir: Union[str, Path]):
    rundir = Path(rundir).expanduser().resolve()
    rundir.mkdir(parents=True, exist_ok=True)
    run_inputs(rundir)
    run_all(rundir)
    run_slice(rundir)
    run_plots(rundir)
