import argparse
from argparse import Namespace
from getpass import getuser
from pathlib import Path

import numpy as np
import xarray as xr

from indica.workflows.pywsxp import batch, jet, plots, results


def _plots(args: Namespace) -> None:
    plots.main(savefile=args.savefile, kind=args.kind, outpath=args.output)


def _run(args: Namespace) -> None:
    if args.slice is not None:
        jet.main(
            args.configfile,
            t=args.slice,
            inputs_only=args.inputs_only,
            verbose=args.verbose,
        )
        return
    plasma = jet.main(
        args.configfile,
        inputs_only=True,
        verbose=args.verbose,
    )[1]
    if args.inputs_only:
        return
    times: list[float] = np.array(plasma.time_to_calculate, ndmin=1).tolist()
    for t in range(len(times)):
        jet.main(
            args.configfile,
            t=t,
            inputs_only=False,
            verbose=args.verbose,
        )


def _results(args: Namespace) -> None:
    results.main(args.datadir)


def _ppf(args: Namespace) -> None:
    fname = Path(args.results)
    assert fname.exists()
    data = xr.load_dataset(fname)
    seq, *_ = jet.dataset_to_ppf(data, args.dda, args.ppfuid, args.stat)
    print(f"Wrote ppf to {int(data.pulse.data)}/{args.dda}/{args.ppfuid}:{seq}")


def _batch(args: Namespace) -> None:
    batch.main(args.rundir)


def main() -> None:
    parser = argparse.ArgumentParser("pywsxp")
    subparsers = parser.add_subparsers(required=True, dest="subparser")

    parser_plot = subparsers.add_parser("plot")
    parser_plot.set_defaults(func=_plots)
    parser_plot.add_argument("savefile", type=Path)
    parser_plot.add_argument("-c", "--configfile", type=Path)
    parser_plot.add_argument("-s", "--savefile", type=Path)
    parser_plot.add_argument(
        "-k",
        "--kind",
        type=str,
        nargs="*",
        default="all",
        choices=["all"] + list(plots.kinds.keys()),
    )
    parser_plot.add_argument("-o", "--output", type=Path)
    parser_plot.add_argument("-r", "--report", action="store_true")

    parser_run = subparsers.add_parser("run")
    parser_run.set_defaults(func=_run)
    parser_run.add_argument("configfile", type=Path)
    parser_run.add_argument("-t", "--slice", type=int)
    parser_run.add_argument(
        "-b", "--inputs-only", action="store_true", help="Only build plasma"
    )
    parser_run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display fit progress and information",
    )

    parser_results = subparsers.add_parser("results")
    parser_results.set_defaults(func=_results)
    parser_results.add_argument("datadir", type=Path)

    parser_ppf = subparsers.add_parser("ppf")
    parser_ppf.set_defaults(func=_ppf)
    parser_ppf.add_argument("results", type=Path)
    parser_ppf.add_argument("-d", "--dda", type=str, default="idca")
    parser_ppf.add_argument("-u", "--ppfuid", type=str, default=getuser())
    parser_ppf.add_argument("-s", "--stat", type=int, default=0)

    parser_batch = subparsers.add_parser("batch")
    parser_batch.set_defaults(func=_batch)
    parser_batch.add_argument("rundir", type=Path)

    args = parser.parse_args()
    args.func(args)
