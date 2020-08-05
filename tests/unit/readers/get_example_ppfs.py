"""Script to download some PPF data. This will be picked and used for
testing purposes."""

import itertools
import pickle

import click
import numpy as np
import sal.client
import sal.core.exception


QUANTITIES = itertools.chain(
    # Equilibrium data
    [
        f"{dda}/{dtype}"
        for dda, dtype in itertools.product(
            ["efit"],
            [
                "f",
                "faxs",
                "fbnd",
                "ftor",
                "rmji",
                "rmjo",
                "psi",
                "psir",
                "psiz",
                "vjac",
                "rbnd",
                "rmag",
                "zbnd",
                "zmag",
            ],
        )
    ],
    # Thomson scattering
    [
        f"{dda}/{dtype}"
        for dda, dtype in itertools.product(
            ["hrts", "lidr"], ["ne", "dne", "te", "dte", "z"],
        )
    ],
    ["kg10/ne", "kg10/r", "kg10/z"],
    # Electron cyclotron emissions; main data can only be retrieved after
    # inspecting kk3/gen
    ["kk3/gen"],
    # Bolometry
    ["bolo/kb5v", "bolo/kb5h"],
    # Bremsstrahlung spectroscopy
    ["ks3/zefh", "ks3/zefv", "edg7/losh", "edg7/losv"],
    # Other passive spectroscopy?
    ["xcs/cnc"],
    # Charge exchange recombination spectroscopy
    [
        f"{dda}/{dtype}"
        for dda, dtype in itertools.product(
            ["cxg6"],
            [
                "ti",
                "tihi",
                "tilo",
                "angf",
                "afhi",
                "aflo",
                "conc",
                "cohi",
                "colo",
                "rpos",
                "pos",
                "mass",
                "texp",
            ],
        )
    ],
    # Soft x-ray radiation
    [
        f"sxr/{cam}{chan:02d}"
        for cam, chan in itertools.chain.from_iterable(
            map(
                lambda x, y: zip(itertools.repeat(x), y),
                ["h", "t", "v"],
                [range(2, 18), range(1, 36), range(1, 36)],
            )
        )
    ],
)


@click.command()
@click.option("-p", "--pulse", default=90279, help="Pulse number to get data for.")
@click.option("-u", "--uid", default="jetppf", help="UID to get data for.")
@click.option(
    "--url",
    default="https://sal.jet.uk",
    help="URL of SAL server to get PPF data from.",
)
# @click.option("--username", default=lambda: os.environ.get("USER", ""),
#               show_default="current user",
#               help="Username with which to connect to PPF database.")
# @click.password_option(help="Password to connect to PPF database.")
@click.argument("output", type=click.File("wb"))
def get_example_ppfs(pulse, uid, url, output):
    """Script ot download some PPF data. This is done using the SAL
    interface. SAL Signal objects will be pickled and stored in
    file OUTPUT for later use (e.g., when testing). The pickle file
    contains a dictionary where keys are of the format "DDA/DTYPE",
    all lower-case (e.g., "lidr/ne", "cxg6/angf", etc.).

    Depending on the network from which you are accessing the SAL server, you
    may need to provide authentication.

    """
    base_path = f"/pulse/{pulse:5d}/ppf/signal/{uid}/"
    client = sal.client.SALClient(url)
    values = {}
    # Get the main quantities
    for q in QUANTITIES:
        try:
            values[q] = client.get(base_path + q)
        except sal.core.exception.NodeNotFound:
            continue
    # Get data for cyclotron emissions
    if "kk3/gen" in QUANTITIES:
        for channel in np.argwhere(values["kk3/gen"].data[15, :] > 0):
            key = f"kk3/te{channel:02d}"
            values[key] = client.get(base_path + key)
    pickle.dump(values, output)


if __name__ == "__main__":
    get_example_ppfs()
