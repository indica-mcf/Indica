"""Script to download some PPF data. This will be picked and used for
testing purposes."""

import itertools
import pickle

import click
import numpy as np
import sal.client
import sal.core.exception
import sal.dataclass


QUANTITIES = itertools.chain(
    # Equilibrium data
    [
        f"{instrument}/{dtype}"
        for instrument, dtype in itertools.product(
            ["efit", "eftp"],
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
        f"{instrument}/{dtype}"
        for instrument, dtype in itertools.product(
            ["hrts"],
            ["ne", "dne", "te", "dte", "z"],
        )
    ],
    [
        f"{instrument}/{dtype}"
        for instrument, dtype in itertools.product(
            ["lidr"],
            ["ne", "neu", "te", "teu", "z"],
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
        f"{instrument}/{dtype}"
        for instrument, dtype in itertools.product(
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
                [range(1, 18), range(1, 36), range(1, 36)],
            )
        )
    ],
)


@click.command()
@click.option("-p", "--pulse", default=97624, help="Pulse number to get data for.")
@click.option("-u", "--uid", default="jetppf", help="UID to get data for.")
@click.option(
    "--url",
    default="https://sal.jet.uk",
    help="URL of SAL server to get PPF data from.",
)
@click.option(
    "-c",
    "--channel-stride",
    default=1,
    help="The inverse of the fractions of channels to keep in the data.",
)
@click.option(
    "-t",
    "--max-time",
    default=100,
    help="The (rough) maximum number of increments along the time axis for which "
    "to keep data.",
)
@click.option(
    "-sp",
    "--single-precision",
    default=True,
    help="Whether to convert data to single precision floats.",
)
@click.option(
    "-s",
    "--sourcefile",
    default=None,
    type=click.File("rb"),
    help="A pickle file with which initially to populate the dictionary. Only "
    "data not already present in this file will be downloaded.",
)
@click.option(
    "-f",
    "--fake-data",
    default=True,
    help="Whether to overwrite the PPF data with random fake values. This avoids "
    "sharing proprietary PPF data.",
)
@click.argument("output", type=click.File("wb"))
def get_example_ppfs(
    pulse,
    uid,
    url,
    channel_stride,
    max_time,
    single_precision,
    sourcefile,
    fake_data,
    output,
):
    """Script ot download some PPF data. This is done using the SAL
    interface. SAL Signal objects will be pickled and stored in
    file OUTPUT for later use (e.g., when testing). The pickle file
    contains a dictionary where keys are of the format "DDA/DTYPE",
    all lower-case (e.g., "lidr/ne", "cxg6/angf", etc.).

    Optionally, the --sourcefile option may be used. If this is the
    case then pickled data will be read in from that file and used to
    initialise the dictionary. Data will only be downloaded from the
    database if not already present in the SOURCEFILE. This can be
    used, e.g., to collect data from multiple pulses to build up a
    complete set of diagnostics.

    Depending on the network from which you are accessing the SAL server, you
    may need to provide authentication.

    By default, the size of the data will be reduced by limiting the
    number of increments along the time axis and converting double
    precision floats to single precision. These settings can be
    changed using command-line flags.

    """
    base_path = f"/pulse/{pulse:5d}/ppf/signal/{uid}/"
    client = sal.client.SALClient(url)
    if sourcefile:
        values = pickle.load(sourcefile)
        if not isinstance(values, dict):
            print(f"File {sourcefile.name} does not contain valid data.")
            exit(-1)
        sourcefile.close()
    else:
        values = {}
    # Get the main quantities
    for q in itertools.filterfalse(lambda q: q in values, QUANTITIES):
        path = base_path + q
        try:
            print(f"Getting data for {path}...")
            values[q] = thin_data(
                client.get(path), channel_stride, max_time, single_precision
            )
            if fake_data:
                values[q].data = np.random.rand(*values[q].data.shape)
        except sal.core.exception.NodeNotFound:
            print("FAILED! Skipping...")
    # Get data for cyclotron emissions
    if "kk3/gen" in values:
        for channel in np.argwhere(values["kk3/gen"].data[15, :] > 0).flatten():
            key = f"kk3/te{channel + 1:02d}"
            if key in values:
                continue
            path = base_path + key
            try:
                print(f"Getting data for {path}...")
                values[key] = thin_data(
                    client.get(path), channel_stride, max_time, single_precision
                )
                if fake_data:
                    values[key].data = np.random.rand(*values[key].data.shape)
            except sal.core.exception.NodeNotFound:
                print("FAILED! Skipping...")
    pickle.dump(values, output)


def thin_data(signal, channel_stride=2, max_time=100, single_precision=True):
    """Reduces the size of some data by dropping some of the channels,
    reducing the number of time to some maximum, and/or switching to
    single precision.

    Parameters
    ----------
    signal : Signal
        The data to be reduced in size.
    channel_stride : int
        The inverse of the fraction of channels to keep.
    max_time : int
        The maximum number of points to keep along the time axis. The actual
        number may be slightly different, to ensure equal spacing.
    single_precision : bool
        Whether to reduce the data to use single precision

    """
    dtype = np.float32 if single_precision else signal.dtype
    time_dim = None
    other_dims = []
    time_pos = -1
    for i, dim in enumerate(signal.dimensions):
        if dim.temporal:
            time_dim = dim
            time_pos = i
        else:
            other_dims.append(dim)
    if time_dim:
        time_stride = max(int(len(time_dim.data) / max_time), 1)
        new_time = sal.dataclass.ArrayDimension(
            time_dim.data[::time_stride],
            dtype,
            time_dim.units,
            time_dim.error,
            time_dim.temporal,
            time_dim.description,
        )
    new_dims = []
    for dim in other_dims:
        new_dims.append(
            sal.dataclass.ArrayDimension(
                dim.data[::channel_stride],
                dtype,
                dim.units,
                dim.error,
                dim.temporal,
                dim.description,
            )
        )
    slices = [slice(None, None, channel_stride)] * signal.data.ndim
    if time_dim:
        new_dims.insert(time_pos, new_time)
        slices[time_pos] = slice(None, None, time_stride)
    new_mask = (
        sal.dataclass.ArrayStatus(
            signal.mask.status[slices], signal.mask.key, signal.mask.description
        )
        if isinstance(signal.mask, sal.dataclass.ArrayStatus)
        else signal.mask
    )
    return sal.dataclass.Signal(
        new_dims,
        signal.data[slices].astype(np.float32),
        dtype,
        signal.error,
        new_mask,
        signal.units,
        signal.description,
    )


if __name__ == "__main__":
    get_example_ppfs()
