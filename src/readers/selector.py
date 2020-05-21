"""Callback functions for choosing which channels of data should be kept."""

from numbers import Number
from typing import Any, Callable, Container, Iterable

import matplotlib.pyplot as plt
from xarray import DataArray

#: Callback function which can be used to choose which data should be read.
DataSelector = Callable[[DataArray, str, Container[Number], Container[Number]],
                        Iterable[Number]]

MAX_TIME_SLICES = 10


def choose_on_plot(data: DataArray, channel_dim: str, bad_channels:
                   Container[Number], unselected_channels:
                   Container[Number] = []) -> Iterable[Number]:
    """Produces a plot in its own window, on which the user can click data
    from channels which they wish to be discarded.

    Instructions are printed to the command line, so it is probably
    not suitable for use in a GUI.

    Parameters
    ----------
    data:
        The data from which channels should be selected to discard.
    channel_dim:
        The name of the dimension used for storing separate channels. This
        will be used for the x-axis in the plot.
    bad_channels:
        A (possibly empty) list of channel labels which are known to be
        incorrectly calibrated, faulty, or otherwise untrustworty. These will
        be plotted in red, but must still be specifically selected by the user
        to be discared.
    unselected_channels:
        A (possibly empty) list of channels which will be ignored by default.
        These could be, e.g., cached values from a previous run. The user is
        welcome to change this.

    Returns
    -------
    :
        A list of channel labels which the user has selected to be discarded.

    """
    print("Click any channel you wish to ignore. Click a second time to "
          "include it\nagain. Ignored channels are in grey and suspicious "
          "channels are in red.\n")
    print("Once finished, close the plot.")

    channels_to_drop = list(unselected_channels)

    # Get colour array
    original_colours = ["r" if label in data.coords[channel_dim] else "b"
                        for label in bad_channels]
    colours = [c if label not in channels_to_drop else "#ADADAD" for c, label
               in zip(original_colours, data.coords[channel_dim])]

    plots = []

    def on_pick(event):
        ind = event.ind
        label = data.coords[channel_dim][ind]
        if label in channels_to_drop:
            channels_to_drop.remove(label)
            colours[ind] = original_colours[ind]
            print("Including data at {}={}".format(channel_dim, label))
        else:
            channels_to_drop.append(label)
            colours[ind] = "#ADADAD"
            print("Ignoring data at {}={}".format(channel_dim, label))
        for plot in plots:
            plot.set_color(colours)  # Does this work? Or should I use set_facecolor?
        # Not sure whether I'll need this:
        # event.canvas.draw()

    if data.ndim > 2:
        raise ValueError("Received DataArray with more than 2 dimensions.")
    channel_pos = data.dims.index(channel_dim)
    if data.ndim > 1:
        other_dim = data.dims[0] if channel_pos == 1 else data.dims[1]
    else:
        other_dim = None

    fig = plt.figure()
    if other_dim is None:
        plots.append(plt.scatter(data.coords[channel_dim], data, c=colours,
                                 picker=True))
    else:
        # Work out which time slices to plot
        available_rows = len(data.coords[other_dim])
        stride = available_rows % MAX_TIME_SLICES
        for row in range(0, available_rows, stride):
            plots.append(plt.scatter(data.coords[channel_dim],
                                     data[{other_dim: row,
                                           channel_dim: slice(None)}],
                                     c=colours, picker=True))

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()
    return channels_to_drop
