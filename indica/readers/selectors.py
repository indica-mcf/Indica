"""Callback functions for choosing which channels of data should be kept."""

from numbers import Number
from typing import Callable
from typing import Collection
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional

from matplotlib.backend_bases import PickEvent
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from xarray import DataArray

#: Callback function which can be used to choose which data should be read.
DataSelector = Callable[
    [DataArray, str, Collection[Number], Iterable[Number]], Iterable[Number]
]

MAX_TIME_SLICES = 10


class PickStack:
    def __init__(self, stack, on_pick):
        self.stack = stack
        self.ax = [artist.axes for artist in self.stack][0]
        self.on_pick = on_pick
        self.cid = self.ax.figure.canvas.mpl_connect(
            "button_press_event", self.fire_pick_event
        )

    def fire_pick_event(self, event):
        if not event.inaxes:
            return
        cont = [a for a in self.stack if a.contains(event)[0]]
        if not cont:
            return
        pick_event = PickEvent(
            "pick_Event",
            self.ax.figure.canvas,
            event,
            cont[0],
            guiEvent=event.guiEvent,
            **cont[0].contains(event)[1],
        )
        self.on_pick(pick_event)


def choose_on_plot(
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number],
    unselected_channels: Iterable[Number] = [],
) -> Iterable[Number]:
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
    print(
        "Click any channel you wish to ignore. Click a second time to "
        "include it\nagain. Ignored channels are in grey and suspicious "
        "channels are in red.\n"
    )
    print("Once finished, close the plot.\n")

    channels_to_drop = list(unselected_channels)

    # Get colour array
    original_colours = [
        "r" if label in bad_channels else "b" for label in data.coords[channel_dim]
    ]
    colours = [
        c if label not in channels_to_drop else "#ADADAD"
        for c, label in zip(original_colours, data.coords[channel_dim])
    ]

    plots: List[PathCollection] = []

    def on_pick(event):
        ind = event.ind[0]
        label = data.coords[channel_dim].data[ind]
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
        event.canvas.draw()

    if data.ndim > 2:
        raise ValueError("Received DataArray with more than 2 dimensions.")
    channel_pos = data.dims.index(channel_dim)
    if data.ndim > 1:
        other_dim: Optional[Hashable] = (
            data.dims[0] if channel_pos == 1 else data.dims[1]
        )
    else:
        other_dim = None

    if other_dim is None:
        plots.append(
            plt.scatter(data.coords[channel_dim], data, c=colours, picker=True)
        )
    else:
        # Work out which time slices to plot
        available_rows = len(data.coords[other_dim])
        stride = available_rows // MAX_TIME_SLICES
        for row in range(0, available_rows, stride):
            plots.append(
                plt.scatter(
                    data.coords[channel_dim],
                    data[{other_dim: row, channel_dim: slice(None)}],
                    c=colours,
                    picker=True,
                )
            )

    p = PickStack(plots, on_pick)
    plt.xlabel(channel_dim)
    datatype = data.attrs["datatype"]
    plt.ylabel(f"{datatype[1]} {datatype[0]}")
    plt.show()
    print("---------------------------------------------------------------------\n")
    del p
    return channels_to_drop


def use_cached_ignore_channels(
    data: DataArray,
    channel_dim: str,
    bad_channels: Collection[Number],
    unselected_channels: Iterable[Number] = [],
) -> Iterable[Number]:
    """
    Return channels from cache with no modification/input.
    """
    return list(unselected_channels)
