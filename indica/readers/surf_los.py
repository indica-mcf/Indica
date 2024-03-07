"""Routines for getting start and end points of lines of sight using data from
Surf data.

"""

from collections import defaultdict
from pathlib import Path
import re
from typing import DefaultDict
from typing import Iterable
from typing import Optional
from typing import TextIO
from typing import Tuple
from typing import Union

import numpy as np


INSTRUMENT_MAP: DefaultDict[str, Tuple[Optional[str], re.Pattern]] = defaultdict(
    lambda: (None, re.compile(".*")),
    {
        "sxr/h": ("Soft X-ray/KJ5", re.compile(".*")),
        "sxr/t": ("Soft X-ray/KJ3-4 T", re.compile(".*")),
        "sxr/v": ("Soft X-ray/KJ3-4 V", re.compile(".*")),
        "kk3": ("ECE/KK3", re.compile(".*")),
        "bolo/kb5h": ("Bolometry/KB5", re.compile(r"KB5H \d+")),
        "bolo/kb5v": ("Bolometry/KB5", re.compile(r"KB5V \d+")),
        "cwup/c_w": ("XUV-VUV spect/KT7D", re.compile(".*")),
        "cwup/n_w": ("XUV-VUV spect/KT7D", re.compile(".*")),
        "cwuv/c_w": ("XUV-VUV spect/KT7D", re.compile(".*")),
        "cwuv/n_w": ("XUV-VUV spect/KT7D", re.compile(".*")),
    },
)


_DIVIDER = re.compile(r"(?:\s+|\s*,\s*)(?=(?:[^']*'[^']*')*[^']*$)")


class SURFException(Exception):
    """Exception raised when trying to get line of sight data for an
    instrument that does not exist or a pulse number for which data is
    not available.

    """


def _parse_lines(
    data: Iterable[str], criterion: re.Pattern
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the provided lines of data to get the start and end
    coordinates of the lines of sight. This works when the data is in the
    "lines" format.

    Parameters
    ----------
    data
        An iterable returning lines from the SURF file which should be parsed
        to get line-of-sight data.
    criterion
        A regular expression against which to evaluate the name of each
        channel. The channel will only be included in result if the regular
        expression matches the channel name.

    Returns
    -------
    Rstart
        Major radius for the start of the line of sight for each channel.
    Rend
        Major radius for the end of the line of sight for each channel.
    Zstart
        Vertical position for the start of the line of sight for each channel.
    Zend
        Vertical position for the end of the line of sight for each channel.
    Tstart
        Toroidal offset of start of the line of sight for each channel.
    Tend
        Toroidal offset of the end of the line of sight for each channel.

    """
    rstart = []
    rend = []
    zstart = []
    zend = []
    for line in data:
        label, rs, zs, re, ze = _DIVIDER.split(line[:-1])
        if criterion.search(label[1:-1]):
            rstart.append(float(rs))
            rend.append(float(re))
            zstart.append(float(zs))
            zend.append(float(ze))
    return (
        np.array(rstart),
        np.array(rend),
        np.array(zstart),
        np.array(zend),
        np.zeros(len(rstart)),
        np.zeros(len(rstart)),
    )


def _parse_line3d(
    data: Iterable[str], criterion: re.Pattern
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the provided lines of data to get the start and end
    coordinates of the lines of sight. This works when the data is in the
    "line3d" format.

    Parameters
    ----------
    data
        An iterable returning lines from the SURF file which should be parsed
        to get line-of-sight data.
    criterion
        A regular expression against which to evaluate the name of each
        channel. The channel will only be included in result if the regular
        expression matches the channel name.

    Returns
    -------
    Rstart
        Major radius for the start of the line of sight for each channel.
    Rend
        Major radius for the end of the line of sight for each channel.
    Zstart
        Vertical position for the start of the line of sight for each channel.
    Zend
        Vertical position for the end of the line of sight for each channel.
    Tstart
        Toroidal offset of start of the line of sight for each channel.
    Tend
        Toroidal offset of the end of the line of sight for each channel.

    """
    rstart = []
    rend = []
    zstart = []
    zend = []
    Tstart = []
    Tend = []
    for line in data:
        label, rs, Ts, zs, re, Te, ze = _DIVIDER.split(line[:-1])
        if criterion.search(label[1:-1]):
            rstart.append(float(rs))
            rend.append(float(re))
            zstart.append(float(zs))
            zend.append(float(ze))
            Tstart.append(float(Ts))
            Tend.append(float(Te))
    return (
        np.array(rstart),
        np.array(rend),
        np.array(zstart),
        np.array(zend),
        np.array(Tstart),
        np.array(Tend),
    )


def _parse_kj34(
    data: Iterable[str], criterion: re.Pattern
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the provided lines of data to get the start and end
    coordinates of the lines of sight. This works when the data is in the
    "kj34" format.

    Parameters
    ----------
    data
        An iterable returning lines from the SURF file which should be parsed
        to get line-of-sight data.
    criterion
        A regular expression against which to evaluate the name of each
        channel. The channel will only be included in result if the regular
        expression matches the channel name.

    Returns
    -------
    Rstart
        Major radius for the start of the line of sight for each channel.
    Rend
        Major radius for the end of the line of sight for each channel.
    Zstart
        Vertical position for the start of the line of sight for each channel.
    Zend
        Vertical position for the end of the line of sight for each channel.
    Tstart
        Toroidal offset of start of the line of sight for each channel.
    Tend
        Toroidal offset of the end of the line of sight for each channel.

    """
    PIXEL_WIDTH = 0.00099
    indices = np.arange(-17, 18)
    rstart = []
    rend = []
    zstart = []
    zend = []
    for line in data:
        (
            label,
            theta_chip,
            central_index,
            focal_length,
            R_pinhole,
            z_pinhole,
            gamma,
            step,
        ) = _DIVIDER.split(line[:-1])
        if criterion.search(label[1:-1]):
            indices = np.arange(-17, 18, int(step))
            rs = np.empty(len(indices))
            rs[:] = float(R_pinhole) / 1e3
            zs = np.empty(len(indices))
            zs[:] = float(z_pinhole) / 1e3
            f = float(focal_length) / 1e3
            theta = np.radians(float(theta_chip)) + int(gamma) * np.arctan2(
                indices * PIXEL_WIDTH, f
            )
            rstart.append(rs)
            zstart.append(zs)
            rend.append(rs + np.cos(theta))
            zend.append(zs + np.sin(theta))
    rstart = np.concatenate(rstart)  # type: ignore
    rend = np.concatenate(rend)
    zstart = np.concatenate(zstart)  # type: ignore
    zend = np.concatenate(zend)
    Tstart = np.zeros_like(rstart, dtype=float)
    Tend = np.zeros_like(rend, dtype=float)
    return rstart, rend, zstart, zend, Tstart, Tend  # type: ignore


def _parse_kt7d(
    data: Iterable[str], criterion: re.Pattern
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the provided lines of data to get the start and end
    coordinates of the lines of sight.
    This works when the data is in the "kt7d" format.

    Parameters
    ----------
    data
        An iterable returning lines from the SURF file which should be parsed
        to get line-of-sight data.
    criterion
        A regular expression against which to evaluate the name of each
        channel. The channel will only be included in result if the regular
        expression matches the channel name.

    Returns
    -------
    Rstart
        Major radius for the start of the line of sight.
    Rend
        Major radius for the end of the line of sight.
    Zstart
        Vertical position for the start of the line of sight.
    Zend
        Vertical position for the end of the line of sight.
    Tstart
        Toroidal offset of start of the line of sight.
    Tend
        Toroidal offset of the end of the line of sight.

    """
    data = list(data)

    if len(data) > 1:
        raise SURFException("Only one line expected for KT7D data.")

    # expect theta to be angle (radians) of
    # line of sight from horizontal (vector along major radius)
    numbers = _DIVIDER.split(data[0].strip())[1:]
    (R_pinhole, Z_pinhole, theta_start, theta_end) = (float(x) for x in numbers)

    # for now get middle of line of sight, should account for spread instead
    theta = (theta_start + theta_end) / 2

    R_end = R_pinhole + np.cos(theta)
    Z_end = Z_pinhole + np.sin(theta)

    # assume no toroidal skew
    return (
        np.array([R_pinhole]),
        np.array([R_end]),
        np.array([Z_pinhole]),
        np.array([Z_end]),
        np.zeros(1),
        np.zeros(1),
    )


SURF_PARSERS = {
    "lines": _parse_lines,
    "line3d": _parse_line3d,
    "kj34": _parse_kj34,
    "kt7d": _parse_kt7d,
}


def read_surf_los(
    filename: Union[str, Path], pulse: int, instrument: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read beginning and ends of lines of sight from Surf data.

    Parameters
    ----------
    filename
        Name of the file containing the FLUSH data.
    pulse
        The pulse number the data is required for.
    instrument
        Which instrument to get the lines of sight for. Format within SURF file
        is inconsistent, but this routine will endeavour to support at least
        the following: SXR/H, SXR/T, SXR/V, KK3, BOLO/KB5H, BOLO/KB5V. For any
        other instrument, try a string which is present within its heading in
        the SURF file.

    Returns
    -------
    Rstart
        Major radius for the start of the line of sight for each channel.
    Rend
        Major radius for the end of the line of sight for each channel.
    Zstart
        Vertical position for the start of the line of sight for each channel.
    Zend
        Vertical position for the end of the line of sight for each channel.
    Tstart
        Toroidal offset of start of the line of sight for each channel.
    Tend
        Toroidal offset of the end of the line of sight for each channel.
    """
    with open(filename, "r", encoding="latin-1") as f:
        instrument_id, criterion = INSTRUMENT_MAP[instrument.lower()]
        if not instrument_id:
            instrument_id = instrument
        data_format, data = _get_text_block(f, pulse, instrument_id)
    return SURF_PARSERS[data_format](data, criterion)


def _get_text_block(
    file: TextIO, pulse: int, instrument_id: str
) -> Tuple[str, Iterable[str]]:
    """Extract the lines from the SURF file containing the data on lines of
    sight for the requested instrument and pulse number.

    Parameters
    ----------
    file
        A file object from which to extract the data.
    pulse
        The pulse number for which this data is desired.
    instrument_id
        The ID used for the instrument in the SURF file, obtained from
        ``INSTRUMENT_MAP``.

    Returns
    -------
    data_format
        A string indicating how the data is stored and, therefore, which
        function will need to be used to parse it.
    text_lines
        An iterable of the lines of data for the requested instrument/pulse.

    """
    for line in file:
        if line.startswith("*"):
            split = line.split("/")
            if (
                instrument_id in line[1:]
                and pulse >= int(split[-2])
                and pulse <= int(split[-1])
            ):
                break
    else:
        raise SURFException(
            f"File {file.name} has no LOS data for pulse {pulse} and instrument "
            f"{instrument_id}."
        )
    num_lines, columns, data_format = file.readline().split()
    lines = []
    for i in range(int(num_lines)):
        lines.append(file.readline())
    return data_format, lines
