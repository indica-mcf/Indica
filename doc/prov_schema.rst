PROV Schema
===========

The origin and evolution of data in this software is tracked using
PROV information. While we try to use existing standards and
namespaces when creating these descriptions, it was necessary to
define the following properties.

This list will likely continue to grow for some time, as new features
are added to the code.

Calculation
    The activity of performing a calculation on some diagnostic(s).

directory
    Directory from which the software is being run.

host
    Name of the computer the software is being run on.

ignored_channels
    A list of indices for channels which were ignored when reading in
    data for an :py:class:`xarray.DataArray` object.

interval
    The spacing of times at which to save data.

method
    A type of interpolation to perform. Must be a valid value for the
    ``kind`` argument of :py:class:`scipy.interpolate.interp1d`

os
    Name of the operating system the software is being run on

ReadData
    The activity of reading data from the disk or a database into an
    :py:class:`xarray.DataArray`.

tend
    End time for range of data to be read in for a pulse.

tstart
    Start time for range of data to be read in for a pulse.
