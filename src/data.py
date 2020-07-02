"""`"Custom accessors"
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_,
are defined to provide additional functionality to
:py:class:`xarray.DataArray` and :py:class:`xarray.Dataset`
objects. These accessors group methods under the namespace
``composition``. E.g.::

  data_array.composition.remap_like(array2)
  data_array.composition.check_datatype(("temperature", "electron"))
  dataset.composition.check_datatype(("electron", {"T": "temperature",
                                                   "n": "number_density"}))

"""

from typing import Iterable
from typing import Optional
from typing import Tuple

import xarray as xr

from .datatypes import ArrayType
from .datatypes import DatasetType
from .equilibrium import Equilibrium


@xr.register_dataarray_accessor("composition")
class CompositionArrayAccessor:
    """Class providing additional functionality to
    :py:class:`xarray.DataArray` objects which is useful for this software.

    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def remap_like(self, other: xr.DataArray):
        """Remap and interpolate the data in this array onto a new coordinate
        system.

        Parameters
        ----------
        other
            An array whose coordinate system will be mapped onto.

        Returns
        -------
        :
            An array representing the same data as this one, but interpolated
            so that it is on the coordinate system of ``other``.

        """
        # TODO: add more checks that a remap is possibe.
        # TODO: add some checks to catch trivial cases
        # TODO: check the mapping methods are actually present
        # TODO: add provenance, such as making result an alternate version of original
        dims_other = list(other.dims)
        try:
            dims_other.remove("t")
            t_other = other.coords["t"]
        except ValueError:
            t_other = None
        if len(dims_other) > 0:
            x1_other = other.coords[dims_other[0]]
        else:
            x1_other = None
        if len(dims_other) > 1:
            x2_other = other.coords[dims_other[1]]
        else:
            x2_other = None

        x1_map, x2_map, t_map = self._obj.attrs["transform"](
            other.attrs["transform"], x1_other, x2_other, t_other
        )

        dims_self = list(other.dims)
        coords_self = []
        coords_map = {}
        try:
            dims_self.remove("t")
            has_t = True
        except ValueError:
            has_t = False
        if len(dims_self) > 0:
            coords_self.append((dims_self[0], other.coords[dims_self[0]]))
            has_x1 = True
        else:
            has_x1 = False
        if len(dims_self) > 1:
            coords_self.append((dims_self[1], other.coords[dims_self[1]]))
            has_x2 = True
        else:
            has_x2 = False
        if has_t:
            coords_self.append(("t", other.coords["t"]))

        if has_x1:
            coords_map[dims_self[0]] = xr.DataArray(x1_map, coords=coords_self)
        if has_x2:
            coords_map[dims_self[1]] = xr.DataArray(x2_map, coords=coords_self)
        if has_t:
            coords_map["t"] = xr.DataArray(t_map, coords=coords_self)
        return self._obj.interp(coords_map)

    def check_datatype(self, data_type: ArrayType) -> Tuple[bool, Optional[str]]:
        """Checks that the dasta type of this :py:class:`xarray.DataArray`
        matches the argument.

        Parameters
        ----------
        datatype
            The datatype to check this array against.

        Returns
        -------
        status
            Whether the datatype of this array matches the argument.
        message
            If ``status == False``, an explaination of why.

        """
        pass

    @property
    def equilibrium(self) -> Equilibrium:
        """The equilibrium object currently used by this DataArray (or, more
        accurately, by its
        :py:class:`~src.converters.CoordinateTransform` object). When
        setting this porperty, ensures provenance will be updated
        accordingly.

        """
        pass

    @equilibrium.setter
    def equilibrium(self, value: Equilibrium):
        pass

    @property
    def with_ignored_data(self) -> xr.DataArray:
        """The full version of this data, including the channels which were
        dropped at read-in.

        """
        pass

    @property
    def ignored_data(self) -> xr.DataArray:
        """The data which were dropped at read-in.

        """

    def ignore_data(self, indices: Iterable[int], dimension: str) -> xr.DataArray:
        """Create a copy of this array which masks the specified data.

        Parameters
        ----------
        indices
            The indices for which data should be ignored.
        dimension
            The name of the dimension the indices are along

        Returns
        -------
        :
            A copy of this object, but with data for the specified indices
            along the specified dimension marked as NaN.

        """


@xr.register_dataset_accessor("composition")
class CompositionDatasetAccessor:
    """Class providing additional functionality to
    :py:class:`xarray.Dataset` objects which is useful for this software.

    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def attach(self, key: str, array: xr.DataArray, overwrite: bool = False):
        """Adds an additional :py:class:`xarray.DataArray` to this
        :py:class:`xarray.Dataset`. This dataset must be used for
        aggregating data with the same specific datatype (see
        :py:data:`SPECIFIC_DATATYPES`).

        It will update the metadata (datatype and provenance) to
        ensure the new DataArray is correctly included. If there is
        already an item with that key then it will raise an exception
        (unless the value is the same). This behaviour can be
        overridden with the `overwrite` argument.

        This function will fail if the specific datatyp for ``array``
        differs from that for this Dataset. It will also fail if the
        dimensions of ``array`` differ from those of the Dataset.

        Parameters
        ----------
        key
            The label which will map to the new data
        array
            The data to be added to this :py:class:`xarray.Dataset`
        overwrite
            If ``True`` and ``key`` already exists in this Dataset then
            overwrite the old value. Otherwise raise an error.
        """
        pass

    @property
    def datatype(self) -> DatasetType:
        """A structure describing the data contained within this Dataset.

        """
        pass

    def check_datatype(self, datatype: DatasetType) -> Tuple[bool, Optional[str]]:
        """Checks that the dasta type of this :py:class:`xarray.DataArray`
        matches the argument.

        This checks that all of the key/value pairs present in
        ``datatype`` match those in this Dataset. It will still return
        ``True`` even if there are _additional_ members of this
        dataset not included in ``datatype``.

        Parameters
        ----------
        datatype
            The datatype to check this Dataset against.

        Returns
        -------
        status
            Whether the datatype of this array matches the argument.
        message
            If ``status == False``, an explaination of why.

        """
        pass


def aggregate(**kwargs: xr.DataArray) -> xr.Dataset:
    """Combines the key-value pairs in ``kwargs`` into a Dataset,
    performing various checks.

    In order for this to succeed, the following must hold:

    - All arguments must have the same specific datatype (see
      :py:data:`SPECIFIC_DATATYPES`).
    - All arguments must use the same coordinate system.
    - All arguments must use the same :py:class`CoordinateTransform` object
    - All arguments need to store data on the same grid

    In addition to performing these checks, this function will create
    the correct provenance.

    """
    pass
