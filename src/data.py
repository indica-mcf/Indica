"""Routines for any impurity-specific methods on the DataArray class.

"""

from typing import Optional
from typing import Tuple

import xarray as xr

from .datatypes import ArrayType
from .datatypes import DatasetType


@xr.register_dataarray_accessor("composition")
class CompositionArrayAccessor:
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


@xr.register_dataset_accessor("composition")
class CompositionDatasetAccessor:
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
        """Returns a structure describing the data contained within this Dataset.

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
