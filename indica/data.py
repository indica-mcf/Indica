"""`"Custom accessors"
<http://xarray.pydata.org/en/stable/internals.html#extending-xarray>`_,
are defined to provide additional functionality to
:py:class:`xarray.DataArray` and :py:class:`xarray.Dataset`
objects. These accessors group methods under the namespace
``indica``. E.g.::

  data_array.indica.remap_like(array2)
  data_array.indica.check_datatype(("temperature", "electron"))
  dataset.indica.check_datatype(("electron", {"T": "temperature",
                                              "n": "number_density"}))

"""

from itertools import filterfalse
from typing import Optional
from typing import Tuple

import xarray as xr

from .converters import CoordinateTransform
from .datatypes import ArrayType
from .datatypes import DatasetType
from .equilibrium import Equilibrium
from .numpy_typing import ArrayLike


@xr.register_dataarray_accessor("indica")
class InDiCAArrayAccessor:
    """Class providing additional functionality to
    :py:class:`xarray.DataArray` objects which is useful for this software.

    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def remap_like(self, other: xr.DataArray) -> xr.DataArray:
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
            t_other: Optional[xr.DataArray] = other.coords["t"]
        except ValueError:
            t_other = None
        if len(dims_other) > 0:
            x1_other: Optional[xr.DataArray] = other.coords[dims_other[0]]
        else:
            x1_other = None
        if len(dims_other) > 1:
            x2_other: Optional[xr.DataArray] = other.coords[dims_other[1]]
        else:
            x2_other = None

        self_transform: CoordinateTransform = self._obj.attrs["transform"]
        other_transform: CoordinateTransform = other.attrs["transform"]
        x1_map, x2_map, t_map = self_transform.convert_to(
            other_transform, x1_other, x2_other, t_other
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
        data_type
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
        :py:class:`~indica.converters.CoordinateTransform` object). When
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
        if "dropped" in self._obj.attrs:
            ddim = self.drop_dim
            dropped = self._obj.attrs["dropped"]
            result = self._obj.copy()
            result.loc[{ddim: dropped.coords[ddim]}] = dropped
            if "error" in self._obj.attrs:
                result.attrs["error"] = result.attrs["error"].copy()
                result.attrs["error"].loc[{ddim: dropped.coords[ddim]}] = dropped.attrs[
                    "error"
                ]
            del result.attrs["dropped"]
            return result
        else:
            return self._obj

    @property
    def drop_dim(self) -> Optional[str]:
        """The dimension, if any, which contains dropped channels."""
        if "dropped" in self._obj.attrs:
            return next(
                filterfalse(
                    lambda dim: self._obj.coords[dim].equals(
                        self._obj.attrs["dropped"].coords[dim]
                    ),
                    self._obj.dims,
                )
            )
        return None

    def ignore_data(self, labels: ArrayLike, dimension: str) -> xr.DataArray:
        """Create a copy of this array which masks the specified data.

        Parameters
        ----------
        labels
            The channel labels for which data should be ignored.
        dimension
            The name of the dimension the labels are along

        Returns
        -------
        :
            A copy of this object, but with data for the specified labels
            along the specified dimension marked as NaN.

        """
        ddim = self.drop_dim
        if ddim and ddim != dimension:
            raise ValueError(
                f"Can not not ignore data along dimension {dimension}; channels "
                f"have already been ignored in dimension {ddim}."
            )
        if ddim:
            unique_labels = list(
                filter(
                    lambda l: l not in self._obj.attrs["dropped"].coords[dimension],
                    labels,
                )
            )
        else:
            unique_labels = labels
        if len(unique_labels) == 0:
            return self._obj
        result = self._obj.copy()
        result.loc[{dimension: unique_labels}] = float("nan")
        result.attrs["dropped"] = self._obj.loc[{dimension: unique_labels}]
        if "error" in result.attrs:
            result.attrs["error"] = result.attrs["error"].copy()
        result.attrs["dropped"].attrs = {}
        if "dropped" in self._obj.attrs:
            result.attrs["dropped"] = xr.concat(
                [self._obj.attrs["dropped"], result.attrs["dropped"]], dim=dimension
            )
        if "error" in self._obj.attrs:
            result.attrs["error"].loc[{dimension: unique_labels}] = float("nan")
            result.attrs["dropped"].attrs["error"] = self._obj.attrs["error"].loc[
                {dimension: unique_labels}
            ]
            result.attrs["dropped"].attrs["error"].attrs = {}
            if "dropped" in self._obj.attrs:
                result.attrs["dropped"].attrs["error"] = xr.concat(
                    [
                        self._obj.attrs["dropped"].attrs["error"],
                        result.attrs["dropped"].attrs["error"],
                    ],
                    dim=dimension,
                )
        return result


@xr.register_dataset_accessor("indica")
class InDiCADatasetAccessor:
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
