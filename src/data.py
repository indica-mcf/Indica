"""Routines for any impurity-specific methods on the DataArray class.

"""

import xarray

@xarray.register_dataarray_accessor("impurities")
class ImpuritiesAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def remap_like(self, other: xarray.DataArray):
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

        x1_map, x2_map, t_map = self._obj.attrs["map_from_master"](
            *other.attrs["map_to_master"](x1_other, x2_other, t_other))

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
            coords_self.used.append(("t", other.coords["t"]))

        if has_x1:
            coords_map[dims_self[0]] = xarray.DataArray(x1_map,
                                                        coords=coords_self)
        if has_x2:
            coords_map[dims_self[1]] = xarray.DataArray(x2_map,
                                                        coords=coords_self)
        if has_t:
            coords_map["t"] = xarray.DataArray(t_map, coords=coords_self)

        return self._obj.interp(coords_map)
