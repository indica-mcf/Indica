"""Some strategies for generating artificial data on which computations can be
 performed.

"""

import datetime
from unittest.mock import MagicMock

from hypothesis.strategies import booleans
from hypothesis.strategies import composite
from hypothesis.strategies import dates
from hypothesis.strategies import dictionaries
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import just
from hypothesis.strategies import lists
from hypothesis.strategies import sampled_from
from hypothesis.strategies import text
import numpy as np
from xarray import DataArray
from xarray import Dataset

from indica.converters import FluxSurfaceCoordinates
from indica.converters import TrivialTransform
import indica.datatypes as dt
from indica.utilities import coord_array
from .converters.test_abstract_transform import coordinate_transforms
from .strategies import monotonic_series
from .strategies import noisy_functions
from .strategies import separable_functions
from .strategies import smooth_functions


@composite
def general_datatypes(draw, specific_datatype=None):
    """A strategy to select one of the general data types defined in
    :py:mod:`indica.datatypes`. If ``specific_datatype`` is present then the
    result will be compatible with that.

    """

    if specific_datatype:
        return draw(sampled_from(dt.COMPATIBLE_DATATYPES[specific_datatype]))
    else:
        return draw(sampled_from(sorted(dt.GENERAL_DATATYPES.keys())))


@composite
def specific_datatypes(draw, general_datatype=None):
    """A strategy to select one of the specific data types defined in
    :py:mod:`indica.datatypes`. If ``general_datatype`` is present then the result
    will be compatible with that.

    """

    def valid_datatype(datatype):
        if general_datatype:
            return general_datatype in dt.COMPATIBLE_DATATYPES[datatype]
        return True

    return draw(
        sampled_from(sorted(dt.SPECIFIC_DATATYPES.keys())).filter(valid_datatype)
    )


@composite
def compatible_dataset_types(draw, datatype):
    """Strategy to generate a datatype for a dataset that is compatible
    with the argument. This means the result contains no variables not
    present in ``datatype``, the specific type of all variables is the
    same as that for the dataset as a whole, and all variables have either
    the same or unconstrained general datatype."""
    result_vars = draw(
        lists(sampled_from(datatype[1]), min_size=1, unique=True).map(
            lambda keys: {k: datatype[1][k] for k in keys}
        )
    )
    return (
        datatype[0],
        {k: None if draw(booleans()) else v for k, v in result_vars.items()},
    )


@composite
def incompatible_dataset_types(draw, datatype):
    """Strategy to generate a datatype for a dataset that is incompatible
    with the argument. This means the result has a different specific
    type than that of ``datatype``, contains one or more variables not
    present in ``datatype``, or the general type of one or more
    variables does not match those in ``datatype``.

    """
    result_vars = draw(
        lists(sampled_from(datatype[1]), unique=True).map(
            lambda keys: {k: datatype[1][k] for k in keys}
        )
    )
    errors = draw(lists(integers(0, 2), min_size=1, unique=True))
    specific_type = (
        draw(specific_datatypes().filter(lambda d: d != datatype[0]))
        if 0 in errors
        else datatype[0]
    )
    general_types = {k: None if draw(booleans()) else v for k, v in result_vars.items()}
    if 1 in errors:
        change = draw(
            lists(
                booleans(), min_size=len(result_vars), max_size=len(result_vars)
            ).filter(lambda l: any(l))
        )
        for k, v in [(k, v) for (k, v), c in zip(result_vars.items(), change) if c]:
            general_types[k] = draw(
                general_datatypes(specific_type).filter(lambda d: d != v)
            )
    if 2 in errors:
        for key in draw(
            lists(
                text().filter(lambda t: t not in general_types), min_size=1, max_size=5
            )
        ):
            general_types[key] = draw(general_datatypes(specific_type))
    return specific_type, general_types


@composite
def dropped_channels(draw, size, max_dropped=0.1):
    """A strategy to generate a list of channels to drop, given the total
    number of channels.

    Parameters
    ----------
    size
        The number of channels
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.

    """
    return draw(
        lists(integers(0, size - 1), max_size=int(max_dropped * size), unique=True)
    )


@composite
def data_arrays_from_coords(
    draw,
    data_type=(None, None),
    coordinates=TrivialTransform(0.0, 0.0, 0.0, 0.0, 0.0),
    data=separable_functions(
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
    ),
    override_coords=[None, None, None],
    rel_sigma=0.02,
    abs_sigma=1e-3,
    uncertainty=True,
    max_dropped=0.1,
    require_dropped=False,
):
    """Returns a DataArray which uses the given coordinate transform.

    Parameters
    ----------
    data_type : Tuple[str, str]
        The data type of the data_array to be generated. If either element of
        the tuple is ``None`` then that element will be drawn from a strategy.
    coordinates
        A coordinate transform to use for this data.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates. Note that all coordinates will be
        normalised before being passed to this function.
    override_coords
        If item is not None, use those coordinates rather than the defaults
        from the coordinate transform. Should be ordered ``[x1, x2, t]``.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.
    require_dropped
        If True, ensure at least one channel is dropped.

    """

    general_type = (
        data_type[0] if data_type[0] else draw(general_datatypes(data_type[1]))
    )
    specific_type = (
        data_type[1] if data_type[1] else draw(specific_datatypes(general_type))
    )

    x1 = coordinates.default_x1 if override_coords[0] is None else override_coords[0]
    x2 = coordinates.default_x2 if override_coords[1] is None else override_coords[1]
    t = coordinates.default_t if override_coords[2] is None else override_coords[2]
    func = (
        draw(noisy_functions(draw(data), rel_sigma, abs_sigma))
        if rel_sigma or abs_sigma
        else draw(data)
    )
    coords = [
        (c[0], c[1].flatten() if isinstance(c[1], np.ndarray) else c[1])
        for c in [("x1", x1), ("x2", x2), ("t", t)]
        if isinstance(c[1], (np.ndarray, DataArray)) and c[1].ndim > 0
    ]
    shape = tuple(len(c) for _, c in coords)
    if isinstance(x1, (np.ndarray, DataArray)) and x1.ndim > 0:
        min_val = np.min(x1)
        width = np.abs(np.max(x1) - min_val)
        x1_scaled = (x1 - min_val) / (width if width else 1.0)
    else:
        x1_scaled = 0.0
    if isinstance(x2, (np.ndarray, DataArray)) and x2.ndim > 0:
        min_val = np.min(x2)
        width = np.abs(np.max(x2) - min_val)
        x2_scaled = (x2 - min_val) / (width if width else 1.0)
    else:
        x2_scaled = 0.0
    if isinstance(t, (np.ndarray, DataArray)) and t.ndim > 0:
        min_val = np.min(t)
        width = np.abs(np.max(t) - min_val)
        t_scaled = (t - min_val) / (width if width else 1.0)
    else:
        t_scaled = 0.0
    tmp = func(x1_scaled, x2_scaled, t_scaled)
    result = DataArray(np.reshape(tmp, shape), coords=coords)
    if isinstance(x1, np.ndarray):
        flat_x1 = x1.flatten()
    else:
        flat_x1 = x1
    dropped = (
        [flat_x1[i] for i in draw(dropped_channels(len(x1), max_dropped))]
        if isinstance(x1, np.ndarray)
        else []
    )
    if require_dropped and len(dropped) == 0:
        dropped = [flat_x1[0]]
    if uncertainty and (rel_sigma or abs_sigma):
        error = rel_sigma * result + abs_sigma
        result.attrs["error"] = error
    if dropped and flat_x1[0] != flat_x1[-1]:
        to_keep = np.logical_not(
            DataArray(flat_x1, coords=[("x1", flat_x1)]).isin(dropped)
        )
        dropped_result = result.sel(x1=dropped)
        result = result.where(to_keep)
        if uncertainty and (rel_sigma or abs_sigma):
            dropped_result.attrs["error"] = result.attrs["error"].sel(x1=dropped)
            result.attrs["error"] = result.attrs["error"].where(to_keep)
        result.attrs["dropped"] = dropped_result
    result.attrs["datatype"] = (general_type, specific_type)
    result.attrs["provenance"] = MagicMock()
    result.attrs["partial_provenance"] = MagicMock()
    result.attrs["transform"] = coordinates
    if hasattr(coordinates, "equilibrium"):
        result.indica.equilibrium = coordinates.equilibrium
    return result


@composite
def data_arrays(
    draw,
    data_type=(None, None),
    coordinates=coordinate_transforms(((1.83, 3.9), (-1.75, 2.0), (50.0, 120.0)), 4, 3),
    data=separable_functions(
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
    ),
    override_coords=[None, None, None],
    rel_sigma=0.02,
    abs_sigma=1e-3,
    uncertainty=True,
    max_dropped=0.1,
    require_dropped=False,
):
    """Returns a DataArray, with appropriate metadata for use in testing
    calculations.

    Parameters
    ----------
    data_type : Tuple[str, str]
        The data type of the data_array to be generated. If either element of
        the tuple is ``None`` then that element will be drawn from a strategy.
    coordinates
        A strategy for generating :py:class:`indica.converters.CoordinateTransform`
        objects. If absent, any type of transform could be used.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates.
    override_coords
        If item is not None, use those coordinates rather than the defaults
        from the coordinate transform. Should be ordered ``[x1, x2, t]``.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.
    require_dropped
        If True, ensure at least one channel is dropped.

    """
    transform = draw(coordinates)
    return draw(
        data_arrays_from_coords(
            data_type,
            transform,
            data,
            override_coords,
            rel_sigma,
            abs_sigma,
            uncertainty,
            max_dropped,
            require_dropped,
        )
    )


@composite
def array_dictionaries(
    draw,
    coordinates,
    options,
    data=separable_functions(
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
    ),
    override_coords=[None, None, None],
    rel_sigma=0.02,
    abs_sigma=1e-3,
    uncertainty=True,
    max_dropped=0.1,
    min_size=1,
    require_dropped=False,
):
    """Create a dictionary of DataArrays, all with the same coordinate
    transform, with keys selected from those in ``options``.

    Parameters
    ----------
    coordinates
        The coordinate transform to use for this data.
    options
        A dictionary where keys are those which may be present in the result
        and values are the datatype of the array associated with that key in
        the result.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates.
    override_coords
        If item is not None, use those coordinates rather than the defaults
        from the coordinate transform. Should be ordered ``[x1, x2, t]``.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.
    min_size
        The minumum number of DataArrays in the dictionary. Must not be larger
        than ``options``.
    require_dropped
        If True, ensure at least one channel is dropped.

    """
    result = {}
    keys = draw(
        lists(sampled_from(sorted(options.keys())), unique=True, min_size=min_size)
    )
    for key in keys:
        result[key] = draw(
            data_arrays_from_coords(
                options[key],
                coordinates,
                data,
                override_coords,
                rel_sigma,
                abs_sigma,
                uncertainty,
                max_dropped,
                require_dropped,
            )
        )
    return result


@composite
def datasets(
    draw,
    data_type=(None, {}),
    coordinates=coordinate_transforms(((1.83, 3.9), (-1.75, 2.0), (50.0, 120.0)), 4, 3),
    data=separable_functions(
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
        smooth_functions(max_val=1e3),
    ),
    rel_sigma=0.02,
    abs_sigma=1e-3,
    uncertainty=True,
    max_dropped=0.1,
    require_dropped=False,
):
    """Returns a Dataset, with appropriate metadata for use in testing
    calculations.

    Parameters
    ----------
    data_type : Tuple[str, Dict[str, str]]
        The data type of the dataset to be generated. If the first element or
        any value in the dictionary is None then it will be drawn from a
        strategy. If the dictionary is empty then its contents will be drawn
        from a strategy.
    coordinates
        A strategy for generating :py:class:`indica.converters.CoordinateTransform`
        objects. If absent, any type of transform could be used.
    data
        A strategy to generate functions which calculate the contents of the
        DataArray from the coordinates.
    rel_sigma
        Standard deviation of relative noise applied to the data
    abs_sigma
        Standard deviation of absolute noise applied to the data
    uncertainty
        If ``True``, generate uncertainty metadata using ``rel_sigma`` and
        ``abs_sigma`` (if they are non-zero).
    max_dropped
        The maximum number of channels to drop, as a fraction of the total
        number of channels.
    require_dropped
        If True, ensure at least one channel is dropped.

    """

    def compatible(specific):
        return all(
            [
                (general in dt.COMPATIBLE_DATATYPES[specific])
                for general in data_type[1].values()
                if general
            ]
        )

    specific_type = (
        data_type[0] if data_type[0] else draw(specific_datatypes().filter(compatible))
    )
    if data_type[1]:
        general_type = {
            k: (v if v else draw(general_datatypes(specific_type)))
            for k, v in data_type[1].items()
        }
    else:
        general_type = draw(
            dictionaries(
                text(), general_datatypes(specific_type), min_size=1, max_size=5
            )
        )
    transform = draw(coordinates)
    data = {}
    for key, gtype in general_type.items():
        data[key] = draw(
            data_arrays(
                (gtype, specific_type),
                just(transform),
                rel_sigma=rel_sigma,
                abs_sigma=abs_sigma,
                uncertainty=uncertainty,
                max_dropped=max_dropped,
                require_dropped=require_dropped,
            )
        )
    return Dataset(
        data,
        attrs={"datatype": (specific_type, general_type), "provenance": MagicMock()},
    )


@composite
def equilibrium_data(
    draw,
    machine_dims=((1.83, 3.9), (-1.75, 2.0)),
    min_spatial_points=3,
    max_spatial_points=12,
    min_time_points=2,
    max_time_points=15,
    start_time=75.0,
    end_time=80.0,
    Btot_factor=None,
):
    """Returns a dictionary containing the data necessary to construct an
    :py:class:`indica.equilibrium.Equilibrium` object.

    Parameters
    ----------
    machine_dims
        The size of the reactor, ((Rmin, Rmax), (zmin, zmax))
    min_spatial_points
        The minimum number of points to use for spatial coordinate axes
    max_spatial_points
        The maximum number of points to use for spatial coordinate axes
    min_time_points
        The minimum number of points to use for the time axis
    max_time_points
        The maximum number of points to use for the time axis
    Btot_factor
        If present, the equilibrium will have total magnetic field strength
        Btot_factor/R.
    """
    result = {}
    nspace = draw(integers(min_spatial_points, max_spatial_points))
    ntime = draw(integers(min_time_points, max_time_points))
    times = np.linspace(start_time - 0.5, end_time + 0.5, ntime)
    tfuncs = smooth_functions((start_time, end_time), 0.01)
    r_centre = (machine_dims[0][0] + machine_dims[0][1]) / 2
    z_centre = (machine_dims[1][0] + machine_dims[1][1]) / 2
    raw_result = {}
    attrs = {
        "transform": TrivialTransform(0.0, 0.0, 0.0, 0.0, 0.0),
        "provenance": MagicMock(),
        "partial_provenance": MagicMock(),
    }
    result["rmag"] = DataArray(
        r_centre + draw(tfuncs)(times), coords=[("t", times)], name="rmag", attrs=attrs
    )
    result["rmag"].attrs["datatype"] = ("major_rad", "mag_axis")
    result["zmag"] = DataArray(
        z_centre + draw(tfuncs)(times), coords=[("t", times)], name="zmag", attrs=attrs
    )
    result["zmag"].attrs["datatype"] = ("z", "mag_axis")
    fmin = draw(floats(0.0, 1.0))
    result["faxs"] = DataArray(
        fmin + np.abs(draw(tfuncs)(times)),
        coords=[("t", times)],
        name="faxs",
        attrs=attrs,
    )
    result["faxs"].attrs["datatype"] = ("magnetic_flux", "mag_axis")
    a_coeff = DataArray(
        np.vectorize(lambda x: draw(floats(0.75 * x, x)))(
            np.minimum(
                np.abs(machine_dims[0][0] - result["rmag"]),
                np.abs(machine_dims[0][1] - result["rmag"]),
            )
        ),
        coords=[("t", times)],
    )
    if Btot_factor is None:
        b_coeff = DataArray(
            np.vectorize(lambda x: draw(floats(0.75 * x, x)))(
                np.minimum(
                    np.abs(machine_dims[1][0] - result["zmag"].data),
                    np.abs(machine_dims[1][1] - result["zmag"].data),
                ),
            ),
            coords=[("t", times)],
        )
        n_exp = 0.5  # 1 # draw(floats(-0.5, 1.0))
        fmax = draw(floats(max(1.0, 2 * fmin), 10.0))
        result["fbnd"] = DataArray(
            fmax - np.abs(draw(tfuncs)(times)),
            coords=[("t", times)],
            name="fbnd",
            attrs=attrs,
        )
    else:
        b_coeff = a_coeff
        n_exp = 1
        fdiff_max = Btot_factor * a_coeff
        result["fbnd"] = DataArray(
            np.vectorize(lambda axs, diff: axs + draw(floats(0.001 * diff, diff)))(
                result["faxs"], fdiff_max.values
            ),
            coords=[("t", times)],
            name="fbnd",
            attrs=attrs,
        )
    result["fbnd"].attrs["datatype"] = ("magnetic_flux", "separatrix")
    thetas = DataArray(
        np.linspace(0.0, 2 * np.pi, nspace, endpoint=False), dims=["arbitrary_index"]
    )
    result["rbnd"] = (
        result["rmag"]
        + a_coeff * b_coeff / np.sqrt(a_coeff ** 2 * np.tan(thetas) ** 2 + b_coeff ** 2)
    ).assign_attrs(**attrs)
    result["rbnd"].name = "rbnd"
    result["rbnd"].attrs["datatype"] = ("major_rad", "separatrix")
    result["zbnd"] = (
        result["zmag"]
        + a_coeff
        * b_coeff
        / np.sqrt(a_coeff ** 2 + b_coeff ** 2 * np.tan(thetas) ** -2)
    ).assign_attrs(**attrs)
    result["zbnd"].name = "zbnd"
    result["zbnd"].attrs["datatype"] = ("z", "separatrix")

    r = np.linspace(machine_dims[0][0], machine_dims[0][1], nspace)
    z = np.linspace(machine_dims[1][0], machine_dims[1][1], nspace)
    rgrid = DataArray(r, coords=[("R", r)])
    zgrid = DataArray(z, coords=[("z", z)])
    psin = (
        (-result["zmag"] + zgrid) ** 2 / b_coeff ** 2
        + (-result["rmag"] + rgrid) ** 2 / a_coeff ** 2
    ) ** (0.5 / n_exp)
    psi = psin * (result["fbnd"] - result["faxs"]) + result["faxs"]
    psi.name = "psi"
    psi.attrs["transform"] = attrs["transform"]
    psi.attrs["provenance"] = MagicMock()
    psi.attrs["partial_provenance"] = MagicMock()
    psi.attrs["datatype"] = ("magnetic_flux", "plasma")
    result["psi"] = psi

    psin_coords = np.linspace(0.0, 1.0, nspace)
    rho = np.sqrt(psin_coords)
    psin_data = DataArray(psin_coords, coords=[("rho_poloidal", rho)])
    attrs["transform"] = FluxSurfaceCoordinates(
        "poloidal",
        coord_array(rho, "rho_poloidal"),
        DataArray(0.0),
        DataArray(0.0),
        DataArray(0.0),
        coord_array(times, "t"),
    )
    ftor_min = draw(floats(0.0, 1.0))
    ftor_max = draw(floats(max(1.0, 2 * fmin), 10.0))
    result["ftor"] = DataArray(
        np.outer(
            1 + draw(tfuncs)(times), draw(monotonic_series(ftor_min, ftor_max, nspace))
        ),
        coords=[("t", times), ("rho_poloidal", rho)],
        name="ftor",
        attrs=attrs,
    )
    result["ftor"].attrs["datatype"] = ("toroidal_flux", "plasma")
    if Btot_factor is None:
        f_min = draw(floats(0.0, 1.0))
        f_max = draw(floats(max(1.0, 2 * fmin), 10.0))
        time_vals = draw(tfuncs)(times)
        space_vals = draw(monotonic_series(f_min, f_max, nspace))
        f_raw = np.outer(abs(1 + time_vals), space_vals)
    else:
        f_raw = np.outer(
            np.sqrt(
                Btot_factor ** 2
                - (raw_result["fbnd"] - raw_result["faxs"]) ** 2 / a_coeff ** 2
            ),
            np.ones_like(rho),
        )
        f_raw[:, 0] = Btot_factor
    result["f"] = DataArray(
        f_raw, coords=[("t", times), ("rho_poloidal", rho)], name="f", attrs=attrs
    )
    result["f"].attrs["datatype"] = ("f_value", "plasma")
    result["rmjo"] = (result["rmag"] + a_coeff * psin_data ** n_exp).assign_attrs(
        **attrs
    )
    result["rmjo"].name = "rmjo"
    result["rmjo"].attrs["datatype"] = ("major_rad", "lfs")
    result["rmji"] = (result["rmag"] - a_coeff * psin_data ** n_exp).assign_attrs(
        **attrs
    )
    result["rmji"].name = "rmji"
    result["rmji"].attrs["datatype"] = ("major_rad", "hfs")
    result["vjac"] = (
        4
        * n_exp
        * np.pi ** 2
        * result["rmag"]
        * a_coeff
        * b_coeff
        * psin_data ** (2 * n_exp - 1)
    ).assign_attrs(**attrs)
    result["vjac"].name = "vjac"
    result["vjac"].attrs["datatype"] = ("volume_jacobian", "plasma")
    return result


@composite
def adf11_data(
    draw,
    min_num_densities=3,
    max_num_densities=8,
    min_num_temps=5,
    max_num_temps=15,
    max_z=74,
    quantities=["scd", "acd", "plt", "prb"],
):
    """Generates fake ADF11 data.

    Parameters
    ----------
    min_num_densities
        The minimum number of densities to use.
    max_num_densities
        The maximum number of densities to use.
    min_num_temps
        The minimum number of temperatures to use.
    max_num_temps
        The maximum number of temperatures to use.
    max_z
        The maximum atomic number to use.
    quantities
        The types of quantities from which to select.

    """
    nd = draw(integers(min_num_densities, max_num_densities))
    nt = draw(integers(min_num_temps, max_num_temps))
    min_dens = draw(integers(3, 12))
    max_dens = draw(integers(min_dens + 1, 17))
    densities = DataArray(
        np.linspace(min_dens, max_dens, nd) + 6, dims="log_electron_density"
    )
    min_temp = draw(floats(-2.0, 2.5))
    max_temp = draw(floats(min_temp + 1, 6.0))
    temperatures = DataArray(
        np.linspace(min_temp, max_temp, nt), dims="log_electron_temperature"
    )
    z = draw(integers(1, 74))
    min_z = draw(integers(1, z))
    max_z = draw(integers(min_z, z))
    ion_states = DataArray(np.arange(min_z - 1, max_z, dtype=int), dims="ion_charges")
    func = draw(
        separable_functions(
            smooth_functions((min_z, max_z + 1), max_val=0.1),
            smooth_functions((min_temp, max_temp), max_val=0.1),
            smooth_functions((min_dens, max_dens), max_val=0.1),
        )
    )
    data = np.clip(func(ion_states, temperatures, densities) - 6, -99, 99)
    result = DataArray(data, coords=[ion_states, temperatures, densities])
    q = draw(sampled_from(quantities))
    result.attrs["datatype"] = (dt.ADF11_GENERAL_DATATYPES[q], dt.ORDERED_ELEMENTS[z])
    result.attrs["provenance"] = MagicMock()
    result.attrs["date"] = draw(
        dates(datetime.date(1940, 1, 1), datetime.date(2020, 12, 31))
    )
    result.name = f"log_{dt.ORDERED_ELEMENTS[z]}_{dt.ADF11_GENERAL_DATATYPES[q]}"
    return result
