"""Implements a reader class using mocked methods for the private
reader methods."""

from collections import defaultdict
import inspect
from unittest.mock import MagicMock

from indica.readers import DataReader


def set_results(mock_name):
    """Wrap common functionality for handling default/specific data
    setting mock results. It should be used as a wrapper on a function
    which takes a default and dictionary of specific DataArrays and
    returns dicts of non-optional data, default data values, and a
    dictionary of dicts for specific data values.

    """

    def inner(func):
        def setter(self, default, specific={}):
            non_optional, def_vals, spec_vals = func(self, default, specific)
            self._set_quantities[mock_name[1:]] = defaultdict(
                lambda: default.attrs["datatype"]
            )
            for k, v in specific.items():
                self._set_quantities[mock_name[1:]][k] = v.attrs["datatype"]

            def side_effects(uid, instrument, revision, quantities):
                result = dict(non_optional)
                for q in quantities:
                    vals = spec_vals.get(q, def_vals)
                    for desc, val in vals.items():
                        key = q + "_" + desc if desc else q
                        result[key] = val
                return result

            getattr(self, mock_name).side_effect = side_effects

        return setter

    return inner


def get_vals_error_records(sample):
    """Returns a dictionary for the common pattern where there are entries
    for a quantity's values, uncertainty, and list of recrods used to
    created it.

    """
    s = sample.indica.with_ignored_data
    result = {"": s.values, "error": s.attrs["error"].values, "records": []}
    if "error" in sample.attrs:
        result["error"] = s.attrs["error"].values
    return result


def get_vals_error_records_los(sample):
    """Expands upon :py:func:`get_vals_error_records` to also provide
    start and top locations for lines of sight.

    """
    result = get_vals_error_records(sample)
    result["Rstart"] = sample.attrs["transform"].R_start.data
    result["Rstop"] = sample.attrs["transform"]._original_R_end.data
    result["zstart"] = sample.attrs["transform"].z_start.data
    result["zstop"] = sample.attrs["transform"]._original_z_end.data
    result["Tstart"] = sample.attrs["transform"].T_start.data
    result["Tstop"] = sample.attrs["transform"]._original_T_end.data
    return result


class ConcreteReader(DataReader):
    """Minimal implementation of reader class that can be instantiated."""

    @property
    def requires_authentication(self):
        return False

    def close(self):
        return


class MockReader(ConcreteReader):
    """Implementation of a reaeder which uses mocked methods for abstract
    methods. Also contains routines to reverse engineer what data the
    private reader methods need to provide in order to produce a given
    DataArray.

    """

    def __init__(
        self,
        mock_select_channels=True,
        mock_provenance=True,
        tstart=0.0,
        tend=1e10,
        max_freq=1e50,
        machine_dims=((1.83, 3.9), (-1.75, 2.0)),
    ):
        self._reader_cache_id = "mock"
        self._tstart = tstart
        self._tend = tend
        self._max_freq = max_freq
        self._machine_dims = machine_dims
        self._get_thomson_scattering = MagicMock()
        self._get_charge_exchange = MagicMock()
        self._get_equilibrium = MagicMock()
        self._get_cyclotron_emissions = MagicMock()
        self._get_radiation = MagicMock()
        self._get_bremsstrahlung_spectroscopy = MagicMock()
        self._set_quantities = {}

        def dummy_selector(k, d, c, b=[]):
            return self.drop_channels[k.split("-")[1]][k.split("-")[-1]]

        if mock_select_channels:
            self._select_channels = MagicMock()
            self.drop_channels = {}
            self._select_channels.side_effect = dummy_selector
        if mock_provenance:
            self.create_provenance = MagicMock()

    def _add_dropped_channel_data(self, diagnostic, default, specific):
        """Figure out which channels have been dropped from data and add that
        list as an option for the mock implementation of _select_channels."""
        if not hasattr(self, "drop_channels"):
            return

        def get_drop_list(data):
            if "dropped" not in data.attrs:
                channels = []
            else:
                dim = data.indica.drop_dim
                channels = data.attrs["dropped"].coords[dim].values
            return channels

        self.drop_channels[diagnostic] = defaultdict(lambda: get_drop_list(default))
        self.drop_channels[diagnostic].update(
            {k: get_drop_list(v) for k, v in specific.items()}
        )

    @set_results("_get_thomson_scattering")
    def set_thomson_scattering(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_thomson_scattering. Requested quantities will be
        constructed from ``default``, unless an alternative is
        provided for its key in the ``specific`` dict.

        """
        self._add_dropped_channel_data("thomson", default, specific)
        non_optional = {}
        non_optional["R"] = default.attrs["transform"].default_R
        non_optional["z"] = default.attrs["transform"].default_z
        non_optional["times"] = default.coords["t"].values
        non_optional["length"] = default.shape[1]
        default_vals = get_vals_error_records(default)
        specific_vals = {k: get_vals_error_records(v) for k, v in specific.items()}
        return non_optional, default_vals, specific_vals

    @set_results("_get_charge_exchange")
    def set_charge_exchange(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_charge_exchange. Requested quantities will be constructed
        from ``default``, unless an alternative is provided for its
        key in the ``specific`` dict.

        """
        self._add_dropped_channel_data("cxrs", default, specific)
        non_optional = {}
        non_optional["R"] = default.attrs["transform"].default_R
        non_optional["z"] = default.attrs["transform"].default_z
        non_optional["times"] = default.coords["t"].values
        non_optional["length"] = default.shape[1]
        non_optional["element"] = default.attrs["datatype"][1]
        non_optional["texp"] = default.attrs["texp"]
        default_vals = get_vals_error_records(default)
        specific_vals = {k: get_vals_error_records(v) for k, v in specific.items()}
        return non_optional, default_vals, specific_vals

    @set_results("_get_equilibrium")
    def set_equilibrium(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_equilibrium. Requested quantities will be constructed
        from ``default``, unless an alternative is provided for its
        key in the ``specific`` dict. ``default`` should use normalised
        poloidal flux coordinates.

        """
        non_optional = {}
        non_optional["times"] = default.coords["t"].values
        default_vals = default.values
        specific_vals = {k: {"": v.values, "records": []} for k, v in specific.items()}
        psin_intersection = {"f", "ftor", "rmji", "rmjo", "vjac"} & set(specific)
        if len(psin_intersection) > 0:
            psin = (
                specific[next(iter(psin_intersection))].coords["rho_poloidal"].values
                ** 2
            )
            non_optional["psin"] = psin
        if "psi" in specific:
            specific_vals["psi"]["r"] = specific["psi"].coords["R"].values
            specific_vals["psi"]["z"] = specific["psi"].coords["z"].values
        return non_optional, default_vals, specific_vals

    @set_results("_get_cyclotron_emissions")
    def set_cyclotron_emissions(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_cyclotron_emissions. Requested quantities will be
        constructed from ``default``, unless an alternative is
        provided for its key in the ``specific`` dict.

        """
        self._add_dropped_channel_data("cyclotron", default, specific)
        non_optional = {}
        non_optional["Btot"] = default.coords["Btot"]
        non_optional["z"] = default.attrs["transform"].default_z
        non_optional["times"] = default.coords["t"].values
        non_optional["length"] = default.shape[1]
        default_vals = get_vals_error_records(default)
        specific_vals = {k: get_vals_error_records(v) for k, v in specific.items()}
        return non_optional, default_vals, specific_vals

    @set_results("_get_radiation")
    def set_radiation(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_radiation. Requested quantities will be
        constructed from ``default``, unless an alternative is
        provided for its key in the ``specific`` dict.

        """
        self._add_dropped_channel_data("radiation", default, specific)
        non_optional = {}
        non_optional["length"] = defaultdict(
            lambda: default.shape[1], {k: v.shape[1] for k, v in specific.items()},
        )
        non_optional["machine_dims"] = self._machine_dims
        default_vals = get_vals_error_records_los(default)
        default_vals["times"] = default.coords["t"].values
        specific_vals = {}
        for k, v in specific.items():
            specific_vals[k] = get_vals_error_records_los(v)
            specific_vals[k]["times"] = v.coords["t"].values
        return non_optional, default_vals, specific_vals

    @set_results("_get_bolometry")
    def set_bolometry(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_bolometry. Requested quantities will be constructed from
        ``default``, unless an alternative is provided for its key in
        the ``specific`` dict.

        """
        self._add_dropped_channel_data("bolometry", default, specific)
        non_optional = {}
        non_optional["times"] = default.coords["t"].values
        non_optional["length"] = defaultdict(
            lambda: default.shape[1], {k: v.shape[1] for k, v in specific.items()},
        )
        default_vals = get_vals_error_records_los(default)
        specific_vals = {k: get_vals_error_records_los(v) for k, v in specific.items()}
        return non_optional, default_vals, specific_vals

    @set_results("_get_bremsstrahlung_spectroscopy")
    def set_bremsstrahlung_spectroscopy(self, default, specific={}):
        """Specify DataArrays from which to reverse engineer mock results for
        _get_bremsstrahlung_spectroscopy. Requested quantities will be
        constructed from ``default``, unless an alternative is
        provided for its key in the ``specific`` dict.

        """
        self._add_dropped_channel_data("bremsstrahlung", default, specific)
        non_optional = {}
        non_optional["times"] = default.coords["t"].values
        non_optional["machine_dims"] = self._machine_dims
        non_optional["length"] = defaultdict(
            lambda: default.shape[1], {k: v.shape[1] for k, v in specific.items()},
        )
        default_vals = get_vals_error_records_los(default)
        specific_vals = {k: get_vals_error_records_los(v) for k, v in specific.items()}
        return non_optional, default_vals, specific_vals

    def available_quantities(self, instrument):
        method = inspect.stack()[1].function
        if method in self._set_quantities:
            return self._set_quantities[method]
        elif instrument in self._IMPLEMENTATION_QUANTITIES:
            breakpoint()
            return self._IMPLEMENTATION_QUANTITIES[instrument]
        else:
            return self._AVAILABLE_QUANTITIES[method]
