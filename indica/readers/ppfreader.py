"""Provides implementation of :py:class:`readers.DataReader` for
reading PPF data produced by JET.

"""

import socket
from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple

from sal.client import SALClient
from sal.core.exception import AuthenticationFailed
from sal.dataclass import Signal

from .abstractreader import DataReader
from .abstractreader import DataSelector
from .selectors import choose_on_plot
from ..session import global_session
from ..session import Session


class PPFReader(DataReader):
    """Class to read JET PPF data using SAL.

    Currently the following types of data are supported.

    ==========  ==================  ================  =================
    Key         Data type           Data for          Instrument
    ==========  ==================  ================  =================
    cxg6_angf   Angular frequency   Pulse dependant   CXG6
    cxg6_conc   Concentration       Pulse dependant   GXG6
    cxg6_ti     Temperature         Pulse dependant   CXG6
    efit_rmag   Major radius        Magnetic axis     EFIT equilibrium
    efit_zmag   Z position          Magnetic axis     EFIT equilibrium
    efit_rsep   Major radius        Separatrix        EFIT equilibrium
    efit_zsep   Z position          Separatrix        EFIT equilibrium
    hrts_ne     Number density      Electrons         HRTS
    hrts_te     Temperature         Electrons         HRTS
    kk3_te      Temperature         Electrons         KK3
    lidr_ne     Number density      Electrons         LIDR
    lidr_te     Temperature         Electrons         LIDR
    sxr_h       Luminous flux       Soft X-rays       SXR camera H
    sxr_t       Luminous flux       Soft X-rays       SXR camera T
    sxr_v       Luminous flux       Soft X-rays       SXR camera V
    ==========  ==================  ================  =================

    Note that there will need to be some refactoring to support other
    data types. However, **this is guaranteed not to affect the public
    interface**.

    Parameters
    ----------
    times : np.ndarray
        An ordered array of times to which data will be
        downsampled/interpolated.
    pulse : int
        The ID number for the pulse from which to get data.
    uid : str
        The UID for the particular data to be read.
    server : str
        The URL for the SAL server to read data from.
    default_error : float
        Relative uncertainty to use for diagnostics which do not provide a
        value themselves.
    sess : Session
        An object representing the session being run. Contains information
        such as provenance data.

    Attributes
    ----------
    DIAGNOSTIC_QUANTITIES: Dict[str, Dict[str, Dict[str, Dict[str, ArrayType]]]]
        Hierarchical information on the quantities which are available for
        reading. These are indexed by (in order) diagnostic name, UID,
        instrument name, and quantity name. The values of the innermost
        dictionary describe the physical type of the data to be read.
    NAMESPACE: Tuple[str, str]
        The abbreviation and full URL for the PROV namespace of the reader
        class.

    """

    DDA_METHODS = {
        "hrts": "get_thomson_scattering",
        "lidr": "get_thomson_scattering",
        "efit": "get_equilibrium",
        "eftp": "get_equilibrium",
        "kk3": "get_cyclotron_emissions",
        "cxg6": "get_charge_exchange",
        "ks3": "get_bremsstrahlung_spectroscopy",
        "sxr": "get_radiation",
        "bolo": "get_radiation",
        "kg10": "get_thomson_scattering",
    }
    _IMPLEMENTATION_QUANTITIES = {
        "kg10": {"ne": ("number_density", "electron")},
        "sxr": {
            "h": ("luminous_flux", "sxr"),
            "t": ("luminous_flux", "sxr"),
            "v": ("luminous_flux", "sxr"),
        },
        "bolo": {
            "kb5h": ("luminous_flux", "bolometric"),
            "kb5v": ("luminous_flux", "bolometric"),
        },
        "ks3": {
            "zefh": ("effective_charge", "plasma"),
            "zefv": ("effective_charge", "plasma"),
        },
    }

    #    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {
    #        "cxg6_angf": ("angular_freq", None),
    #        "cxg6_conc": ("concentration", None),
    #        "cxg6_ti": ("temperature", None),
    #        "efit_rmag": ("major_rad", "mag_axis"),
    #        "efit_zmag": ("z", "mag_axis"),
    #        "efit_rsep": ("major_rad", "separatrix_axis"),
    #        "efit_zsep": ("z", "separatrix_axis"),
    #        "hrts_ne": ("number_density", "electrons"),
    #        "hrts_te": ("temperature", "electrons"),
    #        "kk3_te": ("temperature", "electrons"),
    #        "lidr_ne": ("number_density", "electrons"),
    #        "lidr_te": ("temperature", "electrons"),
    #        "sxr_h": ("luminous_flux", "sxr"),
    #        "sxr_t": ("luminous_flux", "sxr"),
    #        "sxr_v": ("luminous_flux", "sxr"),
    #    }

    _SXR_RANGES = {"H": (2, 17), "sxr_t": (1, 35), "sxr_v": (1, 35)}
    _KK3_RANGE = (1, 96)

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "https://sal.jet.uk",
        default_error: float = 0.05,
        max_freq: float = 1e6,
        selector: DataSelector = choose_on_plot,
        session: Session = global_session,
    ):
        self.reader_cache_id = f"ppf:{server.replace('-', '_')}:{pulse}"
        self.NAMESPACE: Tuple[str, str] = ("jet", server)
        super().__init__(
            tstart, tend, max_freq, session, selector, pulse=pulse, server=server
        )
        self.pulse = pulse
        self._client = SALClient(server)
        self._client.prompt_for_password = False
        self._default_error = default_error

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: int
    ) -> Tuple[Signal, str]:
        """Gets the signal for the given DDA, at the given revision."""
        path_template = "/pulse/{:d}/ppf/signal/{}/{}/{}:{:d}"
        path = path_template.format(self.pulse, uid, instrument, quantity, revision)
        info = self._client.list(path)
        path = path_template.format(
            self.pulse, uid, instrument, quantity, info.revision_current,
        )
        return self._client.get(path), path

    def _get_charge_exchange(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Return temperature, angular frequency, or concentration data for an
        ion, measured using charge exchange recombination
        spectroscopy.

        """
        results = {}
        R, R_path = self._get_signal(uid, instrument, "rpos", revision)
        z, z_path = self._get_signal(uid, instrument, "pos", revision)
        mass, m_path = self._get_signal(uid, instrument, "mass", revision)
        texp, t_path = self._get_signal(uid, instrument, "texp", revision)
        # TODO: get element from mass
        results["R"] = R
        results["z"] = z
        results["length"] = len(R)
        results["element"] = None
        results["texp"] = None
        paths = [R_path, z_path, m_path, t_path]
        if "angf" in quantities:
            angf, a_path = self._get_signal(uid, instrument, "angf", revision)
            afhi, e_path = self._get_signal(uid, instrument, "afhi", revision)
            results["angf"] = angf.data
            results["angf_err"] = afhi.data
            results["angf_records"] = paths + [a_path, e_path]
        if "conc" in quantities:
            conc, c_path = self._get_signal(uid, instrument, "conc", revision)
            cohi, e_path = self._get_signal(uid, instrument, "cohi", revision)
            results["conc"] = conc.data
            results["conc_err"] = cohi.data
            results["conc_records"] = paths + [c_path, e_path]
        if "ti" in quantities:
            ti, t_path = self._get_signal(uid, instrument, "ti", revision)
            tihi, e_path = self._get_signal(uid, instrument, "ti_hi", revision)
            results["ti"] = ti.data
            results["ti_err"] = tihi.data
            results["ti_records"] = paths + [t_path, e_path]
        return results

    def _get_thomson_scattering(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Produce :py:class:`xarray.DataArray` for electron temperature or
        number density."""
        results = {}
        z, z_path = self._get_signal(uid, instrument, "z", revision)
        results["z"] = z.data
        results["R"] = z.dimensions[0].data
        results["length"] = len(z.data)
        nstart = 0
        nend = -1
        if "te" in quantities:
            te, t_path = self._get_signal(uid, instrument, "te", revision)
            self._set_times_item(results, te.dimensions[0].data, nstart, nend)
            results["te"] = te.data
            if instrument == "lidr":
                tehi, e_path = self._get_signal(uid, instrument, "teu", revision)
                results["te_error"] = tehi.data - results["te"]
            else:
                dte, e_path = self._get_signal(uid, instrument, "dte", revision)
                results["te_error"] = dte.data
            results["te_records"] = [z_path, t_path, e_path]
        if "ne" in quantities:
            ne, d_path = self._get_signal(uid, instrument, "ne", revision)
            self._set_times_item(results, ne.dimensions[0].data, nstart, nend)
            results["ne"] = ne.data
            if instrument == "lidr":
                nehi, e_path = self._get_signal(uid, instrument, "neu", revision)
                results["ne_error"] = nehi.data - results["ne"]
            else:
                dne, e_path = self._get_signal(uid, instrument, "dne", revision)
                results["ne_error"] = dne.data
            results["ne_records"] = [z_path, d_path, e_path]
        return results

    # def _handle_equilibrium_position(self, key: str,
    #                                  revision: int) -> DataArray:
    #     """Produce :py:class:`xarray.DataArray` for data relating to position
    #     of equilibrium."""
    #     signal, uid = self._get_signal(key, revision)
    #     meta = {"datatype": self.AVAILABLE_DATA[key]}
    #     data = DataArray(signal.data, [("t", signal.dimensions[0].data)],
    #                      name=key, attrs=meta)
    #     data.attrs['provenance'] = self.create_provenance(key, revision,
    #                                                       [uid])
    #     return data

    # def _handle_kk3(self, key: str, revision: int) -> DataArray:
    #     """Produce :py:class:`xarray.DataArray` for electron temperature."""
    #     uid, general_dat = self._get_signal("kk3_gen", revision)
    #     channel_index = np.argwhere(general_dat.data[0, :] > 0)
    #     f_chan = general_dat.data[15, channel_index]
    #     nharm_chan = general_dat.data[11, channel_index]
    #     uids = [uid]
    #     temperatures = []
    #     Btot = []

    #     for i, f, nharm in zip(channel_index, f_chan, nharm_chan):
    #         uid, signal = self._get_signal("{}{:02d}".format(key, i),
    #                                        revision)
    #         uids.append(uid)
    #         temperatures.append(signal.data)
    #         Btot.append(2 * np.pi, f * sc.m_e / (nharm * sc.e))

    #     uncalibrated = Btot[general_dat.data[18, channel_index] != 0.0]
    #     temps_array = np.array(temperatures)
    #     coords = [("Btot", np.array(Btot)), ("t", signal.dimensions[0].data)]
    #     meta = {"datatype": self.AVALABLE_DATA[key],
    #             "error": DataArray(0.1*temps_array, coords)}
    #     # TODO: Select correct time range
    #     data = DataArray(temps_array, coords, name=key,
    #                      attrs=meta)
    #     drop = self._select_channels(uid, data, "Btot", uncalibrated)
    #     data.attrs["provenance"] = self.create_provenance(key, revision,
    #                                                       uids, drop)
    #     return data.drop_sel({"Btot": drop})

    # def _handle_sxr(self, key: str, revision: int) -> DataArray:
    #     """Produce :py:class:`xarray.DataArray` for line-of-site soft X-ray
    #     luminous flux."""
    #     uids = []
    #     luminosities = []
    #     channels = []
    #     for i in range(self._SXR_RANGES[key][0],
    #                    self._SXR_RANGES[key][1] + 1):
    #         try:
    #             uid, signal = self._get_signal("{}{:02d}".format(key, i),
    #                                            revision)
    #         except (InvalidPath, NodeNotFound):
    #             continue
    #         uids.append(uid)
    #         luminosities.append(signal.data)
    #         channels.append(i)
    #     # TODO: embed coordinate transform data
    #     # TODO: select only the required times
    #     lum_array = np.array(luminosities)
    #     keychan = key + "_channel"
    #     coords = [(keychan, channels), ("t", signal.dimensions[0].data)]
    #     meta = {"datatype": self.AVALABLE_DATA[key],
    #             "error": DataArray(self._default_error * lum_array, coords)}
    #     data = DataArray(lum_array, coords, name=key, attrs=meta)
    #     drop = self._selector(uid, data, keychan, [])
    #     data.attrs['provenance'] = self.create_provenance(key, revision,
    #                                                       uids, drop)
    #     return data.drop_sel({keychan: drop})

    def close(self):
        """Ends connection to the SAL server from which PPF data is being
        read."""
        del self._client

    @property
    def requires_authentication(self):
        """Indicates whether authentication is required to read data.

        Returns
        -------
        :
            True if authenticationis needed, otherwise false.
        """
        # Perform the necessary logic to know whether authentication is needed.
        return not socket.gethostname().startswith("heimdall")

    def authenticate(self, name: str, password: str):
        """Log onto the JET/SAL system to access data.

        Parameters
        ----------
        name:
            Your username when logging onto Heimdall.
        password:
            SecureID passcode (pin followed by value displayed on token).

        Returns
        -------
        :
            Indicates whether authentication was succesful.
        """
        try:
            self._client.authenticate(name, password)
            return True
        except AuthenticationFailed:
            return False
