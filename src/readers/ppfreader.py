"""Provides implementation of :py:class:`readers.DataReader` for
reading PPF data produced by JET.

"""

import socket
from typing import ClassVar, Dict, Tuple

import numpy as np
from sal.client import SALClient, AuthenticationFailed
from sal.dataclass import Signal
from sal.core.exception import InvalidPath, NodeNotFound
import scipy.constants as sc
from xarray import DataArray

from reader import DataReader, DataSelector
from datatypes import DataType
import session


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
    sess : session.Session
        An object representing the session being run. Contains information
        such as provenance data.

    Attributes
    ----------
    AVAILABLE_DATA: Dict[str, DataType]
        A mapping of the keys used to get each piece of data to the type of
        data associated with that key.
    NAMESPACE: Tuple[str, str]
        The abbreviation and full URL for the PROV namespace of the reader
        class.

    """

    AVAILABLE_DATA: ClassVar[Dict[str, DataType]] = {
        "cxg6_angf": ("angular_freq", None),
        "cxg6_conc": ("concentration", None),
        "cxg6_ti": ("temperature", None),
        "efit_rmag": ("major_rad", "mag_axis"),
        "efit_zmag": ("z", "mag_axis"),
        "efit_rsep": ("major_rad", "separatrix_axis"),
        "efit_zsep": ("z", "separatrix_axis"),
        "hrts_ne": ("number_density", "electrons"),
        "hrts_te": ("temperature", "electrons"),
        "kk3_te": ("temperature", "electrons"),
        "lidr_ne": ("number_density", "electrons"),
        "lidr_te": ("temperature", "electrons"),
        "sxr_h": ("luminous_flux", "sxr"),
        "sxr_t": ("luminous_flux", "sxr"),
        "sxr_v": ("luminous_flux", "sxr"),
    }

    _HANDLER_METHODS: ClassVar[Dict[str, str]] = {
        "cxg6_angf": "_handle_cxrs",
        "cxg6_conc": "_handle_cxrs",
        "cxg6_ti": "_handle_cxrs",
        "efit_rmag": "_handle_equilibrium_position",
        "efit_zmag": "_handle_equilibrium_position",
        "efit_rsep": "_handle_equilibrium_position",
        "efit_zsep": "_handle_equilibrium_position",
        "hrts_ne": "_handle_electron_data",
        "hrts_te": "_handle_electron_data",
        "kk3_te": "_handle_kk3",
        "lidr_ne": "_handle_electron_data",
        "lidr_te": "_handle_electron_data",
        "sxr_h": "_handle_sxr",
        "sxr_t": "_handle_sxr",
        "sxr_v": "_handle_sxr",
    }

    _CXRS_ERROR = {'angf': 'afhi', 'conc': 'cohi', 'ti': 'ti_hi'}
    _SXR_RANGES = {'sxr_h': (2, 17), 'sxr_t': (1, 35), 'sxr_v': (1, 35)}

    def __init__(self, times: np.ndarray, pulse: int, uid: str = "jetppf",
                 server: str = "https://sal.jet.uk",
                 default_error: float = 0.05, selector: DataSelector = None,
                 sess: session.Session = session.global_session):
        self.NAMESPACE: Tuple[str, str] = ("jet", server)
        super().__init__(sess, selector, pulse=pulse, uid=uid, server=server)
        self.pulse = pulse
        self.uid = uid
        self.times = times
        self._client = SALClient(server)
        self._default_error = default_error

    def _get_data(self, key: str, revision: int = 0) -> DataArray:
        """Reads and returns the data for the given key. Should only be called
        by :py:meth:`DataReader.get`."""
        return getattr(self, self._HANDLER_METHODS[key])(key, revision)

    def _get_signal(self, key: str, revision: int) -> Tuple[Signal, str]:
        """Gets the signal for the given DDA, at the given revision."""
        path = "/pulse/{:i}/ppf/signal/{}/{}:{:d}".format(
            self.pulse, self.uid, key.replace("_", "/"), revision)
        # TODO: if revision == 0 update it with absolute revision
        # number in path before returning
        return self._client.get(path), path

    def _handle_cxrs(self, key: str, revision: int) -> DataArray:
        """Return temperature, angular frequency, or concentration data for an
        ion, measured using charge exchange recombination
        spectroscopy.

        """
        key_parts = key.split("_")
        instrument = key_parts[0]
        diagnostic = key_parts[1]
        uid, signal = self._get_signal(key, revision)
        uids = [uid]
        uid, high = self._get_signal(instrument + self._CXRS_ERROR[diagnostic],
                                     revision)
        uids.append(uid)
        uid, atomic_mass = self._get_signal(instrument + "_mass",
                                            revision)
        uids.append(uid)
        # TODO: get ion species from atomic mass
        ion = None
        uid, texp = self._get_signal(instrument + "_texp", revision)
        uids.append(uid)
        ticks = np.arange(signal.dimensions[1].length)
        keycoord = instrument + "_coord"
        coords = [("t", signal.dimensions[0].data), (keycoord, ticks)]
        error = (high.data - signal.data)/2
        meta = {"datatype": (self.AVALABLE_DATA[key][0], ion),
                "error": DataArray(error, coords), "exposure_time": texp.data}
        data = DataArray(signal.data, coords, name=key, attrs=meta)
        r0, uid = self._get_signal(instrument + "_rpos", revision)
        uids.append(uid)
        z, uid = self._get_signal(instrument + "_pos", revision)
        uids.append(uid)
        # TODO: use r0, z to create some sort of converter object
        # TODO: should I use one cache-key for all diagnostics from
        # the same instrument?
        drop = self._select_channels(uids[0], data, keycoord)
        data.attrs['provenance'] = self.create_provenance(key, revision, uids,
                                                          drop)
        return data.drop_sel({keycoord: drop})

    def _handle_electron_data(self, key: str, revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for electron temperature or
        number density."""
        uids = []
        uid, signal = self._get_signal(key, revision)
        uids.append(uid)
        uid, error = self._get_signal(key[:-2] + "d" + key[-2:], revision)
        uids.append(uid)
        ticks = np.arange(signal.dimensions[1].length)
        r0 = signal.dimensions[1].data
        z, uid = self._get_signal(key[:-2] + "z", revision)
        uids.append(uid)
        keycoord = key[:-2] + "coord"
        coords = [("t", signal.dimensions[0].data), (keycoord, ticks)]
        meta = {"datatype": self.AVAILABLE_DATA[key],
                "error": DataArray(error.data, coords)}
        data = DataArray(signal.data, coords, name=key, attrs=meta)
        drop = self._select_channels(uids[0], data, keycoord)
        data.attrs['provenance'] = self.create_provenance(key, revision, uids,
                                                          drop)
        return data.drop_sel({keycoord: drop})

    def _handle_equilibrium_position(self, key: str,
                                     revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for data relating to position
        of equilibrium."""
        signal, uid = self._get_signal(key, revision)
        meta = {"datatype": self.AVAILABLE_DATA[key]}
        data = DataArray(signal.data, [("t", signal.dimensions[0].data)],
                         name=key, attrs=meta)
        data.attrs['provenance'] = self.create_provenance(key, revision,
                                                          [uid])
        return data

    def _handle_kk3(self, key: str, revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for electron temperature."""
        uid, general_dat = self._get_signal("kk3_gen", revision)
        channel_index = np.argwhere(general_dat.data[0, :] > 0)
        f_chan = general_dat.data[15, channel_index]
        nharm_chan = general_dat.data[11, channel_index]
        uids = [uid]
        temperatures = []
        Btot = []

        for i, f, nharm in zip(channel_index, f_chan, nharm_chan):
            uid, signal = self._get_signal("{}{:02d}".format(key, i),
                                           revision)
            uids.append(uid)
            temperatures.append(signal.data)
            Btot.append(2 * np.pi, f * sc.m_e / (nharm * sc.e))

        uncalibrated = Btot[general_dat.data[18, channel_index] != 0.0]
        temps_array = np.array(temperatures)
        coords = [("Btot", np.array(Btot)), ("t", signal.dimensions[0].data)]
        meta = {"datatype": self.AVALABLE_DATA[key],
                "error": DataArray(0.1*temps_array, coords)}
        # TODO: Select correct time range
        data = DataArray(temps_array, coords, name=key,
                         attrs=meta)
        drop = self._select_channels(uid, data, "Btot", uncalibrated)
        data.attrs["provenance"] = self.create_provenance(key, revision, uids,
                                                          drop)
        return data.drop_sel({"Btot": drop})

    def _handle_sxr(self, key: str, revision: int) -> DataArray:
        """Produce :py:class:`xarray.DataArray` for line-of-site soft X-ray
        luminous flux."""
        uids = []
        luminosities = []
        channels = []
        for i in range(self._SXR_RANGES[key][0], self._SXR_RANGES[key][1] + 1):
            try:
                uid, signal = self._get_signal("{}{:02d}".format(key, i),
                                               revision)
            except (InvalidPath, NodeNotFound):
                continue
            uids.append(uid)
            luminosities.append(signal.data)
            channels.append(i)
        # TODO: embed coordinate transform data
        # TODO: select only the required times
        lum_array = np.array(luminosities)
        keychan = key + "_channel"
        coords = [(keychan, channels), ("t", signal.dimensions[0].data)]
        meta = {"datatype": self.AVALABLE_DATA[key],
                "error": DataArray(self._default_error * lum_array, coords)}
        data = DataArray(lum_array, coords, name=key, attrs=meta)
        drop = self._selector(uid, data, keychan, [])
        data.attrs['provenance'] = self.create_provenance(key, revision, uids,
                                                          drop)
        return data.drop_sel({keychan: drop})

    def close(self):
        """Ends connection to the SAL server from which PPF data is being
        read."""
        del self.server

    @property
    def requires_authentication(self):
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
