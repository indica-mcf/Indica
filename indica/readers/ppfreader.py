"""Provides implementation of :py:class:`readers.DataReader` for
reading PPF data produced by JET.

"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple

import numpy as np
from sal.client import SALClient
from sal.core.exception import AuthenticationFailed
from sal.core.exception import NodeNotFound
from sal.dataclass import Signal

import indica.readers.surf_los as surf_los
from .abstractreader import DataReader
from .abstractreader import DataSelector
from .selectors import choose_on_plot
from ..session import global_session
from ..session import Session


SURF_PATH = Path(surf_los.__file__).parent / "surf_los.dat"


class PPFError(Exception):
    """An exception which occurs when trying to read PPF data which would
    not be caught by the lower-level SAL library. An example would be
    failing to find any valid channels for an instrument when each channel
    is a separate DTYPE.

    """


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
    _BREMSSTRAHLUNG_LOS = {
        "ks3": "edg7",
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

    _RADIATION_RANGES = {
        "sxr/h": 17,
        "sxr/t": 35,
        "sxr/v": 35,
        "bolo/kb5h": 24,
        "bolo/kb5v": 32,
    }
    _KK3_RANGE = (1, 96)
    MACHINE_DIMS = ((1.83, 3.9), (-1.75, 2.0))

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
        self._reader_cache_id = f"ppf:{server.replace('-', '_')}:{pulse}"
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
        results["R"] = R.data
        results["z"] = z.data
        results["length"] = len(R.data)
        results["element"] = None
        results["texp"] = None
        results["times"] = None
        paths = [R_path, z_path, m_path, t_path]
        if "angf" in quantities:
            angf, a_path = self._get_signal(uid, instrument, "angf", revision)
            afhi, e_path = self._get_signal(uid, instrument, "afhi", revision)
            results["angf"] = angf.data
            results["angf_error"] = afhi.data
            results["angf_records"] = paths + [a_path, e_path]
        if "conc" in quantities:
            conc, c_path = self._get_signal(uid, instrument, "conc", revision)
            cohi, e_path = self._get_signal(uid, instrument, "cohi", revision)
            results["conc"] = conc.data
            results["conc_error"] = cohi.data
            results["conc_records"] = paths + [c_path, e_path]
        if "ti" in quantities:
            ti, t_path = self._get_signal(uid, instrument, "ti", revision)
            tihi, e_path = self._get_signal(uid, instrument, "tihi", revision)
            results["ti"] = ti.data
            results["ti_error"] = tihi.data
            results["ti_records"] = paths + [t_path, e_path]
        return results

    def _get_thomson_scattering(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for electron temperature or number density
        calculated from Thomson scattering.

        """
        results = {}
        z, z_path = self._get_signal(uid, instrument, "z", revision)
        results["z"] = z.data
        results["R"] = z.dimensions[0].data
        results["length"] = len(z.data)
        if "te" in quantities:
            te, t_path = self._get_signal(uid, instrument, "te", revision)
            self._set_times_item(results, te.dimensions[0].data)
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
            self._set_times_item(results, ne.dimensions[0].data)
            results["ne"] = ne.data
            if instrument == "lidr":
                nehi, e_path = self._get_signal(uid, instrument, "neu", revision)
                results["ne_error"] = nehi.data - results["ne"]
            elif instrument == "kg10":
                results["ne_error"] = 0.0 * results["ne"]
            else:
                dne, e_path = self._get_signal(uid, instrument, "dne", revision)
                results["ne_error"] = dne.data
            results["ne_records"] = [z_path, d_path]
            if instrument != "kg10":
                results["ne_records"].append(e_path)
        return results

    def _get_equilibrium(
        self, uid: str, calculation: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for plasma equilibrium.

        """
        results: Dict[str, Any] = {}
        for q in quantities:
            qval, q_path = self._get_signal(uid, calculation, q, revision)
            self._set_times_item(results, qval.dimensions[0].data)
            if (
                len(qval.dimensions) > 1
                and q not in {"psi", "rbnd", "zbnd"}
                and "psin" not in results
            ):
                results["psin"] = qval.dimensions[1].data
            if q == "psi":
                r, r_path = self._get_signal(uid, calculation, "psir", revision)
                z, z_path = self._get_signal(uid, calculation, "psiz", revision)
                results["psi_r"] = r.data
                results["psi_z"] = z.data
                results["psi"] = qval.data.reshape(
                    (len(results["times"]), len(z.data), len(r.data))
                )
                results["psi_records"] = [q_path, r_path, z_path]
            else:
                results[q] = qval.data
                results[q + "_records"] = [q_path]
        return results

    def _get_radiation(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for radiation quantities such as SXR and bolometric
        fluxes..

        """
        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        for q in quantities:
            qtime = q + "_times"
            luminosities = []
            channels = []
            records = [SURF_PATH.name]
            for i in range(1, self._RADIATION_RANGES[instrument + "/" + q] + 1):
                try:
                    qval, q_path = self._get_signal(
                        uid, instrument, f"{q}{i:02d}", revision
                    )
                except NodeNotFound:
                    continue
                records.append(q_path)
                luminosities.append(qval.data)
                channels.append(i - 1)
                if qtime not in results:
                    results[qtime] = qval.dimensions[0].data
            if len(channels) == 0:
                # Try getting information on the DDA, to determine if
                # the failure is actually due to requesting an invalid
                # DDA or revision
                self._client.list(
                    f"/pulse/{self.pulse:d}/ppf/signal/{uid}/{instrument}:{revision:d}"
                )
                raise PPFError(f"No channels available for {instrument}/{q}.")
            results["length"][q] = len(channels)
            results[q] = np.array(luminosities).T
            results[q + "_error"] = self._default_error * results[q]
            results[q + "_records"] = records
            rstart, rend, zstart, zend, Tstart, Tend = surf_los.read_surf_los(
                SURF_PATH, self.pulse, instrument.lower() + "/" + q.lower()
            )
            results[q + "_Rstart"] = rstart[channels]
            results[q + "_Rstop"] = rend[channels]
            results[q + "_zstart"] = zstart[channels]
            results[q + "_zstop"] = zend[channels]
            results[q + "_Tstart"] = Tstart[channels]
            results[q + "_Tstop"] = Tend[channels]
        return results

    def _get_bremsstrahlung_spectroscopy(
        self, uid: str, instrument: str, revision: int, quantities: Set[str],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        los_dda = self._BREMSSTRAHLUNG_LOS[instrument]
        for q in quantities:
            qval, q_path = self._get_signal(uid, instrument, q, revision)
            los, l_path = self._get_signal(uid, los_dda, "los" + q[-1], revision)
            if "times" not in results:
                results["times"] = qval.dimensions[0].data
            results["length"][q] = 1
            results[q] = qval.data
            results[q + "_error"] = 0.0 * results[q]
            results[q + "_Rstart"] = np.array([los.data[1] / 1000])
            results[q + "_Rend"] = np.array([los.data[4] / 1000])
            results[q + "_zstart"] = np.array([los.data[2] / 1000])
            results[q + "_zend"] = np.array([los.data[5] / 1000])
            results[q + "_records"] = [q_path, l_path]
        return results

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
        try:
            self._client.list("/")
            return False
        except AuthenticationFailed:
            return True

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
