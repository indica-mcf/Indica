"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""

from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from MDSplus import Connection
from MDSplus.mdsExceptions import TreeNNF
import numpy as np

from .abstractreader import DataReader
from .. import session
from ..numpy_typing import RevisionLike


class MDSError(Exception):
    """An exception which occurs when trying to read MDS+ data which would
    not be caught by the lower-level MDSplus library. An example would be
    failing to find any valid channels for an instrument when each channel
    is a separate DTYPE.

    """


class MDSWarning(UserWarning):
    """A warning that occurs while trying to read MDS+ data. Typically
    related to caching in some way.

    """


class ST40Reader(DataReader):
    """Class to read ST40 MDS+ data using MDSplus.

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
    DIAGNOSTIC_QUANTITIES: Dict[str, Dict[str, Dict[str, Dict[str, ArrayType]]]]
        Hierarchical information on the quantities which are available for
        reading. These are indexed by (in order) diagnostic name, UID,
        instrument name, and quantity name. The values of the innermost
        dictionary describe the physical type of the data to be read.
    NAMESPACE: Tuple[str, str]
        The abbreviation and full URL for the PROV namespace of the reader
        class.

    """

    MACHINE_DIMS = ((0.15, 0.85), (-0.75, 0.75))
    INSTRUMENT_METHODS = {
        "efit": "get_equilibrium",
        "xrcs": "get_helike_spectroscopy",
        "princeton": "get_charge_exchange",
        "brems": "get_diode_filters",
        "halpha": "get_diode_filters",
        "nirh1": "get_interferometry",
        "nirh1_bin": "get_interferometry",
        "smmh1": "get_interferometry",
        "astra": "get_astra",
        "transp_test": "get_transp_test",
        "sxr_diode_1": "get_diode_filters",
        "sxr_diode_2": "get_diode_filters",
        "sxr_diode_3": "get_diode_filters",
        "sxr_diode_4": "get_diode_filters",
        "sxr_camera_4": "get_radiation",
        "sxrc_xy1": "get_radiation",
        "sxrc_xy2": "get_radiation",
        "cxff_pi": "get_charge_exchange",
        "cxff_tws_c": "get_charge_exchange",
        "cxqf_tws_c": "get_charge_exchange",
        "ts": "get_thomson_scattering",
    }
    UIDS_MDS = {
        "efit": "",
        "xrcs": "sxr",
        "princeton": "spectrom",
        "brems": "spectrom",
        "halpha": "spectrom",
        "nirh1": "interferom",
        "nirh1_bin": "interferom",
        "smmh1": "interferom",
        "astra": "",
        "transp_test": "",
        "sxr_diode_1": "sxr",
        "sxr_diode_2": "sxr",
        "sxr_diode_3": "sxr",
        "sxr_diode_4": "sxr",
        "sxr_camera_1": "sxr",
        "sxr_camera_2": "sxr",
        "sxr_camera_3": "sxr",
        "sxr_camera_4": "sxr",
        "cxff_pi": "",
        "cxff_tws_c": "",
        "cxqf_tws_c": "",
        "ts": "",
    }
    QUANTITIES_MDS = {
        "efit": {
            "f": ".profiles.psi_norm:f",
            "faxs": ".global:faxs",
            "fbnd": ".global:fbnd",
            "ftor": ".profiles.psi_norm:ftor",
            "rmji": ".profiles.psi_norm:rmji",
            "rmjo": ".profiles.psi_norm:rmjo",
            "psi": ".psi2d:psi",
            "vjac": ".profiles.psi_norm:vjac",
            "ajac": ".profiles.psi_norm:ajac",
            "rmag": ".global:rmag",
            "rgeo": ".global:rgeo",
            "rbnd": ".p_boundary:rbnd",
            "zmag": ".global:zmag",
            "zbnd": ".p_boundary:zbnd",
            "ipla": ".constraints.ip:cvalue",
            "wp": ".virial:wp",
            "df": ".constraints.df:cvalue",
        },
        "xrcs": {
            "int_k": ".te_kw:int_k",
            "int_w": ".te_kw:int_w",
            "int_z": ".te_kw:int_z",
            "int_q": ".te_kw:int_q",
            "int_r": ".te_kw:int_r",
            "int_a": ".te_kw:int_a",
            "int_n3": ".te_n3w:int_n3",
            "int_tot": ".te_n3w:int_tot",
            "te_kw": ".te_kw:te",
            "te_n3w": ".te_n3w:te",
            "ti_w": ".ti_w:ti",
            "ti_z": ".ti_z:ti",
            "ampl_w": ".ti_w:amplitude",
            "spectra": ":intensity",
        },
        "nirh1": {
            "ne": ".line_int:ne",
        },
        "nirh1_bin": {
            "ne": ".line_int:ne",
        },
        "smmh1": {
            "ne": ".line_int:ne",
        },
        "brems": {
            "brightness": ".brem_mp1:intensity",
        },
        "halpha": {
            "brightness": ".h_alpha_mp1:intensity",
        },
        "sxr_camera_4": {
            "brightness": ".middle_head.filter_4:",
            "location": ".middle_head.geometry:location",
            "direction": ".middle_head.geometry:direction",
        },
        "diode_arrays": {
            "brightness": ".middle_head.filter_4:",
            "location": ".middle_head.geometry:location",
            "direction": ".middle_head.geometry:direction",
        },
        "sxrc_xy1": {
            "brightness": ".profiles:emission",
        },
        "sxrc_xy2": {
            "brightness": ".profiles:emission",
        },
        "sxr_diode_1": {
            "brightness": ".filter_001:signal",
        },
        "sxr_diode_2": {
            "brightness": ".filter_002:signal",
        },
        "sxr_diode_3": {
            "brightness": ".filter_003:signal",
        },
        "sxr_diode_4": {
            "brightness": ".filter_004:signal",
        },
        "cxff_pi": {
            "int": ".profiles:int",
            "ti": ".profiles:ti",
            "vtor": ".profiles:vtor",
        },
        "cxff_tws_c": {
            "int": ".profiles:int",
            "ti": ".profiles:ti",
            "vtor": ".profiles:vtor",
        },
        "cxqf_tws_c": {
            "int": ".profiles:int",
            "ti": ".profiles:ti",
            "vtor": ".profiles:vtor",
        },
        "ts": {
            "ne": ".profiles:ne",
            "te": ".profiles:te",
            "pe": ".profiles:pe",
            "chi2": ".profiles:chi2",
        },
        "astra": {
            "f": ".profiles.psi_norm:fpol",
            "faxs": ".global:faxs",
            "fbnd": ".global:fbnd",
            "ftor": ".profiles.psi_norm:ftor",  # Wb
            # "rmji": ".profiles.psi_norm:rmji",
            # "rmjo": ".profiles.psi_norm:rmjo",
            "psi_1d": ".profiles.psi_norm:psi",
            "psi": ".psi2d:psi",
            # "vjac": ".profiles.psi_norm:vjac",
            # "ajac": ".profiles.psi_norm:ajac",
            "volume": ".profiles.psi_norm:volume",
            "area": ".profiles.psi_norm:areat",
            "rmag": ".global:rmag",
            "rgeo": ".global:rgeo",
            "zmag": ".global:zmag",
            "zgeo": ".global:zgeo",
            "rbnd": ".p_boundary:rbnd",
            "zbnd": ".p_boundary:zbnd",
            "wp": ".global:wth",
            "ipla": ".global:ipl",
            "upl": ".global:upl",
            "wth": ".global:wth",
            "wtherm": ".global:wtherm",
            "wfast": ".global:wfast",
            "df": ".global.df",
            "elon": ".profiles.astra:elon",  # Elongation profile
            "j_bs": ".profiles.astra:j_bs",  # Bootstrap current density,MA/m2
            "j_nbi": ".profiles.astra:j_nbi",  # NB driven current density,MA/m2
            "j_oh": ".profiles.astra:j_oh",  # Ohmic current density,MA/m2
            "j_rf": ".profiles.astra:j_rf",  # EC driven current density,MA/m2
            "j_tot": ".profiles.astra:j_tot",  # Total current density,MA/m2
            "ne": ".profiles.astra:ne",  # Electron density, 10^19 m^-3
            "ni": ".profiles.astra:ni",  # Main ion density, 10^19 m^-3
            "nf": ".profiles.astra:nf",  # Main ion density, 10^19 m^-3
            "n_d": ".profiles.astra:n_d",  # Deuterium density,10E19/m3
            "n_t": ".profiles.astra:n_t",  # Tritium density	,10E19/m3
            "omega_tor": ".profiles.astra:omega_tor",  # Toroidal rot. freq., 1/s
            "qe": ".profiles.astra:qe",  # electron power flux, MW
            "qi": ".profiles.astra:qi",  # ion power flux, MW
            "qn": ".profiles.astra:qn",  # total electron flux, 10^19/s
            "qnbe": ".profiles.astra:qnbe",  # Beam power density to electrons, MW/m3
            "qnbi": ".profiles.astra:qnbi",  # Beam power density to ions, MW/m3
            "q_oh": ".profiles.astra:q_oh",  # Ohmic heating power profile, MW/m3
            "q_rf": ".profiles.astra:q_rf",  # RF power density to electron,MW/m3
            "rho": ".profiles.astra:rho",  # ASTRA rho-toroidal
            "rmid": ".profiles.astra:rmid",  # Centre of flux surfaces, m
            "rminor": ".profiles.astra:rminor",  # minor radius, m
            "sbm": ".profiles.astra:sbm",  # Particle source from beam, 10^19/m^3/s
            "spel": ".profiles.astra:spel",  # Particle source from pellets, 10^19/m^3/s
            "stot": ".profiles.astra:stot",  # Total electron source,10^19/s/m3
            "swall": ".profiles.astra:swall",  # Wall neutrals source, 10^19/m^3/s
            "te": ".profiles.astra:te",  # Electron temperature, keV
            "ti": ".profiles.astra:ti",  # Ion temperature, keV
            "tri": ".profiles.astra:tri",  # Triangularity (up/down symmetrized) profile
            "t_d": ".profiles.astra:t_d",  # Deuterium temperature,keV
            "t_t": ".profiles.astra:t_t",  # Tritium temperature,keV
            "zeff": ".profiles.astra:zeff",  # Effective ion charge
            "areat": ".profiles.psi_norm:areat",  # Toroidal cross section,m2
            "p": ".profiles.psi_norm:p",  # PRESSURE(PSI_NORM)
            "pblon": ".profiles.astra:pblon",  # PRESSURE(PSI_NORM)
            "pbper": ".profiles.astra:pbper",  # PRESSURE(PSI_NORM)
            "pnb": ".global:pnb",  # Injected NBI power, W
            "pabs": ".global:pabs",  # Absorber NBI power, W
            "p_oh": ".global:p_oh",  # Absorber NBI power, W
            "q": ".profiles.psi_norm:q",  # Q_PROFILE(PSI_NORM)
            "sigmapar": ".profiles.psi_norm:sigmapar",  # Paral. conduct.,1/(Ohm*m)
            "nn": ".profiles.astra:nn",  # Thermal neutral density, 10^19/m^3
            "niz1": ".profiles.astra:niz1",  # Impurity density, 10^19/m^3
            "niz2": ".profiles.astra:niz2",  # Impurity density, 10^19/m^3
            "niz3": ".profiles.astra:niz3",  # Impurity density, 10^19/m^3
        },
        "transp_test": {
            "f": ".profiles.rhotor:f",
            "faxs": ".global:faxs",
            "fbnd": ".global:fbnd",
            "ftor": ".profiles.rhotor:ftor",  # Wb
            # # "rmji": ".profiles.psi_norm:rmji",
            # # "rmjo": ".profiles.psi_norm:rmjo",
            # "psi_1d": ".profiles.psi_norm:psi",
            "psi": ".psi2d:psi",
            # "vjac": ".profiles.psi_norm:vjac",
            # "ajac": ".profiles.psi_norm:ajac",
            "volume": ".profiles.rhotor:volume",
            "area": ".profiles.rhotor:area",
            "rmag": ".global:rmag",
            "rgeo": ".global:rgeo",
            "zmag": ".global:zmag",
            # "zgeo": ".global:zgeo",
            "rbnd": ".p_boundary:rbnd",
            "zbnd": ".p_boundary:zbnd",
            "wp": ".global:ueq",
            "ipla": ".global:ip",
            # "upl": ".global:upl",
            # "wth": ".global:wth",
            # "wtherm": ".global:wtherm",
            # "wfast": ".global:wfast",
            # "df": ".global.df",
            # "elon": ".profiles.astra:elon",  # Elongation profile
            "j_bs": ".profiles.rhotor:j_bs",  # Bootstrap current density,MA/m2
            "j_nbi": ".profiles.rhotor:j_nbi",  # NB driven current density,MA/m2
            "j_oh": ".profiles.rhotor:j_oh",  # Ohmic current density,MA/m2
            # "j_rf": ".profiles.astra:j_rf",  # EC driven current density,MA/m2
            "j_tot": ".profiles.rhotor:j_total",  # Total current density,MA/m2
            "ne": ".profiles.rhotor:ne",  # Electron density, 10^19 m^-3
            "ni": ".profiles.rhotor:ni",  # Main ion density, 10^19 m^-3
            "nf": ".profiles.rhotor:nf",  # Main ion density, 10^19 m^-3
            # "n_d": ".profiles.astra:n_d",  # Deuterium density,10E19/m3
            # "n_t": ".profiles.astra:n_t",  # Tritium density	,10E19/m3
            "omega_tor": ".profiles.rhotor:omega_tor",  # Toroidal rot. freq., 1/s
            # "qe": ".profiles.astra:qe",  # electron power flux, MW
            # "qi": ".profiles.astra:qi",  # ion power flux, MW
            # "qn": ".profiles.astra:qn",  # total electron flux, 10^19/s
            "qnbe": ".profiles.rhotor:q_nbi_e",
            "qnbi": ".profiles.rhotor:q_nbi_i",  # Beam power density to ions, MW/m3
            "q_oh": ".profiles.rhotor:q_oh",  # Ohmic heating power profile, MW/m3
            # "q_rf": ".profiles.astra:q_rf",  # RF power density to electron,MW/m3
            "rho": ".profiles.rhotor:rhotor",  # ASTRA rho-toroidal
            "rmid": ".profiles.rhotor:r_mid",  # Centre of flux surfaces, m
            # "rminor": ".profiles.astra:rminor",  # minor radius, m
            # "sbm": ".profiles.astra:sbm",  # Particle source from beam, 10^19/m^3/s
            # "stot": ".profiles.astra:stot",  # Total electron source,10^19/s/m3
            # "swall": ".profiles.astra:swall",  # Wall neutrals source, 10^19/m^3/s
            "te": ".profiles.rhotor:te",  # Electron temperature, keV
            "ti": ".profiles.rhotor:ti",  # Ion temperature, keV
            # "t_d": ".profiles.astra:t_d",  # Deuterium temperature,keV
            # "t_t": ".profiles.astra:t_t",  # Tritium temperature,keV
            "zeff": ".profiles.rhotor:zeff",  # Effective ion charge
            # "areat": ".profiles.psi_norm:areat",  # Toroidal cross section,m2
            "p": ".profiles.rhotor:pressure",  # PRESSURE(PSI_NORM)
            "pblon": ".profiles.rhotor:pblon",  # PRESSURE(PSI_NORM)
            "pbper": ".profiles.rhotor:pbper",  # PRESSURE(PSI_NORM)
            # "pnb": ".global:pnb",  # Injected NBI power, W
            # "pabs": ".global:pabs",  # Absorber NBI power, W
            # "p_oh": ".global:p_oh",  # Absorber NBI power, W
            "q": ".profiles.rhotor:q",  # Q_PROFILE(PSI_NORM)
            # "sigmapar": ".profiles.psi_norm:sigmapar",  # Paral. conduct.,1/(Ohm*m)
            "nn": ".profiles.rhotor:nwn",  # Thermal neutral density, 10^19/m^3
            "niz1": ".profiles.rhotor:niz1",  # Impurity density, 10^19/m^3
            # "niz2": ".profiles.astra:niz2",  # Impurity density, 10^19/m^3
            # "niz3": ".profiles.astra:niz3",  # Impurity density, 10^19/m^3
        },
    }

    _IMPLEMENTATION_QUANTITIES = {  # TODO: these will be different diagnostics!!!!!!!
        "diode_arrays": {  # GETTING THE DATA OF THE SXR CAMERA
            "sxr_camera_1": ("brightness", "total"),
            "sxr_camera_2": ("brightness", "50_Al_filtered"),
            "sxr_camera_3": ("brightness", "250_Be_filtered"),
            "sxr_camera_4": ("brightness", "10_Be_filtered"),
        },
    }

    _RADIATION_RANGES = {
        "sxr_camera_1": (1, 20),
        "sxr_camera_2": (21, 40),
        "sxr_camera_3": (41, 60),
        "sxr_camera_4": (61, 80),
    }

    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "10.0.40.13",
        tree: str = "ST40",
        default_error: float = 0.05,
        max_freq: float = 1e6,
        session: session.Session = session.global_session,
    ):
        self._reader_cache_id = f"st40:{server.replace('-', '_')}:{pulse}"
        self.NAMESPACE: Tuple[str, str] = ("st40", server)
        super().__init__(
            tstart,
            tend,
            max_freq,
            session,
            pulse=pulse,
            server=server,
            default_error=default_error,
        )
        self.pulse: int = pulse
        self.tree: str = tree
        self.conn: Connection = Connection(server)
        self.conn.openTree(self.tree, self.pulse)
        self._default_error = default_error

    def get_mds_path(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[str, str]:
        """Return the path in the MDS+ database to for the given INSTRUMENT/CODE

        uid: currently redundant --> set to empty string ""
        instrument: e.g. "efit"
        quantity: e.g. ".global:cr0" # minor radius
        revision: if 0 --> looks for "best", else "run##"
        """
        revision_name = self.get_revision_name(revision)
        mds_path = ""
        if len(uid) > 0:
            mds_path += f".{uid}".upper()
        if len(instrument) > 0 and instrument.upper() != self.tree.upper():
            mds_path += f".{instrument}".upper()
        mds_path += f"{revision_name}{quantity}".upper()
        return mds_path, self.mdsCheck(mds_path)

    def get_mds_path_dims(self, mds_path: str, dim: int):
        """Gets the dimensions' path given an mds_path"""

        dims_path = f"dim_of({mds_path},{dim})"
        return dims_path

    def _get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array]]:
        """Gets the signal and its coordinates for the given INSTRUMENT, at the
        given revision."""
        data, _path = self._get_signal(uid, instrument, quantity, revision)
        dims, _ = self._get_signal_dims(_path, len(data.shape))

        return data, dims

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, str]:
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""
        path, path_check = self.get_mds_path(uid, instrument, quantity, revision)
        print(path)
        if quantity.lower() == ":best_run":
            data = str(self.conn.get(path))
        else:
            print("path",path)
            data = np.array(self.conn.get(path))
            # data = np.array(self.conn.get(path_check))

        return data, path

    def _get_signal_dims(
        self,
        mds_path: str,
        ndims: int,
    ) -> Tuple[List[np.array], List[str]]:
        """Gets the dimensions of a signal given the path to the signal
        and the number of dimensions"""

        dimensions = []
        paths = []
        for dim in range(ndims):
            path = f"dim_of({mds_path},{dim})"
            dim_tmp = self.conn.get(self.mdsCheck(path)).data()

            paths.append(path)
            dimensions.append(np.array(dim_tmp))
        return dimensions, paths

    def _get_revision(
        self, uid: str, instrument: str, revision: RevisionLike
    ) -> RevisionLike:
        """
        Gets the effective revision name if latest/best is given in input
        """
        if type(revision) == str:
            return revision

        if revision == 0:
            run_name, _ = self._get_signal(uid, instrument, ":best_run", revision)
            return run_name

        return revision

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for plasma equilibrium."""

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]
        times, _ = self._get_signal(uid, instrument, ":time", revision)
        results["times"] = times
        results["psin"], results["psin_records"] = self._get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        results["psi_r"], results["psi_r_records"] = self._get_signal(
            uid, instrument, ".psi2d:rgrid", revision
        )
        results["psi_z"], results["psi_z_records"] = self._get_signal(
            uid, instrument, ".psi2d:zgrid", revision
        )
        for q in quantities:
            if q not in self.QUANTITIES_MDS[instrument].keys():
                continue
            try:
                qval, q_path = self._get_signal(
                    uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
                )
            except TreeNNF:
                continue

            if q == "psi":
                results["psi"] = qval.reshape(
                    (
                        len(results["times"]),
                        len(results["psi_z"]),
                        len(results["psi_r"]),
                    )
                )
                results["psi_records"] = [
                    results["psin_records"],
                    results["psi_r_records"],
                    results["psi_z_records"],
                ]
            else:
                results[q] = qval
                results[q + "_records"] = [q_path]

        return results

    def _get_astra(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from ASTRA run."""

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # Read time and radial dimensions
        results["boundary_index"], _ = self._get_signal(
            uid, instrument, ".p_boundary:index", revision
        )
        results["psi"], _ = self._get_signal(
            uid, instrument, ".profiles.psi_norm:psi", revision
        )
        results["psin"], psin_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        results["ftor"], _ = self._get_signal(
            uid, instrument, ".profiles.psi_norm:ftor", revision
        )
        results["rho"], rho_path = self._get_signal(
            uid, instrument, ".profiles.astra:rho", revision
        )
        results["psi_r"], _ = self._get_signal(
            uid, instrument, ".psi2d:rgrid", revision
        )
        results["psi_z"], _ = self._get_signal(
            uid, instrument, ".psi2d:zgrid", revision
        )
        results["times"], t_path = self._get_signal(uid, instrument, ":time", revision)
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )

            results[q] = qval
            if "PROFILES.PSI_NORM" in q_path.upper():
                results[q + "_records"] = [q_path, t_path, psin_path]
            elif "PROFILES.ASTRA" in q_path.upper():
                results[q + "_records"] = [q_path, t_path, rho_path]
            else:
                results[q + "_records"] = [q_path, t_path]

        return results

    def _get_radiation_old(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from SXR cameras, legacy old MDS+ structure."""

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        if "sxr_camera" in instrument:
            _instrument = "diode_arrays"
        else:
            _instrument = instrument

        results["revision"] = self._get_revision(uid, _instrument, revision)
        revision = results["revision"]

        location, location_path = self._get_signal(
            uid, _instrument, self.QUANTITIES_MDS[instrument]["location"], revision
        )
        direction, direction_path = self._get_signal(
            uid, _instrument, self.QUANTITIES_MDS[instrument]["direction"], revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        brightness = []
        records = []
        quantity = "brightness"

        times, times_path = self._get_signal(
            uid,
            _instrument,
            f"{self.QUANTITIES_MDS[instrument][quantity]}time",
            revision,
        )

        chan_start, chan_end = self._RADIATION_RANGES[instrument]
        nchan = chan_end - chan_start + 1
        for chan in range(chan_start, chan_end + 1):
            qval, q_path = self._get_signal(
                uid,
                _instrument,
                self.QUANTITIES_MDS[instrument][quantity] + "ch" + str(chan).zfill(3),
                revision,
            )
            records.append(q_path)
            brightness.append(qval)

        results["length"] = nchan
        results["times"] = times
        results[quantity] = np.array(brightness).T
        results[quantity + "_records"] = records
        results[quantity + "_error"] = self._default_error * results[quantity]
        results["location"] = location[chan_start - 1 : chan_end, :]
        results["direction"] = direction[chan_start - 1 : chan_end, :]

        results["quantities"] = quantities

        return results

    def _get_transp_test(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from ASTRA run."""

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # Read time and radial dimensions
        # results["boundary_index"], _ = self._get_signal(
        #     uid, instrument, ".p_boundary:rbnd", revision
        # )
        results["psi"], _ = self._get_signal(uid, instrument, ".psi2d.psi", revision)
        results["psin"], psin_path = self._get_signal(
            uid, instrument, ".profiles.rhotor:psin", revision
        )
        results["ftor"], _ = self._get_signal(
            uid, instrument, ".profiles.rhotor:f", revision
        )
        results["rho"], rho_path = self._get_signal(
            uid, instrument, ".profiles.rhotor:rhotor", revision
        )
        results["psi_r"], _ = self._get_signal(
            uid, instrument, ".psi2d:rgrid", revision
        )
        results["psi_z"], _ = self._get_signal(
            uid, instrument, ".psi2d:zgrid", revision
        )
        results["times"], t_path = self._get_signal(uid, instrument, ":time", revision)
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )

            results[q] = qval
            # if "PROFILES.PSI_NORM" in q_path.upper():
            #     results[q + "_records"] = [q_path, t_path, psin_path]
            if "PROFILES.RHOTOR" in q_path.upper():
                results[q + "_records"] = [q_path, t_path, rho_path]
            else:
                results[q + "_records"] = [q_path, t_path]

        return results

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from SXR cameras."""

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        if "sxr_camera" in instrument:
            return self._get_radiation_old(uid, instrument, revision, quantities)

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self._get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self._get_signal(
            uid, instrument, ".geometry:direction", revision
        )

        quantity = "brightness"
        times, times_path = self._get_signal(
            uid,
            instrument,
            ":time",
            revision,
        )
        qval, qval_record = self._get_signal(
            uid,
            instrument,
            self.QUANTITIES_MDS[instrument][quantity],
            revision,
        )
        qerr, qerr_record = self._get_signal(
            uid,
            instrument,
            self.QUANTITIES_MDS[instrument][quantity] + "_err",
            revision,
        )

        results["length"] = np.shape(qval)[1]
        results["times"] = times
        results[quantity] = np.array(qval)
        results[quantity + "_records"] = qval_record
        results[quantity + "_error"] = qerr
        results["location"] = location
        results["direction"] = direction

        results["quantities"] = quantities

        return results

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self._get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self._get_signal(
            uid, instrument, ".geometry:direction", revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        results["times"], _ = self._get_signal(uid, instrument, ":time", revision)
        results["wavelength"], _ = self._get_signal(
            uid, instrument, ":wavelength", revision
        )
        # TODO: change once wavelength in MDS+ has been fixed to nanometers!
        results["wavelength"] /= 10.0
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            results[q + "_records"] = q_path
            results[q] = qval

            try:
                qval_err, q_path_err = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.full_like(results[q], 0.0)
                q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

        length = location[:, 0].size
        results["length"] = length
        results["location"] = location
        results["direction"] = direction

        return results

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        texp, texp_path = self._get_signal(uid, instrument, ":exposure", revision)
        times, _ = self._get_signal(uid, instrument, ":time", revision)

        x, x_path = self._get_signal(uid, instrument, ":x", revision)
        y, y_path = self._get_signal(uid, instrument, ":y", revision)
        z, z_path = self._get_signal(uid, instrument, ":z", revision)
        R, R_path = self._get_signal(uid, instrument, ":R", revision)

        try:
            location, location_path = self._get_signal(
                uid, instrument, ".geometry:location", revision
            )
            direction, direction_path = self._get_signal(
                uid, instrument, ".geometry:direction", revision
            )
            if len(np.shape(location)) == 1:
                location = np.array([location])
                direction = np.array([direction])

            # TODO: temporary fix until geometry sorted
            if location.shape[0] != x.shape[0]:
                if self.pulse > 10200:
                    index = np.arange(18, 36)
                else:
                    index = np.arange(21, 36)
                location = location[index]
                direction = direction[index]

        except TreeNNF:
            location = None
            direction = None

        for q in quantities:
            qval, q_path = self._get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q],
                revision,
            )
            qval_err, q_path_err = self._get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q] + "_err",
                revision,
            )

            dimensions, _ = self._get_signal_dims(q_path, len(qval.shape))

            results[q + "_records"] = q_path
            results[q] = qval
            results[f"{q}_error"] = qval_err

        results["length"] = len(x)
        results["x"] = x
        results["y"] = y
        results["z"] = z
        results["R"] = R
        results["times"] = times
        results["texp"] = texp
        results["element"] = ""
        results["location"] = location
        results["direction"] = direction

        return results

    def _get_diode_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        # TODO: change once new MDS+ standardisation has been completed
        try:
            location, location_path = self._get_signal(
                uid, instrument, ".geometry:location", revision
            )
            direction, position_path = self._get_signal(
                uid, instrument, ".geometry:direction", revision
            )
        except TreeNNF:
            location = np.array(
                [
                    [1.0, 0, 0],
                ]
            )
            direction = (
                np.array(
                    [
                        [0.17, 0, 0],
                    ]
                )
                - location
            )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        length = location[:, 0].size
        if instrument == "brems":
            _instrument = "lines"
            revision = -1
        elif instrument == "halpha":
            _instrument = "lines"
            revision = -1
        elif "sxr_diode" in instrument:
            _instrument = "diode_detr"
        else:
            raise ValueError(f"{instrument} not yet supported")

        results: Dict[str, Any] = {
            "length": length,
            "machine_dims": self.MACHINE_DIMS,
        }
        results["location"] = location
        results["direction"] = direction
        results["revision"] = self._get_revision(uid, _instrument, revision)
        results["revision"] = revision
        revision = results["revision"]

        quantity = "brightness"
        qval, q_path = self._get_signal(
            uid, _instrument, self.QUANTITIES_MDS[instrument][quantity], revision
        )
        times, _ = self._get_signal_dims(q_path, len(qval.shape))
        times = times[0]
        results["times"] = times
        results[quantity + "_records"] = q_path
        results[quantity] = qval
        try:
            qval_err, q_path_err = self._get_signal(
                uid,
                _instrument,
                self.QUANTITIES_MDS[instrument][quantity] + "_ERR",
                revision,
            )
        except TreeNNF:
            qval_err = 0.0 * results[quantity]
            q_path_err = ""
        results[quantity + "_error"] = qval_err
        results[quantity + "_error" + "_records"] = q_path_err

        return results

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self._get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self._get_signal(
            uid, instrument, ".geometry:direction", revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        times, _ = self._get_signal(uid, instrument, ":time", revision)

        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )

            if "times" not in results:
                results["times"] = times
            results[q + "_records"] = q_path
            results[q] = qval

            try:
                qval_err, q_path_err = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.zeros_like(qval)
                q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

            try:
                qval_syserr, q_path_syserr = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_syserr",
                    revision,
                )
                results[q + "_error"] = np.sqrt(qval_err**2 + qval_syserr**2)
                results[q + "_error" + "_records"] = [q_path_err, q_path_err]
            except TreeNNF:
                results[q + "_error"] = results[q + "_error"]

        length = location[:, 0].size
        results["length"] = length
        results["location"] = np.array(location)
        results["direction"] = np.array(direction)

        return results

    def _get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for electron temperature or number density
        calculated from Thomson scattering.

        """

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        times, times_path = self._get_signal(uid, instrument, ":time", revision)
        # location, location_path = self._get_signal(
        #     uid, instrument, ".geometry:location", revision
        # )
        # direction, direction_path = self._get_signal(
        #     uid, instrument, ".geometry:direction", revision
        # )
        x, x_path = self._get_signal(uid, instrument, ":x", revision)
        y, y_path = self._get_signal(uid, instrument, ":y", revision)
        z, z_path = self._get_signal(uid, instrument, ":z", revision)
        R, R_path = self._get_signal(uid, instrument, ":R", revision)

        for q in quantities:
            qval, q_path = self._get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q],
                revision,
            )
            try:
                qval_err, q_path_err = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.full_like(qval, 0.0)

            dimensions, _ = self._get_signal_dims(q_path, len(qval.shape))

            results[q + "_records"] = q_path
            results[q] = qval
            results[f"{q}_error"] = qval_err

        results["length"] = len(x)
        results["x"] = x
        results["y"] = y
        results["z"] = z
        results["R"] = R
        results["times"] = times
        results["element"] = ""
        # results["location"] = location
        # results["direction"] = direction

        return results

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
        # try:
        #     self._client.list("/")
        #     return False
        # except AuthenticationFailed:
        #     return True
        #
        return False

    def mdsCheck(self, mds_path):
        """Return FAILED if node doesn't exist or other error
        Return FAILED if: lenght(data)==1 and data==nan"""

        mds_path_test = (
            f"_dummy = IF_ERROR (IF ((SIZE ({mds_path})==1), "
            + f'IF ({mds_path}+1>{mds_path}, {mds_path}, "FAILED"),'
            + f' {mds_path}), "FAILED")'
        )

        return mds_path_test

    def get_revision_name(self, revision) -> str:
        """Return string defining RUN## or BEST if revision = 0"""

        if type(revision) == int:
            if revision < 0:
                rev_str = ""
            elif revision == 0:
                rev_str = ".best"
            elif revision < 10:
                rev_str = f".run0{int(revision)}"
            elif revision > 9:
                rev_str = f".run{int(revision)}"
        else:
            rev_str = f".{revision}"

        return rev_str

    def get_los(self, position, direction):
        """
        Return start and stop (x, y, z) of line-of-sight given position and direction

        Parameters
        ----------
        position
            (x, y, z) of LOS starting point
        direction
            (delta_x, delta_y, delta_y) direction of LOS

        Returns
        -------
            (x, y, z) of start and stop

        """

        xstart, ystart, zstart = position
        xstop, ystop, zstop = position + direction

        return (xstart, ystart, zstart), (xstop, ystop, zstop)
