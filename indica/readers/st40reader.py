"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""

import re
from numbers import Number
from pathlib import Path
import pickle
import stat
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union
import warnings
from xarray import DataArray

import numpy as np
from MDSplus import Connection
import scipy.constants as sc

from .abstractreader import CACHE_DIR
from .abstractreader import DataReader
from .abstractreader import DataSelector
from .selectors import choose_on_plot
from .. import session
from ..datatypes import ELEMENTS_BY_MASS
from ..utilities import to_filename


# SURF_PATH = Path(surf_los.__file__).parent / "surf_los.dat"


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

    MACHINE_DIMS = ((0.15, 0.8), (-0.8, 0.8))
    INSTRUMENT_METHODS = {
        "efit": "get_equilibrium",
        "xrcs": "get_helike_spectroscopy",
        "nirh1": "get_interferometry",
        "smmh1": "get_interferometry",
        "astra": "get_astra",
    }
    UIDS_MDS = {
        "efit": "",
        "xrcs": "sxr",
        "nirh1": "interferom",
        "smmh1": "interferom",
        "astra": "",
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
            "rmag": ".global:rmag",
            "rgeo": ".global:rgeo",
            "rbnd": ".p_boundary:rbnd",
            "zmag": ".global:zmag",
            "zbnd": ".p_boundary:zbnd",
            "ipla": ".constraints.ip:cvalue",
            "wp": ".virial:wp",
        },
        "xrcs": {
            "te_kw": ".te_kw:te",
            "te_n3w": ".te_n3w:te",
            "ti_w": ".ti_w:ti",
            "ti_z": ".ti_z:ti",
        },
        "nirh1": {
            "ne": ".line_int.ne",
        },
        "smmh1": {
            "ne": ".line_int.ne",
        },
        "astra": {
            "cc": ".profiles.astra:cc",  # Parallel current conductivity, 1/(Ohm*m)
            "chi_e": ".profiles.astra:chi_e",  # Total electron heat conductivity, m^2/s
            "chi_e_anom": ".profiles.astra:chi_e_anom",  # anomalous electron heat conductivity, m^2/s
            "chi_e_neo": ".profiles.astra:chi_e_neo",  # neoclassical electron heat conductivity, m^2/s
            "chi_i": ".profiles.astra:chi_i",  # Total ion heat conductivity, m^2/s
            "chi_i_anom": ".profiles.astra:chi_i_anom",  # anomalous ion heat conductivity, m^2/s
            "chi_i_neo": ".profiles.astra:chi_i_neo",  # neoclassical ion heat conductivity, m^2/s
            "chi_phi": ".profiles.astra:chi_phi",  # Momentum transport coefficient, m2/s
            "cn": ".profiles.astra:cn",  # Particle pinch velocity , m/s
            "diff": ".profiles.astra:diff",  # diffusion coefficient, m^2/s
            "elon": ".profiles.astra:elon",  # Elongation profile
            "j_bs": ".profiles.astra:j_bs",  # Bootstrap current density,MA/m2
            "j_nbi": ".profiles.astra:j_nbi",  # NB driven current density,MA/m2
            "j_oh": ".profiles.astra:j_oh",  # Ohmic current density,MA/m2
            "j_rf": ".profiles.astra:j_rf",  # EC driven current density,MA/m2
            "j_tot": ".profiles.astra:j_tot",  # Total current density,MA/m2
            "ne": ".profiles.astra:ne",  # Electron density, 10^19 m^-3
            "ni": ".profiles.astra:ni",  # Main ion density, 10^19 m^-3
            "n_d": ".profiles.astra:n_d",  # Deuterium density,10E19/m3
            "n_t": ".profiles.astra:n_t",  # Tritium density	,10E19/m3
            "omega_tor": ".profiles.astra:omega_tor",  # Toroidal rotation frequency, 1/s
            "pegn": ".profiles.astra:pegn",  # electron convective heat flux, MW
            "pign": ".profiles.astra:pign",  # ion convective heat flux, MW
            "psin": ".profiles.astra:psin",  # Normalized poloidal flux -
            "qe": ".profiles.astra:qe",  # electron power flux, MW
            "qi": ".profiles.astra:qi",  # ion power flux, MW
            "qn": ".profiles.astra:qn",  # total electron flux, 10^19/s
            "qnbe": ".profiles.astra:qnbe",  # Beam power density to electrons, MW/m3
            "qnbi": ".profiles.astra:qnbi",  # Beam power density to ions, MW/m3
            "q_alpha_e": ".profiles.astra:q_alpha_e",  # Alpha power density to electrons,MW/m3
            "q_alpha_i": ".profiles.astra:q_alpha_i",  # Alpha power density to ions,MW/m3
            "q_oh": ".profiles.astra:q_oh",  # Ohmic heating power profile, MW/m3
            "q_rf": ".profiles.astra:q_rf",  # RF power density to electron,MW/m3
            "rmid": ".profiles.astra:rmid",  # Centre of flux surfaces, m
            "rminor": ".profiles.astra:rminor",  # minor radius, m
            "sbm": ".profiles.astra:sbm",  # Particle source from beam, 10^19/m^3/s
            "spel": ".profiles.astra:spel",  # Particle source from pellets, 10^19/m^3/s
            "stot": ".profiles.astra:stot",  # Total electron source,10^19/s/m3
            "swall": ".profiles.astra:swall",  # Particle source from wall neutrals, 10^19/m^3/s
            "te": ".profiles.astra:te",  # Electron temperature, keV
            "ti": ".profiles.astra:ti",  # Ion temperature, keV
            "torq_den": ".profiles.astra:torq_den",  # Total torque density from NB, N*m/m3
            "torq_den_bcx": ".profiles.astra:torq_den_bcx",  # CX losses torque density from NB, N*m/m3
            "torq_den_be": ".profiles.astra:torq_den_be",  # Collisional to electron torque density from NB, N*m/m3
            "torq_den_bi": ".profiles.astra:torq_den_bi",  # Collisional to ions torque density from NB, N*m/m3
            "torq_den_bth": ".profiles.astra:torq_den_bth",  # Beam thermalisation torque density from NB, N*m/m3
            "torq_den_jxb": ".profiles.astra:torq_den_jxb",  # JXB torque density from NB, N*m/m3
            "tri": ".profiles.astra:tri",  # Triangularity (up/down symmetrized) profile
            "t_d": ".profiles.astra:t_d",  # Deuterium temperature,keV
            "t_t": ".profiles.astra:t_t",  # Tritium temperature,keV
            "zeff": ".profiles.astra:zeff",  # Effective ion charge
            "areat": ".profiles.psi_norm:areat",  # Toroidal cross section,m2
            "ffprime": ".profiles.psi_norm:ffprime",  # FFPRIME
            "ftor": ".profiles.psi_norm:ftor",  # Toroidal flux, Wb
            "p": ".profiles.psi_norm:p",  # PRESSURE(PSI_NORM)
            "pprime": ".profiles.psi_norm:pprime",  # PPRIME
            "psi": ".profiles.psi_norm:psi",  # PSI
            "q": ".profiles.psi_norm:q",  # Q_PROFILE(PSI_NORM)
            "sigmapar": ".profiles.psi_norm:sigmapar",  # Parallel conductivity,1/(Ohm*m)
            "volume": ".profiles.psi_norm:volume",  # Volume inside magnetic surface,m3
        },
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
        selector: DataSelector = choose_on_plot,
        session: session.Session = session.global_session,
    ):
        self._reader_cache_id = f"st40:{server.replace('-', '_')}:{pulse}"
        self.NAMESPACE: Tuple[str, str] = ("st40", server)
        super().__init__(
            tstart,
            tend,
            max_freq,
            session,
            selector,
            pulse=pulse,
            server=server,
            default_error=default_error,
        )
        self.pulse = pulse
        self.tree = tree
        self.conn = Connection(server)
        self.conn.openTree(self.tree, self.pulse)
        self._default_error = default_error

    def get_mds_path(
        self, uid: str, instrument: str, quantity: str, revision: int
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

    def _get_data(
        self, uid: str, instrument: str, quantity: str, revision: int
    ) -> Tuple[np.array, List[np.array]]:
        """Gets the signal and its coordinates for the given INSTRUMENT, at the
        given revision."""
        data, _path = self._get_signal(uid, instrument, quantity, revision)
        dims, _ = self._get_signal_dims(_path, len(data.shape))

        return data, dims

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: int
    ) -> Tuple[np.array, str]:
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""
        path, path_check = self.get_mds_path(uid, instrument, quantity, revision)
        if quantity.lower() == ":best_run":
            data = str(self.conn.get(path))
        else:
            data = np.array(self.conn.get(path_check))

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

    def _get_revision(self, uid: str, instrument: str, revision: int) -> int:
        """
        Gets the effective revision name if latest/best is given in input
        """
        if revision == 0:
            run_name, _ = self._get_signal(uid, instrument, ":best_run", revision)
            m = re.search(r"\s??RUN(\d+)", run_name, re.I)
            if isinstance(m, re.Match):
                revision = int(m.group(1))

        return revision

    def _read_cached_ppf(self, path: Path) -> Optional[np.array]:
        """Check if the PPF data specified by `sal_path` has been cached and,
        if so, load it.

        """
        if not path.exists():
            return None
        permissions = stat.filemode(path.stat().st_mode)
        if permissions[5] == "w" or permissions[8] == "w":
            warnings.warn(
                "Can not open cache file which is writeable by anyone other than "
                "the user. (Security risk.)",
                PPFWarning,
            )
            return None
        with path.open("rb") as f:
            try:
                return pickle.load(f)
            except pickle.UnpicklingError:
                warnings.warn(
                    f"Error unpickling cache file {path}. (Possible data corruption.)",
                    PPFWarning,
                )
                return None

    def _write_cached_ppf(self, path: Path, data: np.array):
        """Write the given signal, fetched from `sal_path`, to the disk for
        later reuse.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(data, f)
        path.chmod(0o644)

    def _sal_path_to_file(self, sal_path: str) -> Path:
        """Get the file path which would be used to cache data from the given
        `sal_path`.

        """
        return (
            Path.home()
            / CACHE_DIR
            / self.__class__.__name__
            / to_filename(self._reader_cache_id + sal_path + ".pkl")
        )

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: int,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for plasma equilibrium."""

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)

        times, _ = self._get_signal(uid, instrument, ":time", revision)
        if np.array_equal(times, "FAILED"):
            return {}

        psin, _ = self._get_signal(uid, instrument, ".profiles.psi_norm:xpsn", revision)
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            self._set_times_item(results, times)
            if (
                len(qval.shape) > 1
                and q not in {"psi", "rbnd", "zbnd"}
                and "psin" not in results
            ):
                results["psin"] = psin
            if q == "psi":
                r, r_path = self._get_signal(uid, instrument, ".psi2d:rgrid", revision)
                z, z_path = self._get_signal(uid, instrument, ".psi2d:zgrid", revision)
                results["psi_r"] = r
                results["psi_z"] = z
                results["psi"] = qval.reshape((len(results["times"]), len(z), len(r)))
                results["psi_records"] = [q_path, r_path, z_path]
            else:
                results[q] = qval
                results[q + "_records"] = [q_path]

        return results

    def _get_astra(
        self,
        uid: str,
        instrument: str,
        revision: int,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from ASTRA run."""

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)

        # Read time and radial dimensions
        psin, psin_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        ftor, rho_path = self._get_signal(
            uid, instrument, ".profiles.astra:rho", revision
        )
        times, t_path = self._get_signal(uid, instrument, ":time", revision)
        results["psin"] = psin
        results["ftor"] = ftor
        self._set_times_item(results, times)
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

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: int,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        # position_instrument = "raw_sxr"
        # position, position_path = self._get_signal(uid, position_instrument, ".xrcs.geometry:position", -1)
        # direction, position_path = self._get_signal(uid, position_instrument, ".xrcs.geometry:direction", -1)
        if instrument == "xrcs":
            position = np.array([1.0, 0, 0])
            direction = np.array([0.175, 0, 0]) - position
        else:
            raise ValueError(f"No geometry available for {instrument}")
        los_start, los_end = self.get_los(position, direction)
        times, _ = self._get_signal(uid, instrument, ":time_mid", revision)
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            times, _ = self._get_signal_dims(q_path, len(qval.shape))
            times = times[0]
            if "times" not in results:
                results["times"] = times
            results[q + "_records"] = q_path
            results[q] = qval
            qval_err, q_path_err = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q] + "_ERR", revision
            )
            if np.array_equal(qval_err, "FAILED"):
                qval_err = 0.0 * results[q]
                q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

        results["length"] = 1
        results["Rstart"] = np.array([los_start[0]])
        results["Rstop"] = np.array([los_end[0]])
        results["zstart"] = np.array([los_start[1]])
        results["zstop"] = np.array([los_end[1]])
        results["Tstart"] = np.array([los_start[2]])
        results["Tstop"] = np.array([los_end[2]])

        return results

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: int,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)

        # position_instrument = ""
        # position, position_path = self._get_signal(uid, position_instrument, "..geometry:position", -1)
        # direction, position_path = self._get_signal(uid, position_instrument, "..geometry:direction", -1)
        if instrument == "nirh1":
            position = np.array([0.380, -0.925, 0])
            direction = np.array([0.0, 1.0, 0.0]) - position
        elif instrument == "smmh1":
            position = np.array([1.0, 0, 0])
            direction = np.array([0.175, 0, 0]) - position
        else:
            raise ValueError(f"No geometry available for {instrument}")
        los_start, los_end = self.get_los(position, direction)
        times, _ = self._get_signal(uid, instrument, ":time", revision)

        if np.array_equal(times, "FAILED"):
            return {}

        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )

            if "times" not in results:
                results["times"] = times
            results[q + "_records"] = q_path
            results[q] = qval
            qval_err = np.zeros_like(qval)
            q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

        results["length"] = 1
        results["Rstart"] = np.array([los_start[0]])
        results["Rstop"] = np.array([los_end[0]])
        results["zstart"] = np.array([los_start[1]])
        results["zstop"] = np.array([los_end[1]])
        results["Tstart"] = np.array([los_start[2]])
        results["Tstop"] = np.array([los_end[2]])

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
            Your single sign-on password.

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

    def mdsCheck(self, mds_path):
        """Return FAILED if node doesn't exist or other error
        Return FAILED if: lenght(data)==1 and data==nan"""

        mds_path_test = (
            f"_dummy = IF_ERROR (IF ((SIZE ({mds_path})==1), "
            + f'IF ({mds_path}+1>{mds_path}, {mds_path}, "FAILED"),'
            + f' {mds_path}), "FAILED")'
        )

        return mds_path_test

    def get_revision_name(self, revision):
        """Return string defining RUN## or BEST if revision = 0"""

        if revision < 0:
            rev_str = ""
        elif revision == 0:
            rev_str = ".best"
        elif revision < 10:
            rev_str = f".run0{int(revision)}"
        elif revision > 9:
            rev_str = f".run{int(revision)}"

        return rev_str

    def get_los(self, position, direction):
        """
        Return (R, z, T) of start and end of line-of-sight given position and direction

        Parameters
        ----------
        position
            (R, z, T) of LOS starting point
        direction
            (delta_R, delta_z, delta_T) direction of LOS

        Returns
        -------
            start and end (R, z, T)

        """

        x0, y0, z0 = position
        x1, y1, z1 = position + direction

        Rstart = x0  # np.sqrt(x0 ** 2 + y0 ** 2)
        Rstop = x1  # np.sqrt(x1 ** 2 + y1 ** 2)
        zstart = z0
        zstop = z1
        Tstart = y0
        Tstop = y1

        return (Rstart, zstart, Tstart), (Rstop, zstop, Tstop)
