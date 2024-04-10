"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""

from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple



from .st40conf import ST40Conf

from .mdsutils import MDSUtils

from MDSplus import Connection
from MDSplus.mdsExceptions import TreeNNF
from MDSplus.mdsExceptions import TreeNODATA
import numpy as np

from .abstractreader import DataReader
from .. import session
from ..numpy_typing import RevisionLike


class ST40Reader(DataReader):
    """Class to read ST40 MDS+ data using MDSplus.

    Parameters
    ----------
    time : np.ndarray
        An ordered array of time to which data will be
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

   
    def __init__(
        self,
        pulse: int,
        tstart: float,
        tend: float,
        server: str = "smaug",
        tree: str = "ST40",
        default_error: float = 0.05,
        session: session.Session = session.global_session,
    ):
        self._reader_cache_id = f"st40:{server.replace('-', '_')}:{pulse}"
        self.NAMESPACE: Tuple[str, str] = ("st40", server)
        super().__init__(
            tstart,
            tend,
            session,
            pulse=pulse,
            server=server,
            default_error=default_error,
        )
        self.pulse: int = pulse

        self._default_error = default_error
        self.mdsutils=MDSUtils(server,tree,pulse)


        #ST40 configurations. ACtually should modify the code not to set as self..
        self.st40conf = ST40Conf()
        self.MACHINE_DIMS=self.st40conf.MACHINE_DIMS
        self.INSTRUMENT_METHODS=self.st40conf.INSTRUMENT_METHODS
        self.UIDS_MDS=self.st40conf.UIDS_MDS
        self.QUANTITIES_MDS=self.st40conf.QUANTITIES_MDS




    def _get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array], str, str]:
        """Gets the signal and its coordinates for the given INSTRUMENT, at the
        given revision."""
        data, _path = self.mdsutils.get_signal(uid, instrument, quantity, revision)
        dims, _ = self.mdsutils.get_signal_dims(_path, len(data.shape))
        unit = self.mdsutils.get_signal_units(_path)

        return data, dims, unit, _path
    

    
    def _get_revision(
        self, uid: str, instrument: str, revision: RevisionLike
    ) -> RevisionLike:
        """
        Gets the effective revision name if latest/best is given in input
        """
        if type(revision) == str:
            return revision

        if revision == 0:
            run_name, _ = self.mdsutils.get_signal(uid, instrument, ":best_run", revision)
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
        times, _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        results["times"] = times
        results["psin"], results["psin_records"] = self.mdsutils.get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        results["psi_r"], results["psi_r_records"] = self.mdsutils.get_signal(
            uid, instrument, ".psi2d:rgrid", revision
        )
        results["psi_z"], results["psi_z_records"] = self.mdsutils.get_signal(
            uid, instrument, ".psi2d:zgrid", revision
        )
        for q in quantities:
            if q not in self.QUANTITIES_MDS[instrument].keys():
                continue
            try:
                qval, q_path = self.mdsutils.get_signal(
                    uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
                )
            except TreeNNF:
                continue

            if q == "psi":
                results["psi"] = qval.reshape(
                    (
                        len(results["t"]),
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
        results["boundary_index"], _ = self.mdsutils.get_signal(
            uid, instrument, ".p_boundary:index", revision
        )
        results["psi"], _ = self.mdsutils.get_signal(
            uid, instrument, ".profiles.psi_norm:psi", revision
        )
        results["psin"], psin_path = self.mdsutils.get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        results["ftor"], _ = self.mdsutils.get_signal(
            uid, instrument, ".profiles.psi_norm:ftor", revision
        )
        results["rho"], rho_path = self.mdsutils.get_signal(
            uid, instrument, ".profiles.astra:rho", revision
        )
        results["psi_r"], _ = self.mdsutils.get_signal(
            uid, instrument, ".psi2d:rgrid", revision
        )
        results["psi_z"], _ = self.mdsutils.get_signal(
            uid, instrument, ".psi2d:zgrid", revision
        )
        results["times"], t_path = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        for q in quantities:
            qval, q_path = self.mdsutils.get_signal(
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

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:direction", revision
        )

        quantity = "brightness"
        times, times_path = self.mdsutils.get_signal(
            uid,
            instrument,
            ":time",
            revision,
        )
        qval, qval_record = self.mdsutils.get_signal(
            uid,
            instrument,
            self.QUANTITIES_MDS[instrument][quantity],
            revision,
        )
        qerr, qerr_record = self.mdsutils.get_signal(
            uid,
            instrument,
            self.QUANTITIES_MDS[instrument][quantity] + "_err",
            revision,
        )

        results["length"] = np.shape(qval)[1]
        results["t"] = time
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

        location, location_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:direction", revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        results["times"], _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        wavelength, _ = self.mdsutils.get_signal(uid, instrument, ":wavelength", revision)
        # TODO: change once wavelength in MDS+ has been fixed to nanometers!
        wavelength /= 10.0
        if self.pulse >= 10307:
            dlambda = float(np.abs(wavelength[1] - wavelength[0])) * 4
            wavelength += dlambda
        results["wavelength"] = wavelength

        for q in quantities:
            qval, q_path = self.mdsutils.get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            results[q + "_records"] = q_path
            results[q] = qval

            try:
                qval_err, q_path_err = self.mdsutils.get_signal(
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

        texp, texp_path = self.mdsutils.get_signal(uid, instrument, ":exposure", revision)
        times, _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        try:
            wavelength, _ = self.mdsutils.get_signal(uid, instrument, ":wavelen", revision)
        except TreeNODATA:
            wavelength = None

        x, x_path = self.mdsutils.get_signal(uid, instrument, ":x", revision)
        y, y_path = self.mdsutils.get_signal(uid, instrument, ":y", revision)
        z, z_path = self.mdsutils.get_signal(uid, instrument, ":z", revision)
        R, R_path = self.mdsutils.get_signal(uid, instrument, ":R", revision)

        # TODO: temporary fix until geometry sorted (especially pulse if statement..)
        try:
            location, location_path = self.mdsutils.get_signal(
                uid, instrument, ".geometry:location", revision
            )
            direction, direction_path = self.mdsutils.get_signal(
                uid, instrument, ".geometry:direction", revision
            )
            if len(np.shape(location)) == 1:
                location = np.array([location])
                direction = np.array([direction])

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
            try:
                qval, q_path = self.mdsutils.get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q],
                    revision,
                )
            except TreeNODATA:
                continue

            try:
                qval_err, q_path_err = self.mdsutils.get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.full_like(qval, 0.0)

            dimensions, _ = self.mdsutils.get_signal_dims(q_path, len(qval.shape))
            results[q + "_records"] = q_path
            results[q] = qval
            results[f"{q}_error"] = qval_err

        results["length"] = len(x)
        results["x"] = x
        results["y"] = y
        results["z"] = z
        results["R"] = R
        results["t"] = time
        results["texp"] = texp
        results["element"] = ""
        # TODO: check whether wlength should be channel agnostic or not...
        if wavelength is not None:
            results["wavelength"] = wavelength[0, :]
        results["location"] = location
        results["direction"] = direction

        return results

    def _get_spectrometer(
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

        times, _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        wavelength, _ = self.mdsutils.get_signal(uid, instrument, ":wavelen", revision)

        location, location_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:direction", revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        for q in quantities:
            qval, q_path = self.mdsutils.get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q],
                revision,
            )

            try:
                qval_err, q_path_err = self.mdsutils.get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.full_like(qval, 0.0)

            dimensions, _ = self.mdsutils.get_signal_dims(q_path, len(qval.shape))

            results[q + "_records"] = q_path
            results[q] = qval
            results[f"{q}_error"] = qval_err

        results["length"] = location[:, 0].size
        results["t"] = time
        # TODO: check whether wlength should be channel agnostic or not...
        if wavelength is not None:
            results["wavelength"] = wavelength[0, :]
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
        """
        TODO: labels are np.bytes_ type...is this correct?
        """
        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        location, location_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, position_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:direction", revision
        )
        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        length = location[:, 0].size

        results: Dict[str, Any] = {
            "length": length,
            "machine_dims": self.MACHINE_DIMS,
        }
        results["location"] = location
        results["direction"] = direction
        results["revision"] = self._get_revision(uid, instrument, revision)
        results["revision"] = revision
        revision = results["revision"]

        quantity = "brightness"
        qval, q_path = self.mdsutils.get_signal(
            uid, instrument, self.QUANTITIES_MDS[instrument][quantity], revision
        )
        times, _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        _labels, _ = self.mdsutils.get_signal(uid, instrument, ":label", revision)
        if type(_labels[0]) == np.bytes_:
            labels = np.array([label.decode("UTF-8") for label in _labels])
        else:
            labels = _labels

        results["t"] = time
        results["labels"] = labels
        results[quantity + "_records"] = q_path
        results[quantity] = qval
        try:
            qval_err, q_path_err = self.mdsutils.get_signal(
                uid,
                instrument,
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
        """
        TODO: SMMH 2023 launcher/receiver cross plasma on different poloidal paths!
        Currently setting location and direction as average of the two!!!!
        """

        if len(uid) == 0 and instrument in self.UIDS_MDS:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:location", revision
        )
        direction, direction_path = self.mdsutils.get_signal(
            uid, instrument, ".geometry:direction", revision
        )

        if instrument == "smmh":
            location_r, _ = self.mdsutils.get_signal(
                uid, instrument, ".geometry:location_r", revision
            )
            direction_r, _ = self.mdsutils.get_signal(
                uid, instrument, ".geometry:direction_r", revision
            )
            location = (location + location_r) / 2.0
            direction = (direction + direction_r) / 2.0

        if len(np.shape(location)) == 1:
            location = np.array([location])
            direction = np.array([direction])

        times, _ = self.mdsutils.get_signal(uid, instrument, ":time", revision)

        for q in quantities:
            qval, q_path = self.mdsutils.get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )

            if "t" not in results:
                results["t"] = time
            results[q + "_records"] = q_path
            results[q] = qval

            try:
                qval_err, q_path_err = self.mdsutils.get_signal(
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
                qval_syserr, q_path_syserr = self.mdsutils.get_signal(
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

        times, times_path = self.mdsutils.get_signal(uid, instrument, ":time", revision)
        print("\n Hardcoded correction to TS coordinates to be fixed in MDS+ \n")
        # location, location_path = self.mdsutils.get_signal(
        #     uid, instrument, ".geometry:location", revision
        # )
        # direction, direction_path = self.mdsutils.get_signal(
        #     uid, instrument, ".geometry:direction", revision
        # )
        x, x_path = self.mdsutils.get_signal(uid, instrument, ":x", revision)
        y, y_path = self.mdsutils.get_signal(uid, instrument, ":y", revision)
        z, z_path = self.mdsutils.get_signal(uid, instrument, ":z", revision)
        R, R_path = self.mdsutils.get_signal(uid, instrument, ":R", revision)
        z = R * 0.0
        x = R
        y = R * 0.0

        for q in quantities:
            qval, q_path = self.mdsutils.get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q],
                revision,
            )
            try:
                qval_err, q_path_err = self.mdsutils.get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
            except TreeNNF:
                qval_err = np.full_like(qval, 0.0)

            dimensions, _ = self.mdsutils.get_signal_dims(q_path, len(qval.shape))

            results[q + "_records"] = q_path
            results[q] = qval
            results[f"{q}_error"] = qval_err

        results["length"] = len(x)
        results["x"] = x
        results["y"] = y
        results["z"] = z
        results["R"] = R
        results["t"] = time
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



    def get_revision_name(self, revision) -> str:
        """Return string defining RUN## or BEST if revision = 0"""

        if type(revision) == int:
            rev_str = ""
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
