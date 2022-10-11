"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""

import re
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from MDSplus import Connection
from MDSplus.mdsExceptions import TreeNNF
import numpy as np

from .abstractreader import DataReader
from .abstractreader import DataSelector
from .selectors import choose_on_plot
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

    MACHINE_DIMS = ((0.17, 0.8), (-0.75, 0.75))
    INSTRUMENT_METHODS = {
        "efit": "get_equilibrium",
        "xrcs": "get_helike_spectroscopy",
        "princeton": "get_charge_exchange",
        "lines": "get_filters",
        "nirh1": "get_interferometry",
        "nirh1_bin": "get_interferometry",
        "smmh1": "get_interferometry",
        "astra": "get_astra",
        "diode_arrays": "get_radiation",
    }
    UIDS_MDS = {
        "efit": "",
        "xrcs": "sxr",
        "princeton": "spectrom",
        "lines": "spectrom",
        "nirh1": "interferom",
        "nirh1_bin": "interferom",
        "smmh1": "interferom",
        "astra": "",
        "diode_arrays": "sxr",
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
            "psin": ".profiles.psi_norm:xpsn",
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
        "princeton": {  # change to angf
            "int": ".int",
            "int_error": ".int_err",
            "ti": ".ti",
            "ti_error": ".ti_err",
            "vtor": ".vtor",
            "vtor_error": ".vtor_err",
            "times": ".time",
            "exposure": ".exposure",
        },
        "lines": {
            "brems": ".brem_mp1:intensity",
            "h_alpha": ".h_alpha_mp1:intensity",
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
        "astra": {
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
            "psin": ".profiles.psi_norm:psin",  # Normalized poloidal flux -
            "areat": ".profiles.psi_norm:areat",  # Toroidal cross section,m2
            "ftor": ".profiles.psi_norm:ftor",  # Toroidal flux, Wb
            "p": ".profiles.psi_norm:p",  # PRESSURE(PSI_NORM)
            "pblon": ".profiles.astra:pblon",  # PRESSURE(PSI_NORM)
            "pbper": ".profiles.astra:pbper",  # PRESSURE(PSI_NORM)
            "pnb": ".global:pnb",  # Injected NBI power, W
            "pabs": ".global:pabs",  # Absorber NBI power, W
            "p_oh": ".global:p_oh",  # Absorber NBI power, W
            "psi": ".profiles.psi_norm:psi",  # PSI
            "q": ".profiles.psi_norm:q",  # Q_PROFILE(PSI_NORM)
            "sigmapar": ".profiles.psi_norm:sigmapar",  # Paral. conduct.,1/(Ohm*m)
            "volume": ".profiles.psi_norm:volume",  # Volume inside magnetic surface,m3
        },
        "diode_arrays": {  # GETTING THE DATA OF THE SXR CAMERA
            "filter_1": ".middle_head.filter_1:",
            "filter_1_time": ".middle_head.filter_1:time",
            "filter_2": ".middle_head.filter_2:",
            "filter_2_time": ".middle_head.filter_2:time",
            "filter_3": ".middle_head.filter_3:",
            "filter_3_time": ".middle_head.filter_3:time",
            "filter_4": ".middle_head.filter_4:",
            "filter_4_time": ".middle_head.filter_4:time",
            "location": ".middle_head.geometry:location",
            "direction": ".middle_head.geometry:direction",
        },
    }

    _IMPLEMENTATION_QUANTITIES = {
        "diode_arrays": {  # GETTING THE DATA OF THE SXR CAMERA
            "filter_1": ("sxr_radiation", "no_filter"),
            "filter_2": ("sxr_radiation", "50_Al"),
            "filter_3": ("sxr_radiation", "250_Be"),
            "filter_4": ("sxr_radiation", "10_Be"),
        },
    }

    _RADIATION_RANGES = {
        "filter_1": (1, 20),
        "filter_2": (21, 40),
        "filter_3": (41, 60),
        "filter_4": (61, 80),
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

    def _conn_get(self, mds_path):
        """Gets the signal for the given INSTRUMENT, at the
        given revision."""

        mds_data = self.conn.get(mds_path)
        return mds_data

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
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
            m = re.search(r"\s??RUN(\d+)", run_name, re.I)
            if isinstance(m, re.Match):
                revision = int(m.group(1))

        return revision

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch raw data for plasma equilibrium."""

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        times, _ = self._get_signal(uid, instrument, ":time", revision)
        if np.array_equal(times, "FAILED"):
            return {}

        qval, q_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        results["psin"] = qval
        results["psin_records"] = [q_path]
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            self._set_times_item(results, times)
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
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from ASTRA run."""

        results: Dict[str, Any] = {}
        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # Read time and radial dimensions
        psi, psin_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:psi", revision
        )
        psin, psin_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:xpsn", revision
        )
        ftor, ftor_path = self._get_signal(
            uid, instrument, ".profiles.psi_norm:ftor", revision
        )
        rho, rho_path = self._get_signal(
            uid, instrument, ".profiles.astra:rho", revision
        )
        # psi, rho_path = self._get_signal(
        #     uid, instrument, ".profiles.astra:psi", revision
        # )
        times, t_path = self._get_signal(uid, instrument, ":time", revision)
        results["psi"] = psi
        results["psin"] = psin
        results["ftor"] = ftor
        results["rho"] = rho
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

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:
        """Fetch data from SXR camera."""

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        location, location_path = self._get_signal(
            uid, instrument, self.QUANTITIES_MDS[instrument]["location"], revision
        )
        direction, direction_path = self._get_signal(
            uid, instrument, self.QUANTITIES_MDS[instrument]["direction"], revision
        )

        for q in quantities:
            luminosities = []
            records = []
            xstart, xstop, ystart, ystop, zstart, zstop = [], [], [], [], [], []

            times, t_path = self._get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS["diode_arrays"][q + "_time"],
                revision,
            )
            results[q + "_times"] = times
            records.append(t_path)

            chan_start, chan_end = self._RADIATION_RANGES[q]
            nchan = chan_end - chan_start + 1
            for chan in range(chan_start, chan_end + 1):
                qval, q_path = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "ch" + str(chan).zfill(3),
                    revision,
                )

                records.append(q_path)
                luminosities.append(qval)

                los_start, los_stop = self.get_los(
                    location[chan - chan_start], direction[chan - chan_start]
                )
                xstart.append(los_start[0])
                xstop.append(los_stop[0])
                ystart.append(los_start[1])
                ystop.append(los_stop[1])
                zstart.append(los_start[2])
                zstop.append(los_stop[2])

            results["length"][q] = nchan
            results[q] = np.array(luminosities).T
            results[q + "_records"] = records
            results[q + "_error"] = self._default_error * results[q]

            results[q + "location"] = np.array(location)
            results[q + "direction"] = np.array(direction)
            results[q + "_xstart"] = np.array(xstart)
            results[q + "_xstop"] = np.array(xstop)
            results[q + "_ystart"] = np.array(ystart)
            results[q + "_ystop"] = np.array(ystop)
            results[q + "_zstart"] = np.array(zstart)
            results[q + "_zstop"] = np.array(zstop)

            # results[q + "_extension"] = extension[:, chan_start - 1 : chan_end, :]
        results["quantities"] = quantities

        return results

    def _get_helike_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # TODO: update when new MDS+ structure becomes available
        # position, position_path = self._get_signal(uid, position,
        # ".geometry:position", revision)
        # direction, position_path = self._get_signal(uid, position,
        # ".geometry:direction", revision)
        if instrument == "xrcs":
            location = np.array([1.0, 0, 0])
            direction = np.array([0.17, 0, 0]) - location
        else:
            raise ValueError(f"No geometry available for {instrument}")
        times, _ = self._get_signal(uid, instrument, ":time_mid", revision)
        wavelength, _ = self._get_signal(uid, instrument, ":wavelength", revision)
        results["wavelength"] = wavelength
        for q in quantities:
            qval, q_path = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q], revision
            )
            results[q + "_records"] = q_path
            results[q] = qval
            times, _ = self._get_signal_dims(q_path, len(qval.shape))
            if "times" not in results.keys():
                results["times"] = times[0]

            try:
                qval_err, q_path_err = self._get_signal(
                    uid,
                    instrument,
                    self.QUANTITIES_MDS[instrument][q] + "_err",
                    revision,
                )
                if np.array_equal(qval_err, "FAILED"):
                    qval_err = 0.0 * results[q]
                    q_path_err = ""
            except TreeNNF:
                qval_err = 0.0 * results[q]
                q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

        results["length"] = 1
        results["location"] = np.array(location)
        results["direction"] = np.array(direction)

        return results

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(
            uid, instrument + ".CXSFIT_OUT", revision
        )
        revision = results["revision"]

        # Get Geometry data from mds
        if instrument == "princeton":
            tree_path = ""
            location, location_path = self._get_signal(
                uid, tree_path, ".princeton.passive.best.geometry:location", -1
            )
            direction, direction_path = self._get_signal(
                uid, tree_path, ".princeton.passive.best.geometry:direction", -1
            )
        else:
            raise ValueError(f"No geometry available for {instrument}")
        rstart = np.zeros(np.shape(direction)[0], dtype=float)
        rstop = np.zeros(np.shape(direction)[0], dtype=float)
        zstart = np.zeros(np.shape(direction)[0], dtype=float)
        zstop = np.zeros(np.shape(direction)[0], dtype=float)
        tstart = np.zeros(np.shape(direction)[0], dtype=float)
        tstop = np.zeros(np.shape(direction)[0], dtype=float)
        for i in range(np.shape(direction)[0]):
            los_start, los_end = self.get_los(location, direction[i, :])
            rstart[i] = los_start[0]
            rstop[i] = los_end[0]
            zstart[i] = los_start[2]
            zstop[i] = los_end[2]
            tstart[i] = los_start[1]
            tstop[i] = los_end[1]

        # Doesn't yet work, need to write data to the CXSFIT_OUT trees.
        print("quantities={}".format(quantities))
        for q in quantities:
            qval, q_path = self._get_signal(
                uid,
                instrument + ".CXSFIT_OUT",
                self.QUANTITIES_MDS[instrument][q],
                revision,
            )
            results[q + "_records"] = q_path
            results[q] = qval

        print(results)
        print(results["times"])
        print(results["exposure"])

        # Export coordinates
        results["length"] = len(rstart)
        results["Rstart"] = rstart
        results["Rstop"] = rstop
        results["zstart"] = zstart
        results["zstop"] = zstop
        results["Tstart"] = tstart
        results["Tstop"] = tstop

        return results

    def _get_filters(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # TODO: update when new MDS+ structure becomes available
        # position, position_path = self._get_signal(uid, instrument,
        # ".geometry:position", revision)
        # direction, position_path = self._get_signal(uid, instrument,
        # ".geometry:direction", revision)
        if instrument == "lines":
            location = np.array([1.0, 0, 0])
            direction = np.array([0.17, 0, 0]) - location
        else:
            raise ValueError(f"No geometry available for {instrument}")
        los_start, los_stop = self.get_los(location, direction)
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
        results["location"] = np.array(location)
        results["direction"] = np.array(direction)

        return results

    def _get_interferometry(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
    ) -> Dict[str, Any]:

        if len(uid) == 0:
            uid = self.UIDS_MDS[instrument]

        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }

        results["revision"] = self._get_revision(uid, instrument, revision)
        revision = results["revision"]

        # TODO: update when new MDS+ structure becomes available
        # position, position_path = self._get_signal(uid, instrument,
        # ".geometry:position", revision)
        # direction, position_path = self._get_signal(uid, instrument,
        # ".geometry:direction", revision)

        if instrument == "smmh1":
            location = np.array([1.0, 0, 0])
            direction = np.array([0.17, 0, 0]) - location
        elif instrument == "nirh1" or instrument == "nirh1_bin":
            location = np.array([-0.07, 0.9, 0])
            direction = np.array([0.37, -0.75, 0]) - location
        else:
            raise ValueError(f"No geometry available for {instrument}")
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

            qval_err, q_path_err = self._get_signal(
                uid, instrument, self.QUANTITIES_MDS[instrument][q] + "_err", revision
            )
            if np.array_equal(qval_err, "FAILED"):
                qval_err = np.zeros_like(qval)
                q_path_err = ""
            results[q + "_error"] = qval_err
            results[q + "_error" + "_records"] = q_path_err

            qval_syserr, q_path_syserr = self._get_signal(
                uid,
                instrument,
                self.QUANTITIES_MDS[instrument][q] + "_syserr",
                revision,
            )
            if not np.array_equal(qval_syserr, "FAILED"):
                results[q + "_error"] = np.sqrt(qval_err**2 + qval_syserr**2)
                results[q + "_error" + "_records"] = [q_path_err, q_path_err]

        results["length"] = 1
        results["location"] = np.array(location)
        results["direction"] = np.array(direction)

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

    def get_revision_name(self, revision):
        """Return string defining RUN## or BEST if revision = 0"""

        if type(revision) is not str:
            if revision < 0:
                rev_str = ""
            elif revision == 0:
                rev_str = ".best"
            elif revision < 10:
                rev_str = f".run0{int(revision)}"
            elif revision > 9:
                rev_str = f".run{int(revision)}"
        else:
            rev_str = f".run{revision}"

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
