"""Provides implementation of :py:class:`readers.DataReader` for
reading PPF data produced by JET.

"""

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

import numpy as np
from sal.client import SALClient
from sal.core.exception import AuthenticationFailed
from sal.core.exception import NodeNotFound
from sal.dataclass import Signal
import scipy.constants as sc

import indica.readers.surf_los as surf_los
from indica.utilities import CACHE_DIR
from .abstractreader import DataReader
from ..numpy_typing import ArrayLike
from ..numpy_typing import RevisionLike
from ..utilities import to_filename

SURF_PATH = Path(surf_los.__file__).parent.parent / "data/surf_los.dat"


class PPFError(Exception):
    """An exception which occurs when trying to read PPF data which would
    not be caught by the lower-level SAL library. An example would be
    failing to find any valid channels for an instrument when each channel
    is a separate DTYPE.

    """


class PPFWarning(UserWarning):
    """A warning that occurs while trying to read PPF data. Typically
    related to caching in some way.

    """


class PPFReader(DataReader):
    """Class to read JET PPF data using SAL.

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

    INSTRUMENT_METHODS = {
        "hrts": "get_thomson_scattering",
        "lidr": "get_thomson_scattering",
        "efit": "get_equilibrium",
        "eftp": "get_equilibrium",
        "kk3": "get_cyclotron_emissions",
        "ks3h": "get_bremsstrahlung_spectroscopy",
        "ks3v": "get_bremsstrahlung_spectroscopy",
        "sxrh": "get_radiation",
        "sxrv": "get_radiation",
        "sxrt": "get_radiation",
        "kb5h": "get_radiation",
        "kb5v": "get_radiation",
        "kg10": "get_thomson_scattering",
        **{
            "cx{}m".format(val): "get_charge_exchange"
            for val in ("s", "d", "f", "g", "h")
        },
        **{"cx{}6".format(val): "get_charge_exchange" for val in ("s", "d", "f", "g")},
        **{"cx{}4".format(val): "get_charge_exchange" for val in ("s", "d", "f", "h")},
    }
    _IMPLEMENTATION_QUANTITIES = {
        "hrts": {
            "ne": ("number_density", "electrons"),
            "te": ("temperature", "electrons"),
        },
        "lidr": {
            "ne": ("number_density", "electrons"),
            "te": ("temperature", "electrons"),
        },
        "efit": {
            "f": ("f_value", "plasma"),
            "faxs": ("magnetic_flux_axis", "poloidal"),
            "fbnd": ("magnetic_flux_separatrix", "poloidal"),
            "ftor": ("magnetic_flux", "toroidal"),
            "rmji": ("major_rad", "hfs"),
            "rmjo": ("major_rad", "lfs"),
            "psi": ("magnetic_flux", "poloidal"),
            "vjac": ("volume_jacobian", "plasma"),
            "ajac": ("area_jacobian", "plasma"),
            "rmag": ("major_rad", "mag_axis"),
            "rgeo": ("major_rad", "geometric"),
            "rbnd": ("major_rad", "separatrix"),
            "zmag": ("z", "mag_axis"),
            "zbnd": ("z", "separatrix"),
            "wp": ("energy", "plasma"),
            "psin": ("poloidal_flux", "normalised"),
        },
        "kg10": {"ne": ("number_density", "electron")},
        "sxrh": {"h": ("luminous_flux", "sxr")},
        "sxrv": {"v": ("luminous_flux", "sxr")},
        "sxrt": {"t": ("luminous_flux", "sxr")},
        "kb5h": {"kb5h": ("luminous_flux", "bolometric")},
        "kb5v": {"kb5v": ("luminous_flux", "bolometric")},
        "ks3h": {"zefh": ("effective_charge", "plasma")},
        "ks3v": {"zefv": ("effective_charge", "plasma")},
        **{
            "cx{}m".format(val): {
                "angf": ("angular_freq", "ion"),
                "ti": ("temperature", "ion"),
                "conc": ("concentration", "ion"),
            }
            for val in ("s", "d", "f", "g", "h")
        },
        **{
            "cx{}6".format(val): {
                "angf": ("angular_freq", "ions"),
                "ti": ("temperature", "ions"),
                "conc": ("concentration", "ions"),
            }
            for val in ("s", "d", "f", "g")
        },
        **{
            "cx{}4".format(val): {
                "angf": ("angular_freq", "ions"),
                "ti": ("temperature", "ions"),
                "conc": ("concentration", "ions"),
            }
            for val in ("s", "d", "f", "h")
        },
    }
    _BREMSSTRAHLUNG_LOS = {
        "ks3": "edg7",
    }

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
    ):
        self._reader_cache_id = f"ppf:{server.replace('-', '_')}:{pulse}"
        self.NAMESPACE: Tuple[str, str] = ("jet", server)
        super().__init__(
            tstart,
            tend,
            pulse=pulse,
            server=server,
            default_error=default_error,
        )
        self.pulse = pulse
        self._client = SALClient(server)
        self._client.prompt_for_password = False
        self._default_error = default_error

    def get_sal_path(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> str:
        """Return the path in the PPF database to for the given INSTRUMENT
        (DDA in JET)."""
        return (
            f"/pulse/{self.pulse:d}/ppf/signal/{uid}/{instrument}/"
            f"{quantity}:{revision:d}"
        )

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[Signal, str]:
        """Gets the signal for the given INSTRUMENT (DDA in JET), at the
        given revision."""
        path = self.get_sal_path(uid, instrument, quantity, revision)
        info = self._client.list(path)
        path = self.get_sal_path(
            uid,
            instrument,
            quantity,
            info.revision_current,
        )
        cache_path = self._sal_path_to_file(path)
        data = self._read_cached_ppf(cache_path)
        if data is None:
            data = self._client.get(path)
            self._write_cached_ppf(cache_path, data)
        return data, path

    def _get_revision(
        self, uid: str, instrument: str, revision: RevisionLike
    ) -> RevisionLike:
        """
        Get actual revision that's being read from database, converts relative revision
        (e.g. 0, latest) to absolute
        """
        info = self._client.list(
            f"/pulse/{self.pulse:d}/ppf/signal/{uid}/{instrument}:{revision:d}"
        )
        return info.revision_current

    def _read_cached_ppf(self, path: Path) -> Optional[Signal]:
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

    def _write_cached_ppf(self, path: Path, data: Signal):
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

    def _get_charge_exchange(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
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
        atomic_num, atomic_num_path = self._get_signal(
            uid, instrument, "zqnn", revision
        )

        # TODO: is there no string information on the element in the database?
        # mass_int = round(mass.data[0])
        # atomic_num_int = round(atomic_num.data[0])
        # results["element"] = [
        #     value[2]
        #     for value in ELEMENTS.values()
        #     if (value[0] == atomic_num_int and value[1] == mass_int)
        # ][0]

        # We approximate that the positions do not change much in time
        results["R"] = R.data[0, :]
        results["x"] = np.zeros_like(results["R"])
        results["y"] = np.zeros_like(results["R"])
        results["z"] = z.data[0, :]
        results["length"] = R.data.shape[1]
        results["texp"] = texp.data
        results["time"] = None
        paths = [R_path, z_path, m_path, t_path]
        if "angf" in quantities:
            angf, a_path = self._get_signal(uid, instrument, "angf", revision)
            afhi, e_path = self._get_signal(uid, instrument, "afhi", revision)
            if results["time"] is None:
                results["time"] = angf.dimensions[0].data
            results["angf"] = angf.data
            results["angf_error"] = afhi.data - angf.data
            results["angf_records"] = paths + [a_path, e_path]
        if "conc" in quantities:
            try:
                conc, c_path = self._get_signal(uid, instrument, "conc", revision)
                cohi, e_path = self._get_signal(uid, instrument, "cohi", revision)
                if results["times"] is None:
                    results["times"] = conc.dimensions[0].data
                results["conc"] = conc.data
                results["conc_error"] = cohi.data - conc.data
                results["conc_records"] = paths + [c_path, e_path]
            except NodeNotFound:
                # CONC not always produced for JET CXRS
                results["conc"] = None
                results["conc_error"] = None
                results["conc_records"] = paths
        if "ti" in quantities:
            ti, t_path = self._get_signal(uid, instrument, "ti", revision)
            tihi, e_path = self._get_signal(uid, instrument, "tihi", revision)
            if results["time"] is None:
                results["time"] = ti.dimensions[0].data
            results["ti"] = ti.data
            results["ti_error"] = tihi.data - ti.data
            results["ti_records"] = paths + [t_path, e_path]

        results["location"] = None
        results["direction"] = None

        # Currently arbitrary, TODO be more specific/justified. Based approximately on
        # start of JET C38 campaign
        c38_start = 92671
        if self.pulse > c38_start:
            spec = {
                "cxs": "ks5a",
                "cxd": "ks5b",
                "cxf": "ks5c",
                "cxg": "ks5d",
                "cxh": "ks5e",
            }.get(instrument.lower()[:3], None)
            if spec is None:
                raise PPFError(
                    f"Failed to match spectrometer name to instrument {instrument}"
                )
            trck, _ = self._get_signal(uid, instrument, "trck", revision)
            location, direction = _get_cxrs_los_geometry(
                sav_file=_get_cxrs_los_savfile(pulse=self.pulse, spec=spec),
                tracks=_get_cxrs_active_tracks(
                    pulse=self.pulse, spec=spec, trck=trck.data[: len(results["R"]) + 1]
                ),
            )
            results["location"] = location
            results["direction"] = direction
        else:
            warnings.warn(
                f"CXRS LOS geometry not supported for JPN {self.pulse}, "
                f"currently only supporting after {c38_start}"
            )

        results["machine_dims"] = self.MACHINE_DIMS
        results["revision"] = self._get_revision(uid, instrument, revision)
        return results

    def _get_thomson_scattering(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, Any]:
        """Fetch raw data for electron temperature or number density
        calculated from Thomson scattering.

        """
        results = {}
        z, z_path = self._get_signal(uid, instrument, "z", revision)
        results["z"] = z.data
        results["R"] = z.dimensions[0].data
        results["x"] = z.dimensions[0].data
        results["y"] = z.dimensions[0].data
        results["length"] = len(z.data)
        if "te" in quantities:
            te, t_path = self._get_signal(uid, instrument, "te", revision)
            results["te"] = te.data
            results["times"] = te.dimensions[0].data
            if instrument == "lidr":
                tehi, e_path = self._get_signal(uid, instrument, "teu", revision)
                results["te_error"] = tehi.data - results["te"]
            else:
                dte, e_path = self._get_signal(uid, instrument, "dte", revision)
                results["te_error"] = dte.data
            results["te_records"] = [z_path, t_path, e_path]
        if "ne" in quantities:
            ne, d_path = self._get_signal(uid, instrument, "ne", revision)
            results["ne"] = ne.data
            results["times"] = ne.dimensions[0].data
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

        results["revision"] = self._get_revision(uid, instrument, revision)
        results["machine_dims"] = self.MACHINE_DIMS
        return results

    def _get_equilibrium(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, Any]:
        """Fetch raw data for plasma equilibrium."""
        results: Dict[str, Any] = {}
        for q in quantities:
            if q == "psin":
                continue
            qval, q_path = self._get_signal(uid, instrument, q, revision)
            if qval.dimensions[0].temporal:
                results["times"] = qval.dimensions[0].data
            if (
                len(qval.dimensions) > 1
                and q not in {"psi", "rbnd", "zbnd"}
                and "psin" not in results
            ):
                results["psin"] = qval.dimensions[1].data
                results["psin_records"] = [q_path]
            if q == "psi":
                r, r_path = self._get_signal(uid, instrument, "psir", revision)
                z, z_path = self._get_signal(uid, instrument, "psiz", revision)
                results["psi_r"] = r.data
                results["psi_z"] = z.data
                results["psi"] = qval.data.reshape(
                    (len(results["time"]), len(z.data), len(r.data))
                )
                results["psi_records"] = [q_path, r_path, z_path]
            else:
                results[q] = qval.data
                results[q + "_records"] = [q_path]

        results["revision"] = self._get_revision(uid, instrument, revision)
        return results

    def _get_cyclotron_emissions(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, Any]:
        """Fetch raw data for electron cyclotron emissin diagnostics."""
        _, _, zstart, zend, _, _ = surf_los.read_surf_los(
            SURF_PATH, self.pulse, instrument.lower()
        )
        assert zstart[0] == zend[0]

        # gen contains accquisition parameters
        # e.g. which channels are valid (gen[0,:] > 0)
        gen, gen_path = self._get_signal(uid, instrument, "gen", revision)
        channels = np.argwhere(gen.data[0, :] > 0)[:, 0]
        freq = gen.data[15, channels] * 1e9
        nharm = gen.data[11, channels]
        results: Dict[str, Any] = {
            "machine_dims": self.MACHINE_DIMS,
            "z": zstart[0],
            "length": len(channels),
            "Btot": 2 * np.pi * freq * sc.m_e / (sc.e * nharm),
        }
        bad_channels = np.argwhere(
            np.logical_or(gen.data[18, channels] == 0.0, gen.data[19, channels] == 0.0)
        )
        results["bad_channels"] = results["Btot"][bad_channels]
        for q in quantities:
            records = [SURF_PATH.name, gen_path]
            data = []
            for i in channels:
                qval, q_path = self._get_signal(
                    uid, instrument, f"{q}{i + 1:02d}", revision
                )
                records.append(q_path)
                data.append(qval.data)
                if "time" not in results:
                    results["time"] = qval.dimensions[0].data
            results[q] = np.array(data).T
            results[q + "_error"] = self._default_error * results[q]
            results[q + "_records"] = records

        results["revision"] = self._get_revision(uid, instrument, revision)
        return results

    def _get_radiation(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, Any]:
        """Fetch raw data for radiation quantities such as SXR and bolometric
        fluxes..

        """
        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        # HACK: Need a better solution eventually
        if "sxr" in instrument:
            instrument = "sxr"
        elif "kb5" in instrument:
            instrument = "bolo"
        for q in quantities:
            records = [SURF_PATH.name]
            if instrument == "bolo":
                qval, qpath = self._get_signal(uid, instrument, q, revision)
                records.append(qpath)
                results["length"] = qval.dimensions[1].length
                results["times"] = qval.dimensions[0].data
                results[q] = qval.data
                channels: Union[List[int], slice] = slice(None, None)
            else:
                luminosities = []
                channels = []
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
                    if "times" not in results:
                        results["times"] = qval.dimensions[0].data
                if len(channels) == 0:
                    # TODO: Try getting information on the INSTRUMENT (DDA in JET),
                    #  to determine if the failure is actually due to requesting
                    #  an invalid INSTRUMENT (DDA in JET) or revision
                    self._client.list(
                        f"/pulse/{self.pulse:d}/ppf/signal/{uid}/{instrument}:"
                        f"{revision:d}"
                    )
                    raise PPFError(f"No channels available for {instrument}/{q}.")
                results["length"] = len(channels)
                results[q] = np.array(luminosities).T
            results[q + "_error"] = self._default_error * results[q]
            results[q + "_records"] = records
            xstart, xend, zstart, zend, ystart, yend = surf_los.read_surf_los(
                SURF_PATH, self.pulse, instrument.lower() + "/" + q.lower()
            )
            location = np.asarray([xstart, ystart, zstart])
            direction = np.asarray([xend, yend, zend]) - location

        results["location"] = location.transpose()
        results["direction"] = direction.transpose()
        results["revision"] = self._get_revision(uid, instrument, revision)
        return results

    def _get_bremsstrahlung_spectroscopy(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantities: Set[str],
        dl: float = 0.005,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "length": {},
            "machine_dims": self.MACHINE_DIMS,
        }
        if "ks3" in instrument:
            instrument = "ks3"
        los_instrument = self._BREMSSTRAHLUNG_LOS[instrument]
        for q in quantities:
            qval, q_path = self._get_signal(uid, instrument, q, revision)
            los, l_path = self._get_signal(uid, los_instrument, "los" + q[-1], revision)
            if "times" not in results:
                results["times"] = qval.dimensions[0].data
            results["length"] = 1
            results[q] = qval.data
            results[q + "_error"] = 0.0 * results[q]
            location = np.asarray([[(los.data[1] / 1000)], [0], [(los.data[2] / 1000)]])
            direction = (
                np.asarray([[(los.data[4] / 1000)], [0], [(los.data[5] / 1000)]])
                - location
            )
            results[q + "_records"] = [q_path, l_path]

        results["location"] = location.transpose()
        results["direction"] = direction.transpose()
        results["revision"] = self._get_revision(uid, instrument, revision)
        return results

    def _get_bad_channels(
        self, uid: str, instrument: str, quantity: str
    ) -> List[Number]:
        """Returns a list of channels which are known to be bad for all pulses
        on this instrument. Typically this would be for reasons of
        geometriy (e.g., lines of sight facing the diverter). This
        should be overridden with machine-specific information.

        Parameters
        ----------
        uid
            User ID (i.e., which user created this data).
        instrument
            Name of the instrument which measured this data.
        quantities
            Which physical quantity this data represents.

        Returns
        -------
        :
            A list of channels known to be problematic. These will be ignored
            by default.

        """
        if instrument == "bolo":
            if quantity == "kb5h":
                return cast(List[Number], [*range(0, 8), *range(19, 24)])
            elif quantity == "kb5v":
                return cast(List[Number], [*range(0, 1), *range(5, 16), *range(22, 32)])
            else:
                return []
        else:
            return []

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
            True if authentication is needed, otherwise false.
        """
        return self._client.auth_required

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


def _get_cxrs_los_geometry(sav_file: Path, tracks: ArrayLike) -> Any:
    """Read IDL save file to get position and direction for KS5 tracks

    Parameters
    ----------

    sav_file:
        Path to IDL save file to load
    tracks:
        ArrayLike of tracks to select from save file, from JET dtype `TRCK`

    Returns
    -------

    :
        Tuple of position and direction arrays
    """
    import scipy.io as io

    data = io.readsav(sav_file)
    fibres = data.ptrfib.fibres[0]
    numfibview = fibres.numfibview[0].astype(str)
    origin, direction = [], []
    for name, pos, vec in zip(
        numfibview,
        fibres.losdef[0].virtualposition_roomtemp_rot[0].cartesian_ref[0],
        fibres.losdef[0].virtualdirection_rot[0].cartesian_ref[0],
    ):
        if name in tracks:
            origin.append(pos)
            direction.append(vec)
    return (np.asarray(origin), np.asarray(direction))


def _setup_idl(pulse: int) -> Any:
    try:
        import idlbridge as idlb
    except ImportError:
        raise PPFError("Could not find IDLBridge module, required for CXRS geometry")

    idlb.execute(".reset")
    idlb.execute(
        "!PATH=!PATH + ':' + "
        "expand_path( '+~cxs/idl_spectro/' ) + ':' + "
        "expand_path( '+~cxs/idl_spectro/show' ) + ':' + "
        "expand_path( '+~cxs/ks6read/' ) + ':' + "
        "expand_path( '+~cxs/ktread/' ) + ':' + "
        "expand_path( '+~cxs/kx1read/' ) + ':' + "
        "expand_path( '+~cxs/idl_spectro/kt3d' ) + ':' + "
        "expand_path( '+~cxs/utc' ) + ':' + "
        "expand_path( '+~cxs/instrument_data' ) + ':' + "
        "expand_path( '+~cxs/calibration' ) + ':' + "
        "expand_path( '+~cxs/alignment' ) + ':' + "
        "expand_path( '+/usr/local/idl' ) + ':' + "
        "expand_path( '+/home/CXSE/cxsfit/idl/' ) + ':' + "
        "expand_path( '+~/jet/share/lib' ) + ':' + "
        "expand_path( '+~/jet/share/root/lib' ) + ':' + "
        "expand_path( '+~/jet/share/idl' )"
    )

    idlb.execute(
        "!PATH = !PATH + ':' + expand_path('+/u/cxs/utilities',/all_dirs)+ ':'"
    )

    idlb.execute(
        "!PATH = !PATH + ':' + expand_path('+/u/cxs/instrument_data/namelists')+ ':' + "
        "expand_path('+/u/cxs/instrument_data/jpfnodes')"
    )

    idlb.execute("!PATH = !PATH + ':' + expand_path('+/usr/local/idl')")
    idlb.execute("!PATH = !PATH + ':' + expand_path('+/home/CXSE/cxsfit/idl/')")
    idlb.execute("!PATH = !PATH +':/home/CXSE/cxsfit/idl:'")
    idlb.execute(".compile plot")
    idlb.execute(".compile ppfread")
    idlb.execute(".compile cxf_number_to_text")
    idlb.execute(".compile cxf_read_switches")
    idlb.execute(".compile cxf_decompress_history")

    idlb.put("shot", pulse)
    idlb.execute("julian_date = agm_pulse_to_julian(shot, 'DG')")

    return idlb


def _get_cxrs_los_savfile(pulse: int, spec: str) -> Path:
    """Determine correct sav file for given pulse and spectrometer

    Parameters
    ----------
    pulse:
        Pulse to search for
    spec:
        Spectrometer to search for

    Returns
    -------
    :
        Path to save file
    """
    octant = {"ks5a": 7, "ks5b": 1, "ks5c": 7, "ks5d": 7, "ks5e": 1}.get(spec.lower())
    idlb = _setup_idl(pulse=pulse)
    idlb.execute(
        "str1 = periscope_oct{}_hist_align(julian_date=julian_date)".format(str(octant))
    )
    idlb.execute("file_align = str1.file_align")
    savfile = Path(str(idlb.get("file_align")))
    try:
        assert savfile.exists() and savfile.is_file()
    except AssertionError:
        raise PPFError(f"Cannot find savfile {savfile}")
    return savfile


def _get_cxrs_active_tracks(pulse: int, spec: str, trck: ArrayLike) -> ArrayLike:
    """Translate tracks used to track names as in geometry save file"""
    idlb = _setup_idl(pulse=pulse)
    idlb.execute(
        "str1={}_hist_fibresetup(pulse=pulse,julian_date=julian_date)".format(str(spec))
    )
    idlb.execute("viewing_position=str1.pulse_setup.viewing_position")
    idlb.execute("track_reshuffle=str1.pulse_setup.ks4fit_track_reshuffle")
    viewing_position = idlb.get("viewing_position")
    track_reshuffle = idlb.get("track_reshuffle")
    assert len(viewing_position) == len(track_reshuffle) == len(trck)
    return [
        viewing_position[i - 1]
        for i in track_reshuffle
        if int(trck[len(trck) - i]) == 1
    ][::-1]
