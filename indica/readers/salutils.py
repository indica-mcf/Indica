from pathlib import Path
import pickle
import stat
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import numpy as np
from sal.client import SALClient
from sal.dataclass import Signal

from indica.abstractio import BaseIO
from indica.numpy_typing import RevisionLike
from indica.utilities import CACHE_DIR
from indica.utilities import to_filename


class SALError(Exception):
    """An exception which occurs when trying to read PPF data which would
    not be caught by the lower-level SAL library. An example would be
    failing to find any valid channels for an instrument when each channel
    is a separate DTYPE.

    """


class SALWarning(UserWarning):
    """A warning that occurs while trying to read PPF data. Typically
    related to caching in some way.

    """


class SALUtils(BaseIO):
    def __init__(self, pulse: int, server: str = "https://sal.jetdata.eu"):
        self.pulse = pulse
        self._reader_cache_id = f"ppf:{server.replace('-', '_')}:{pulse}"
        self._client = SALClient(server)
        self._client.prompt_for_password = False

    def _get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[Signal, str]:
        """Gets the signal for the given INSTRUMENT (DDA in JET), at the
        given revision."""
        path = self.get_sal_path(
            uid,
            instrument,
            quantity,
            self.get_revision(uid, instrument, quantity, revision),
        )
        cache_path = self._sal_path_to_file(path)
        data = self._read_cached_ppf(cache_path)
        if data is None:
            data = self._client.get(path)
            self._write_cached_ppf(cache_path, data)
        return data, path

    def get_signal(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, str]:
        signal, path = self._get_signal(
            uid=uid,
            instrument=instrument,
            quantity=quantity,
            revision=revision,
        )
        return signal.data, path

    def _get_signal_dims(signal: Signal) -> List[np.array]:
        dims = [dim.data for dim in signal.dimensions]
        return dims

    def get_signal_dims(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[List[np.array], List[str]]:
        signal, path = self._get_signal(
            uid=uid,
            instrument=instrument,
            quantity=quantity,
            revision=revision,
        )
        dims = self._get_signal_dims(signal=signal)
        paths = [path] * len(dims)
        return dims, paths

    def _get_signal_units(signal: Signal) -> str:
        return signal.units

    def get_signal_units(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> str:
        signal, path = self._get_signal(
            uid=uid,
            instrument=instrument,
            quantity=quantity,
            revision=revision,
        )
        return self._get_signal_units(signal=signal)

    def get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, List[np.array], str, str]:
        signal, path = self._get_signal(
            uid=uid,
            instrument=instrument,
            quantity=quantity,
            revision=revision,
        )
        dims = self._get_signal_dims(signal=signal)
        units = self._get_signal_units(signal=signal)
        return signal.data, dims, units, path

    def get_revision(
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

    def get_sal_path(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> str:
        """Return the path in the PPF database to for the given INSTRUMENT
        (DDA in JET)."""
        return (
            f"/pulse/{self.pulse:d}/ppf/signal/{uid}/{instrument}/"
            f"{quantity}:{revision:d}"
        )

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
                SALWarning,
            )
            return None
        with path.open("rb") as f:
            try:
                return pickle.load(f)
            except pickle.UnpicklingError:
                warnings.warn(
                    f"Error unpickling cache file {path}. (Possible data corruption.)",
                    SALWarning,
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
