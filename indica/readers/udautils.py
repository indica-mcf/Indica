"""
To use on Freia, run:

module unload ifort/12.0
module load gcc/11.2.0
module switch uda/2.9.1

"""

from pathlib import Path
import pickle
import stat
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import numpy as np
from pyuda import Client
from pyuda import Signal

from indica.abstractio import BaseIO
from indica.numpy_typing import RevisionLike
from indica.utilities import CACHE_DIR
from indica.utilities import to_filename


class UDAError(Exception):
    """An exception which occurs when trying to read UDA data which would
    not be caught by pyuda - e.g. using an invalid UID.

    """


class UDAWarning(UserWarning):
    """A warning that occurs while trying to read UDA data."""


class UDAUtils(BaseIO):
    def __init__(self, pulse, server: str = ""):
        self._pulse = pulse
        self._reader_cache_id = "uda"
        self._client = Client()

    def requires_authentication(self) -> None:
        return False  # Not supporting external UDA access yet

    def close(self) -> None:
        """Ends connection to the UDA client"""
        self._client.close_connection()
        del self._client

    def _mastu_names(self, instrument: str) -> str:
        """Look up three-letter MAST-U name from instrument name"""
        instrument_mapping = {
            "celeste-3": "act",
            "magnetics_efit": "epm",
            "midplane_thomson": "ayc",
        }
        if instrument not in instrument_mapping.keys():
            raise UDAError(f"Instrument not recognised: {instrument}")
        return instrument_mapping[instrument]

    def get_data(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, np.array, str, str]:
        """This is a function which is exposed by datareader

        Parameters
        ----------

        uid
            The only supported uid is "UDADefault", which refers to the main UDA
            database. Functionality to enable local files to be read from
            uda-scratch will come later.
        instrument
            A name for the diagnostic providing the data
        quantity
            The UDA signal path following the three letter alias, e.g. "t_e_core"
            coming after "ayc"
        revision
            "LATEST" gives most recent version of the data
        """

        # Retrieve the data
        signal, source_id = self._get_signal(
            uid=uid,
            instrument=instrument,
            quantity=quantity,
            revision=revision,
        )
        dims = self._get_signal_dims(signal=signal)
        units = self._get_signal_units(signal=signal)
        return signal.data, dims, units, source_id

    def get_error(
        self, uid: str, instrument: str, quantity: str, revision: RevisionLike
    ) -> Tuple[np.array, np.array, str, str]:
        """This is a function which is exposed by datareader

        Parameters
        ----------

        uid
            The only supported uid is "UDADefault", which refers to the main UDA
            database. Functionality to enable local files to be read from
            uda-scratch will come later.
        instrument
            A name for the diagnostic providing the data
        quantity
            The UDA signal path following the three letter alias, e.g. "t_e_core"
            coming after "ayc"
        revision
            "LATEST" gives most recent version of the data
        """

        # Get the error path
        error_quantity = self._get_error_path(uid, instrument, quantity, revision)

        # Get "blank" entry if error doesn't exist
        if error_quantity is None:
            signal, _ = self._get_signal(
                uid=uid,
                instrument=instrument,
                quantity=quantity,
                revision=revision,
            )
            dims = self._get_signal_dims(signal=signal)
            units = self._get_signal_units(signal=signal)
            return signal.errors, dims, units, None
            # Signal.errors returns zeros by convention when no signal is linked

        return self.get_data(uid, instrument, error_quantity, revision)

    def _get_error_path(
        self,
        uid: str,
        instrument: str,
        quantity: str,
        revision: RevisionLike,
    ) -> Optional[str]:
        """Deduce the signal name of the linked errors"""

        # Check the signal itself exists
        _ = self._get_signal(uid, instrument, quantity, revision)

        # Obtain linked signal name
        try:
            error_signal_link, _ = self._get_signal(
                uid,
                instrument,
                quantity + ".errors",
                revision,
            )
        except UDAError:
            return

        # Construct full error signal
        path_stem = "/".join(quantity.split("/")[:-1])
        if path_stem != "":
            error_path = "/".join([path_stem, error_signal_link.str])
        else:
            error_path = error_signal_link.str

        return error_path

    def _get_signal(
        self,
        uid: str,
        instrument: str,
        quantity: str,
        revision: RevisionLike,
    ) -> Tuple[Signal, str]:
        """Retrieves signal from UDA for shotnum according to attribute "_pulse" """

        if uid != "UDADefault":
            raise UDAError(f'The only supported UID so far is "UDADefault", got {uid}')
        signal_name, source, source_id = self.get_uda_client_args(
            uid=uid,
            instrument=instrument,
            revision=self.get_revision(uid, instrument, revision)[0],
            quantity=quantity,
        )

        # Read from cache if possible / desired
        cache_path = self._uda_args_to_file(signal_name, source)
        signal = self._read_cached_uda_file(cache_path)

        # Otherwise read using pyuda
        if signal is None:
            try:
                signal = self._client.get(signal_name, source)
            except Exception as e:
                raise UDAError(
                    f"UDA client could not read {signal_name} from {source}"
                ) from e
            self._write_cached_uda_file(cache_path, signal)

        return signal, source_id

    def get_uda_client_args(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
        quantity: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Construct the signal name and source for pyuda client

        TODO: Add functionality for reading from uda-scratch
        """

        revision = self.get_revision(uid, instrument, revision)[0]

        signal_name = self._mastu_names(instrument=instrument)
        if quantity is not None:
            if quantity != "":
                signal_name += f"/{quantity}"
        if uid == "UDADefault":
            source = f"{self._pulse:d}/{revision:d}"
        else:
            raise UDAError(f'The only supported UID so far is "UDADefault", got {uid}')
        source_id = signal_name + "::" + source
        return signal_name, source, source_id

    def get_revision(
        self,
        uid: str,
        instrument: str,
        revision: RevisionLike,
    ) -> Tuple[int, bool]:
        """If revision is "LATEST" replace with equivalent pass number"""
        latest_pass = self._client.latest_source_pass(
            self._mastu_names(instrument=instrument), self._pulse
        )
        if revision == "LATEST":
            revision = latest_pass
        is_best = revision == latest_pass
        return revision, is_best

    def _get_signal_dims(self, signal: Signal) -> List[np.array]:
        if hasattr(signal, "dims"):
            dims = [dim.data for dim in signal.dims]
            return dims

    def _get_signal_units(self, signal: Signal) -> str:
        if hasattr(signal, "units"):
            return signal.units

    def _write_cached_uda_file(self, path: Path, data: Signal):
        """Write the given signal, fetched from uda, to the disk for
        later reuse.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("wb") as f:
                pickle.dump(data, f)
            path.chmod(0o644)
        except TypeError:  # Note - signal groups can't be pickled
            path.unlink()
            pass

    def _uda_args_to_file(self, signal_name: str, source: str) -> Path:
        """Get the file path which would be used to cache data from the given
        `sal_path`.

        """
        id_list = [self._reader_cache_id, signal_name, source]
        return (
            Path.home()
            / CACHE_DIR
            / self.__class__.__name__
            / to_filename("_".join(id_list) + ".pkl")
        )

    def _read_cached_uda_file(self, path: Path) -> Optional[Signal]:
        """Check if the UDA data specified by `path` has been cached and,
        if so, load it.

        """
        if not path.exists():
            return None
        permissions = stat.filemode(path.stat().st_mode)
        if permissions[5] == "w" or permissions[8] == "w":
            warnings.warn(
                "Can not open cache file which is writeable by anyone other than "
                "the user. (Security risk.)",
                UDAWarning,
            )
            return None
        with path.open("rb") as f:
            try:
                signal = pickle.load(f)
                return signal
            except pickle.UnpicklingError:
                warnings.warn(
                    f"Error unpickling cache file {path}. (Possible data corruption.)",
                    UDAWarning,
                )
                return None
