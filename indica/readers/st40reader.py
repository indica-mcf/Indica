"""Provides implementation of :py:class:`readers.DataReader` for
reading MDS+ data produced by ST40.

"""


from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple

from MDSplus.mdsExceptions import TreeNNF
from MDSplus.mdsExceptions import TreeNODATA
import numpy as np

from .abstractreader import DataReader
from .mdsutils import MDSUtils
from .st40conf import ST40Conf
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
        conf=ST40Conf,
        utils=MDSUtils,
        server: str = "smaug",
        tree: str = "ST40",
        verbose:bool = False,
        default_error: float = 0.05,
    ):
        super().__init__(
            pulse, 
            tstart,
            tend,
            conf=conf, 
            utils=utils,
            server=server,
            verbose=verbose,
            default_error=default_error,
            tree=tree,
        )
        mds_utils = self.utils(pulse, server, tree)
        self._get_signal = mds_utils.get_signal
        self._get_signal_dims = mds_utils.get_signal_dims
        self._get_data = mds_utils.get_data
        self._get_revision = mds_utils.get_revision

    def _get_thomson_scattering(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        R = database_results["R"]
        database_results["channel"] = np.arange(len(R))
        database_results["z"] = R * 0.0
        database_results["x"] = R
        database_results["y"] = R * 0.0
        return database_results

    def _get_profile_fits(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        return database_results

    def _get_charge_exchange(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["channel"] = np.arange(len(database_results["x"]))
        database_results["element"] = ""
        if "wavelength" in database_results.keys():
            if len(np.size(database_results["wavelength"])) > 1:
                database_results["wavelength"] = database_results["wavelength"][0, :]
            database_results["pixel"] = np.arange(len(database_results["wavelength"]))
        return database_results

    def _get_spectrometer(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        # Sort channel indexing either hardcore or selecting channels with finite data only
        spectra = database_results["spectra"]
        if database_results["instrument"] == "pi":
            has_data = np.arange(21, 28)
        else:
            has_data = np.where(
                np.isfinite(spectra[0, :, 0]) * (spectra[0, :, 0] > 0)
            )[0]
        database_results["spectra"] = database_results["spectra"][:, has_data, :]
        database_results["spectra_error"] = database_results["spectra_error"][:, has_data, :]
        database_results["location"] = database_results["location"][has_data, :]
        database_results["direction"] = database_results["direction"][has_data, :]
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        if len(np.size(database_results["wavelength"])) >1:
            database_results["wavelength"] = database_results["wavelength"][0, :]

        rearrange_geometry(database_results["location"], database_results["direction"])
        return database_results    

    def _get_equilibrium(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["psi"] = database_results.reshape(
            (
                len(database_results["t"]),
                len(database_results["psi_z"]),
                len(database_results["psi_r"]),
            )
        )
        return database_results

    def _get_radiation(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        return database_results

    def _get_helike_spectroscopy(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        return database_results

    def _get_diode_filters(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        _labels = database_results["label"]
        if type(_labels[0]) == np.bytes_:
            database_results["label"] = np.array([label.decode("UTF-8") for label in _labels])
        else:
            database_results["label"] = _labels
        rearrange_geometry(database_results["location"], database_results["direction"])
        return database_results


    def _get_interferometry(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        # if database_results["instrument"] == "smmh":
        #     location = (location + location_r) / 2.0
        #     direction = (direction + direction_r) / 2.0
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        rearrange_geometry(database_results["location"], database_results["direction"])
        return database_results

    def _get_zeff(
        self,
        database_results:dict,
    ) -> Dict[str, Any]:
        database_results["channel"] = np.arange(database_results["location"][:, 0].size)
        rearrange_geometry(database_results["location"], database_results["direction"])
        return database_results

    def close(self):
        """Ends connection to the SAL server from which PPF data is being
        read."""
        raise NotImplementedError
    
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

def rearrange_geometry(location, direction):
    if len(np.shape(location)) == 1:
        location = np.array([location])
        direction = np.array([direction])