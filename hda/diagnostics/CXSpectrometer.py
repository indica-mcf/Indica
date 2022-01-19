from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
import xarray as xr
import pickle
from xarray import DataArray
from scipy import constants
from scipy.interpolate import interp1d

from indica.readers import ADASReader
from indica.operators.atomic_data import FractionalAbundance
from indica.converters import LinesOfSightTransform

from hda.profiles import Profiles

from indica.numpy_typing import ArrayLike

class CXSpectrometer:
    """
    """

    def __init__(
        self,
        name="",
        adf11: dict = None,
        adf12: dict = None,
        adf15: dict = None,
    ):
        """
        Read all atomic data and initialise objects

        Parameters
        ----------
        name
            Identifier for the spectrometer
        adf11
            Dictionary with details of ionisation balance data (see ADF11 class var)
        adf15
            Dictionary with details of photon emission coefficient data (see ADF15 class var)

        Returns
        -------

        """

        self.adasreader = ADASReader()
        self.name = name
        # self.set_ion_data(adf11=adf11)
        # self.set_pec_data(adf15=adf15, marchuk=marchuk)

    def test_flow(self):
        """
        Test module with standard inputs
        """

        print('Hello, I am here!!')
