from ..numpy_typing import LabeledArray
import numpy as np
from scipy.optimize import root
from xarray import DataArray


class NeutralBeam:

    def __init__(self, name: str):
        self.name = name

    def test_flow(self):
        print('Test script, to do!')
    #
#

