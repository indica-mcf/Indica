#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:51:41 2021

@author: sundaresan.sridhar
"""

#TEST BED

from MDSplus import *
import numpy as np
from typing import Set
import pickle
from importlib import reload
from xarray import DataArray


whichTree = 'ASTRA'

whichRun = 'RUN101'

import os
os.chdir('../')

import indica
reload(indica)

from indica.readers import DataReader
from indica.readers import ST40Reader


#EFIT EQUILIBRIUM DATA
pulseNo = 8548
reader = ST40Reader(pulseNo,0,1)
aa_radiation = reader._get_radiation("sxr","diode_arrays",1,{"filter_1"})
bb_radiation = reader.get_radiation("sxr","diode_arrays",1,{"filter_1"})












                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      