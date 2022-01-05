import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, "../../")
from hda.read_st40 import ST40data


# Run
pulseNo = 9537
st40_data = ST40data(pulse=pulseNo, tstart=0.02, tend=0.12)
st40_data.get_all()


st40_data.get_xrcs()

print(st40_data.data)


