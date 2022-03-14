import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, "../../")
from hda.read_st40 import ST40data


# Run
pulseNo = 9780
st40_data = ST40data(pulse=pulseNo, tstart=0.02, tend=0.12)

# XRCS data
#st40_data.get_xrcs()

# PI spectroscopy data
st40_data.get_princeton()
print(st40_data.data["princeton"])


# PI CX results data
#st40_data.get_princeton_cxs()


print(st40_data.data)


