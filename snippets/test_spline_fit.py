import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np

from indica.equilibrium import Equilibrium
from indica.operators import SplineFit
from indica.readers import PPFReader
from indica.utilities import coord_array


rho = coord_array(np.linspace(0, 1.04, 100), "rho_poloidal")
t = coord_array(np.linspace(45, 50, 11), "t")

reader = PPFReader(90279, 45.0, 50.0)
if reader.requires_authentication:
    user = input("JET username: ")
    password = getpass.getpass("JET password: ")
    assert reader.authenticate(user, password)

equilib_dat = reader.get("jetppf", "efit", 0)
hrts = reader.get("jetppf", "hrts", 0)
lidr = reader.get("jetppf", "lidr", 0)

equilibrium = Equilibrium(equilib_dat)

# *********************************************************************

for data in itertools.chain(hrts.values(), lidr.values()):
    data.indica.equilibrium = equilibrium

fitter = SplineFit(lower_bound=0.0)

Te = [lidr["te"], hrts["te"]]
results = fitter(rho, t, *Te)
Te_smoothed = results[0]
Te_spline = Te_smoothed.attrs["splines"]

for time in t:
    print(f"Time = {float(time)}")
    Te_smoothed.sel(t=time).plot(label="Spline fit")
    for d in results[1:]:
        d.sel(t=time).plot.line("o", x="rho_poloidal", label=d.name + " data")
    plt.legend()
    plt.show()
