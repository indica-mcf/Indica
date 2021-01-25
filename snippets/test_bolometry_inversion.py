import getpass
import itertools

import matplotlib.pyplot as plt
import numpy as np

from indica.equilibrium import Equilibrium
from indica.operators import InvertRadiation
from indica.readers import PPFReader
from indica.utilities import coord_array


cameras = ["kb5v", "kb5h"]
R = coord_array(np.linspace(1.83, 3.9, 50), "R")
z = coord_array(np.linspace(-1.75, 2.0, 50), "z")
t = coord_array(np.linspace(45, 50, 5), "t")

reader = PPFReader(90279, 45.0, 50.0)
if reader.requires_authentication:
    user = input("JET username: ")
    password = getpass.getpass("JET password: ")
    assert reader.authenticate(user, password)

equilib_dat = reader.get("jetppf", "efit", 0)
bolo = reader.get("jetppf", "bolo", 0)

equilibrium = Equilibrium(equilib_dat)

# *********************************************************************

for data in itertools.chain(equilib_dat.values(), bolo.values()):
    if hasattr(data.attrs["transform"], "equilibrium"):
        del data.attrs["transform"].equilibrium

for data in itertools.chain(bolo.values()):
    data.indica.equilibrium = equilibrium

inverter = InvertRadiation(len(cameras), "bolometric", 6)

results = inverter(R, z, t, *(bolo[c] for c in cameras))
emissivity = results["emissivity"]

for time in t:
    emissivity.sel(t=time).plot(x="R", y="z", cmap="plasma")
    plt.show()

    for cam, col in zip(cameras, ["r", "g", "b"]):
        print(f"Plotting bolometry camera {cam}")
        print("=======================")
        data = results["bolo_" + cam + "_binned"].sel(t=time)

        x1 = results.coords[data.attrs["transform"].x1_name]
        x2 = results.coords[data.attrs["transform"].x2_name]
        emissivity_vals = results.attrs["emissivity_model"](
            data.attrs["transform"], x1, x2, time
        )
        emissivity_vals.plot(x="R", y="z", cmap="plasma", vmin=0.0)
        plt.show()

        data.plot.line("o", label="From camera")
        results["bolo_" + cam + "_back_integral"].sel(t=time).plot(label="From model")
        plt.legend()
        plt.show()
