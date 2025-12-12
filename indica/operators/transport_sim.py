import matplotlib.pyplot as plt
import aurora

import numpy as np
from omfit_classes import omfit_eqdsk

from st40_database.formatted_data_writers import geqdsk

from indica.configs.operators.aurora_config import AuroraConfig


def define_profiles():
    T_core = 2000  # eV
    T_edge = 50  # eV
    T_alpha1 = 1.0
    T_alpha2 = 1.0
    n_core = 1e14  # cm^-3
    n_edge = 0.1e14  # cm^-3
    n_alpha1 = 2
    n_alpha2 = 0.5
    rhop = np.linspace(0, 1, 100)
    ne = (n_core - n_edge) * (1 - rhop ** n_alpha1) ** n_alpha2 + n_edge
    Te = (T_core - T_edge) * (1 - rhop ** T_alpha1) ** T_alpha2 + T_edge
    return rhop, Te, ne


aurora_config = AuroraConfig

rhop, Te, ne = define_profiles()
kp = aurora_config["kin_profs"]
kp["Te"]["rhop"] = kp["ne"]["rhop"] = rhop
kp["ne"]["vals"] = ne
kp["Te"]["vals"] = Te



filename, content = geqdsk.write(pulseNo=10041, code_run="EFIT#BEST", time_desired=80e-3)
with open(filename, "w") as handle:
    handle.write(content)
geqdsk_content = omfit_eqdsk.OMFITgeqdsk(filename)
asim = aurora.aurora_sim(namelist=aurora_config, geqdsk=geqdsk_content)


D_z = 2e4 * np.ones(len(asim.rvol_grid))  # cm^2/s
V_z = -2e2 * np.ones(len(asim.rvol_grid))  # cm/s

n_steps = 10
max_sim_time = 1000
nz_norm_steady2 = asim.run_aurora_steady(
    D_z,
    V_z,
    nz_init=None,
    tolerance=0.001,
    max_sim_time=max_sim_time,
    dt=1e-3,
    dt_increase=1.05,
    n_steps=n_steps,
    plot=True,
    plot_radial_coordinate="rho_pol"
)


plt.show(block=True)
