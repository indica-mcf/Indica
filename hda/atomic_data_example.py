
from indica.operators.atomic_data import FractionalAbundance
from indica.operators.atomic_data import PowerLoss
from indica.readers import ADASReader

import hda.simple_profiles as profiles

from copy import deepcopy
import xarray as xr
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

INPUT_TE, INPUT_NE, _ = profiles.default_profiles()
INPUT_TE.name = "Electron Temperature (eV)"
INPUT_NE.name = "Electron Density (m$^{-3}$)"

nh0 = 1.e15
nh1 = 1.e11
INPUT_NH = INPUT_TE.rho_poloidal ** 2.5
INPUT_NH *= (nh0 - nh1)
INPUT_NH += nh1
INPUT_NH.name = "Neutral H Density (m$^{-3}$)"

tau0 = 1.e-2
tau1 = 1.e-4
TAU = deepcopy(INPUT_NE)
TAU -= TAU.sel(rho_poloidal=1.0)
TAU /= TAU.sel(rho_poloidal=0.0)
TAU *= (tau0 - tau1)
TAU += tau1
TAU.name = "tau (s)"
TAU_MAX = f"{TAU.max().values*1.e3:.1f}"

def fractional_abundance(element="ar"):
    """Test initialization of FractionalAbundance class."""
    ADAS_file = ADASReader()

    SCD = ADAS_file.get_adf11("scd", element, "89")
    ACD = ADAS_file.get_adf11("acd", element, "89")
    CCD = ADAS_file.get_adf11("ccd", element, "89")

    frac_abundance = FractionalAbundance(
        SCD,
        ACD,
        Ne=INPUT_NE,
        Te=INPUT_TE,
    )
    F_z = frac_abundance.calc_F_z_tinf()
    F_z_tau = frac_abundance(tau=TAU)

    frac_abundance = FractionalAbundance(
        SCD,
        ACD,
        Ne=INPUT_NE,
        Te=INPUT_TE,
        Nh=INPUT_NH,
        CCD=CCD,
    )
    F_z_nh = frac_abundance.calc_F_z_tinf()

    plt.figure()
    INPUT_TE.plot()

    plt.figure()
    INPUT_NE.plot()

    plt.figure()
    INPUT_NH.plot()

    plt.figure()
    TAU.plot()

    plt.figure()
    ne_tau = (TAU * 1.e3 * INPUT_NE)
    ne_tau.name = "Ne * Tau (m$^{-3}$ ms)"
    ne_tau.plot()

    plt.figure()
    cols = cm.rainbow(np.linspace(0, 1, len(F_z.ion_charges)))
    for i, q in enumerate(F_z.ion_charges):
        label, label_tau, label_nh = None, None, None
        if q==F_z.ion_charges.max():
            label, label_tau, label_nh = "tau -> inf", f"tau < {TAU_MAX} ms", "Nh > 0"
        F_z.sel(ion_charges=q).plot(label=label, color=cols[i])
        F_z_tau.sel(ion_charges=q).plot(label=label_tau, linestyle="dashed", color=cols[i])
        F_z_nh.sel(ion_charges=q).plot(label=label_nh, linestyle="dotted", color=cols[i])
    plt.ylabel(f"{element} fractional abundance")
    plt.legend()

    return F_z

def power_loss(element="ar"):
    """Test initialization of PowerLoss class."""
    ADAS_file = ADASReader()

    F_z = test_fractional_abundance(tau, element)

    PLT = ADAS_file.get_adf11("plt", element, "89")
    PRC = ADAS_file.get_adf11("prc", element, "89")
    PRB = ADAS_file.get_adf11("prb", element, "89")

    power_loss = PowerLoss(
        PLT,
        PRB,
        F_z_t=F_z,
        Ne=INPUT_NE,
        Nh=INPUT_NH,
        Te=INPUT_TE,
        PRC=PRC,
    )

    L_z = power_loss()

    return L_z