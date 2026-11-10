# [[file:jet_comparison.org::*Imports][Imports:1]]
import numpy as np
from indica import Equilibrium, Plasma
from indica.converters import LineOfSightTransform
from indica.operators import FractionalAbundanceAdas, NbiADAS
from indica.readers import JETReader, SALUtils
from indica.utilities import build_dataarrays

# Imports:1 ends here

# [[file:jet_comparison.org::*Read plasma data][Read plasma data:1]]
reader = JETReader(97490, 50, 60)
assert isinstance(reader.reader_utils, SALUtils)
# Equilibrium for remapping R -> rhop
efit = Equilibrium(reader.get("jetppf", "efit"))
# n_e and t_e from thomson scattering
hrts = reader.get("jetppf", "hrts")
hrts["ne"].transform.set_equilibrium(efit)
hrts["te"].transform.set_equilibrium(efit)
t = 56.5
rhop = np.linspace(0.0, 1.0, 200)
# This is just converting channel to rhop, filling missing values to full 0->1 range
ne = (
    hrts["ne"]
    .sel(t=t, method="nearest")
    .assign_coords(rhop=hrts["ne"].transform.convert_to_rho_theta(t=t)[0])
    .swap_dims({"channel": "rhop"})
    .drop_vars("channel")
    .interp(rhop=rhop)
    .ffill("rhop")
    .bfill("rhop")
)
te = (
    hrts["te"]
    .sel(t=t, method="nearest")
    .assign_coords(
        rhop=hrts["te"].transform.convert_to_rho_theta(t=t)[0].drop_vars("t")
    )
    .swap_dims({"channel": "rhop"})
    .drop_vars("channel")
    .interp(rhop=rhop)
    .ffill("rhop")
    .bfill("rhop")
)

# t_i and impurity density from CXRS
cx = reader.get("jetppf", "cxd6")
cx["ti"].transform.set_equilibrium(efit)
cx["dens"].transform.set_equilibrium(efit)
ti = (
    cx["ti"]
    .sel(t=t, method="nearest")
    .assign_coords(rhop=cx["ti"].transform.convert_to_rho_theta(t=t)[0].drop_vars("t"))
    .swap_dims({"channel": "rhop"})
    .drop_vars("channel")
    .interp(rhop=rhop)
    .ffill("rhop")
    .bfill("rhop")
)
ni = (
    cx["dens"]
    .sel(t=t, method="nearest")
    .assign_coords(
        rhop=cx["dens"].transform.convert_to_rho_theta(t=t)[0].drop_vars("t"),
        element="ne",
    )
    .swap_dims({"channel": "rhop"})
    .drop_vars("channel")
    .interp(rhop=rhop)
    .bfill("rhop")
    .fillna(0.0)
    .expand_dims("element")
)
# Read plasma data:1 ends here

# [[file:jet_comparison.org::*Read plasma data][Read plasma data:2]]
# Don't stricly need a `Plasma` object but it will automatically calculate main ion
# density for us, so handy to have
plasma = Plasma(t, t + 1, 1, machine="jet", impurities=("ne",))
plasma.electron_density.loc[:] = ne.interp(rhop=plasma.rhop)
plasma.electron_temperature.loc[:] = te.interp(rhop=plasma.rhop)
plasma.ion_temperature.loc[:] = ti.interp(rhop=plasma.rhop)
plasma.impurity_density.loc[:] = ni.interp(rhop=plasma.rhop)
plasma.fract_abu = {
    elem: FractionalAbundanceAdas(elem, full_run=True) for elem in plasma.elements
}
# Read plasma data:2 ends here

# [[file:jet_comparison.org::*Make NBI model][Make NBI model:1]]
dist_centre = 4.723
ang_beamline = -19.63
orig6 = np.asarray([dist_centre, 0.0, 0.0])
dir6 = np.asarray(
    [-np.cos(np.deg2rad(ang_beamline)), np.sin(np.deg2rad(ang_beamline)), 0.0]
)
dir6 /= np.linalg.norm(dir6)
tr6 = LineOfSightTransform(
    np.asarray([orig6[0]]),
    np.asarray([orig6[1]]),
    np.asarray([orig6[2]]),
    np.asarray([dir6[0]]),
    np.asarray([dir6[1]]),
    np.asarray([dir6[2]]),
    name="PINI8.6",
)
tr6.set_equilibrium(efit)
eng6 = float(reader.reader_utils.get_signal("jetppf", "nbi8", "eng6", 0)[0][0])
data6 = build_dataarrays(
    {
        "eng": reader.reader_utils.get_signal("jetppf", "nbi8", "eng6", 0)[0],
        "pow": reader.reader_utils.get_signal("jetppf", "nbi8", "pow6", 0)[0],
        "pfr": reader.reader_utils.get_signal("jetppf", "nbi8", "pfr6", 0)[0],
        "t": reader.reader_utils.get_signal_dims("jetppf", "nbi8", "pow6", 0)[0][0],
    },
    {
        "t": ("t", ["t"]),
        "eng": ("beam_energy", []),
        "pow": ("beam_power", ["t"]),
        "pfr": ("beam_power_fraction", []),
    },
    reader.tstart,
    reader.tend,
    tr6,
)
pini6 = NbiADAS(
    name=tr6.name,
    energy=float(data6["eng"][0].values),
    power=float(data6["pow"].sel(t=t, method="nearest").values),
    nbi_element="d",
    power_fractions=(0.5, 0.3, 0.2),
    power_fractions_identifiers=(1, 1 / 2, 1 / 3),
)
pini6.set_transform(tr6)
# Make NBI model:1 ends here

# [[file:jet_comparison.org::*Make NBI model][Make NBI model:2]]
results = pini6(
    Ti=plasma.ion_temperature,
    Te=plasma.electron_temperature,
    Ne=plasma.electron_density,
    Ni=plasma.ion_density,
    Vtor=plasma.toroidal_rotation,
    t=t,
)
zeta = results["zeta"]  # Currently only result is ZETA factor
# Make NBI model:2 ends here
