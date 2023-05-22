import matplotlib.pylab as plt
import numpy as np
import xarray as xr
from xarray import DataArray

# from indica.converters.line_of_sight import LineOfSightTransform
# from indica.converters.transect import TransectCoordinates
# from indica.equilibrium import Equilibrium
# from indica.models.charge_exchange import ChargeExchange
# from indica.models.diode_filters import BremsstrahlungDiode
# from indica.models.helike_spectroscopy import Helike_spectroscopy
# from indica.models.interferometry import Interferometry
from indica.models.plasma import Plasma
# from indica.models.sxr_camera import SXRcamera
# from indica.models.thomson_scattering import ThomsonScattering
# from indica.readers.read_gacode import get_gacode_data
from indica.readers.read_st40 import ReadST40
from indica.workflows.load_modelling_plasma import plasma_code
def plasma_code(
    pulse: int,
    tstart: float,
    tend: float,
    dt: float,
    data: dict,
    verbose: bool = False,
):
    """
    Assign code data to new Plasma class

    Parameters
    ----------
    pulse
        MDS+ pulse number to read from
    revision
        Tree revision/run number
    tstart, tend, dt
        Times axis of the new plasma object

    Returns
    -------
        Plasma class with code data and the data

    """
    if verbose:
        print("Assigning code data to Plasma class")

    n_rad = len(data["ne"].rho_poloidal)
    main_ion = "h"
    impurities = ("ar", "c", "he")
    impurity_concentration = (0.001, 0.03, 0.01)

    plasma = Plasma(
        tstart=tstart,
        tend=tend,
        dt=dt,
        machine_dimensions=((0.15, 0.95), (-0.7, 0.7)),
        impurities=impurities,
        main_ion=main_ion,
        impurity_concentration=impurity_concentration,
        pulse=pulse,
        full_run=False,
        n_rad=n_rad,
    )

    Te = data["te"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    plasma.electron_temperature.values = Te.values

    Ne = data["ne"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.electron_density.values = Ne.values

    Ti = data["ti"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e3
    for element in plasma.elements:
        plasma.ion_temperature.loc[dict(element=element)] = Ti.values
    for i, impurity in enumerate(plasma.impurities):
        #todo fix hack to add mutiple impurities
        Nimp = data[f"niz{1}"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19 #i+
        plasma.impurity_density.loc[dict(element=impurity)] = Nimp.values

    Nf = data["nf"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.fast_density.values = Nf.values

    Nn = data["nn"].interp(rho_poloidal=plasma.rho, t=plasma.t) * 1.0e19
    plasma.neutral_density.values = Nn.values

    Pblon = data["pblon"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_parallel.values = Pblon.values

    Pbper = data["pbper"].interp(rho_poloidal=plasma.rho, t=plasma.t)
    plasma.pressure_fast_perpendicular.values = Pbper.values

    plasma.build_atomic_data(default=True)

    return plasma
# GaCODE + ASTRA interpretative using HDA/EFIT:
pulse = 9850
pulse_code = 34010009#13109850
run_code = 'V50'#61
code = "transp_test"#"astra"
equil = "efit"
add_gacode = True
tstart=0.02
tend=0.06
dt=0.01
verbose=True
st40 = ReadST40(pulse_code, tstart, tend, dt=dt, tree=code)
# st40 = ReadST40(pulse, tstart, tend, dt=dt, tree=code)
# st40 = ReadST40(pulse, tstart, tend, dt=dt, tree="st40")
# st40(instruments=["smmh1"], map_diagnostics=False)
st40.get_raw_data("", code, run_code)
st40.bin_data_in_time([code], tstart, tend, dt)
data_code = st40.binned_data[code]
plasma = plasma_code(pulse_code, tstart, tend, dt, data_code, verbose=verbose)
help(st40)