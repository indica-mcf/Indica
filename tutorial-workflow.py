# %% imports

from copy import deepcopy
from socket import getfqdn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from indica.converters import FluxSurfaceCoordinates
from indica.equilibrium import Equilibrium
from indica.operators import BolometryDerivation
from indica.operators import ExtrapolateImpurityDensity
from indica.operators import FractionalAbundance
from indica.operators import ImpurityConcentration
from indica.operators import InvertRadiation
from indica.operators import PowerLoss
from indica.operators import SplineFit
from indica.operators.main_ion_density import MainIonDensity
from indica.operators.mean_charge import MeanCharge
from indica.readers import ADASReader
from indica.readers import PPFReader
from indica.utilities import coord_array

# %% set up

pulse = 96375
trange = (49.0, 50.5)

R = coord_array(np.linspace(1.83, 3.9, 50), "R")
rho = coord_array(np.linspace(0, 1, 25), "rho_poloidal")
z = coord_array(np.linspace(-1.75, 2.0, 50), "z")
t = coord_array(np.linspace(*trange, 5), "t")

main_ion = "d"
high_z = "w"
zeff_el = "be"
impurities = [high_z, zeff_el]
elements = impurities + [main_ion]

server = (
    "https://sal.jetdata.eu" if "jetdata" in getfqdn().lower() else "https://sal.jet.uk"
)

# %% reading the data

reader = PPFReader(pulse=pulse, tstart=trange[0], tend=trange[1], server=server)

diagnostics = {
    "efit": reader.get(uid="jetppf", instrument="eftp", revision=0),
    "hrts": reader.get(uid="jetppf", instrument="hrts", revision=0),
    "sxr": reader.get(uid="jetppf", instrument="sxr", revision=0),
    "zeff": reader.get(uid="jetppf", instrument="ks3", revision=0),
    "bolo": reader.get(uid="jetppf", instrument="bolo", revision=0),
}

efit_equilibrium = Equilibrium(equilibrium_data=diagnostics["efit"])
for key, diag in diagnostics.items():
    for data in diag.values():
        if hasattr(data.attrs["transform"], "equilibrium"):
            del data.attrs["transform"].equilibrium
        if "efit" not in key.lower():
            data.indica.equilibrium = efit_equilibrium

flux_surface = FluxSurfaceCoordinates(kind="poloidal")
flux_surface.set_equilibrium(efit_equilibrium)

# %% fitting profiles

knots_te = [0.0, 0.3, 0.6, 0.85, 0.9, 0.98, 1.0, 1.05]
fitter_te = SplineFit(
    lower_bound=0.0,
    upper_bound=diagnostics["hrts"]["te"].max() * 1.1,
    knots=knots_te,
)
results_te = fitter_te(rho, t, diagnostics["hrts"]["te"])
te = results_te[0]

temp_ne = deepcopy(diagnostics["hrts"]["ne"])
temp_ne.attrs["datatype"] = deepcopy(
    diagnostics["hrts"]["te"].attrs["datatype"]
)  # TEMP for SplineFit checks
knots_ne = [0.0, 0.3, 0.6, 0.85, 0.95, 0.98, 1.0, 1.05]
fitter_ne = SplineFit(lower_bound=0.0, upper_bound=temp_ne.max() * 1.1, knots=knots_ne)
results_ne = fitter_ne(rho, t, temp_ne)
ne = results_ne[0]

# %% fitting soft x-ray

cameras = ["v"]
n_knots = 7
inverter = InvertRadiation(num_cameras=len(cameras), datatype="sxr", n_knots=n_knots)

emissivity, emiss_fit, *camera_results = inverter(
    R,
    z,
    t,
    *[diagnostics["sxr"][key] for key in cameras],
)

# %% read ADAS

adas = ADASReader()

SCD = {
    element: adas.get_adf11("scd", element, year)
    for element, year in zip(impurities, ["89"] * len(impurities))
}
SCD[main_ion] = adas.get_adf11("scd", "h", "89")
ACD = {
    element: adas.get_adf11("acd", element, year)
    for element, year in zip(impurities, ["89"] * len(impurities))
}
ACD[main_ion] = adas.get_adf11("acd", "h", "89")
FA = {
    element: FractionalAbundance(SCD=SCD.get(element), ACD=ACD.get(element))
    for element in elements
}

PLT = {
    element: adas.get_adf11("plt", element, year)
    for element, year in zip(impurities, ["89"] * len(impurities))
}
PLT[main_ion] = adas.get_adf11("plt", "h", "89")
PRB = {
    element: adas.get_adf11("prb", element, year)
    for element, year in zip(impurities, ["89"] * len(impurities))
}
PRB[main_ion] = adas.get_adf11("prb", "h", "89")
PL = {
    element: PowerLoss(PLT=PLT.get(element), PRB=PRB.get(element))
    for element in elements
}

# %% Calculating power loss

fzt = {
    elem: xr.concat(
        [
            FA[elem](
                Ne=ne.interp(t=time),
                Te=te.interp(t=time),
                tau=time,
            ).expand_dims("t", -1)
            for time in t.values
        ],
        dim="t",
    )
    .assign_coords({"t": t.values})
    .assign_attrs(transform=flux_surface)
    for elem in elements
}

power_loss = {
    elem: xr.concat(
        [
            PL[elem](
                Ne=ne.interp(t=time),
                Te=te.interp(t=time),
                F_z_t=fzt[elem].sel(t=time, method="nearest"),
            ).expand_dims("t", -1)
            for time in t.values
        ],
        dim="t",
    )
    .assign_coords({"t": t.values})
    .assign_attrs(transform=flux_surface)
    for elem in elements
}

q = (
    xr.concat(
        [MeanCharge()(FracAbundObj=fzt[elem], element=elem) for elem in elements],
        dim="element",
    )
    .assign_coords({"element": elements})
    .assign_attrs(transform=flux_surface)
)

# %% Initial assumptions

n_zeff_el = xr.zeros_like(emissivity).assign_coords({"element": zeff_el})
n_main_ion = xr.zeros_like(emissivity).assign_coords({"element": main_ion})

M = 1
sxr_rescale_factor = 2.5

# %% High Z impurity density

other_densities = xr.concat(
    [
        n_zeff_el,
        n_main_ion.expand_dims({"element": [main_ion]}, -1),
    ],
    dim="element",
).indica.remap_like(emissivity)

other_power_loss = xr.concat(
    [
        val.indica.remap_like(emissivity).sum("ion_charges")
        for key, val in power_loss.items()
        if key != high_z
    ],
    dim="element",
).assign_coords({"element": [key for key in power_loss.keys() if key != high_z]})

n_high_z = (
    sxr_rescale_factor * M * emissivity
    - ne.indica.remap_like(emissivity)
    * (other_densities * other_power_loss).sum("element")
) / (
    ne.indica.remap_like(emissivity)
    * power_loss[high_z].indica.remap_like(emissivity).sum("ion_charges")
).assign_attrs(
    {"transform": flux_surface}
)

extrapolator = ExtrapolateImpurityDensity()
n_high_z, *high_z_extrapolate_params = extrapolator(
    impurity_density_sxr=n_high_z.where(n_high_z > 0.0, other=1.0).fillna(1.0),
    electron_density=ne,
    electron_temperature=te,
    truncation_threshold=1.5e3,
    flux_surfaces=ne.transform,
)

n_high_z = n_high_z.assign_coords({"element": high_z})

# %% low Z density profile

zeff = diagnostics["zeff"]["zefh"].interp(t=t.values)
conc_zeff_el, _ = ImpurityConcentration()(
    element=zeff_el,
    Zeff_LoS=zeff,
    impurity_densities=xr.concat(
        [n_high_z, n_zeff_el.indica.remap_like(emissivity)],
        dim="element",
    )
    .transpose("element", "R", "z", "t")
    .assign_coords({"element": impurities})
    .fillna(0.0),
    electron_density=ne.where(ne > 0.0, other=1.0),
    mean_charge=q.fillna(0.0),
    flux_surfaces=flux_surface,
)
n_zeff_el = (
    (conc_zeff_el.values * ne)
    .assign_attrs({"transform": flux_surface})
    .assign_coords({"element": zeff_el})
)

# %% bolometry LOS data


def bolo_los(bolo_diag_array):
    return [
        [
            np.array([bolo_diag_array.attrs["transform"].x_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_start.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].x_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].z_end.data[i].tolist()]),
            np.array([bolo_diag_array.attrs["transform"].y_end.data[i].tolist()]),
            "bolo_kb5",
        ]
        for i in bolo_diag_array.bolo_kb5v_coords
    ]


nhz_rho_theta = high_z_extrapolate_params[0].assign_coords({"element": high_z})

bolo_derivation = BolometryDerivation(
    flux_surfs=flux_surface,
    LoS_bolometry_data=bolo_los(diagnostics["bolo"]["kb5v"]),
    t_arr=t,
    impurity_densities=xr.concat([nhz_rho_theta, n_zeff_el], dim="element")
    .assign_coords({"element": [high_z, zeff_el]})
    .transpose("element", "rho_poloidal", "theta", "t"),
    frac_abunds=[fzt.get(high_z), fzt.get(zeff_el)],
    impurity_elements=[high_z, zeff_el],
    electron_density=ne,
    main_ion_power_loss=power_loss.get(main_ion).sum("ion_charges"),  # type: ignore
    impurities_power_loss=xr.concat(
        [
            power_loss.get(element).sum("ion_charges")  # type: ignore
            for element in impurities
        ],
        dim="element",
    ).assign_coords({"element": impurities}),
)
derived_power_los = bolo_derivation(trim=False)

# %% optimise high z density profile

n_high_z = extrapolator.optimize_perturbation(
    extrapolated_smooth_data=nhz_rho_theta,
    orig_bolometry_data=diagnostics["bolo"]["kb5v"],
    bolometry_obj=bolo_derivation,
    impurity_element=high_z,
    asymmetry_modifier=extrapolator.asymmetry_modifier,
)

n_high_z.attrs["transform"] = flux_surface

# %% calculate main ion density

n_main_ion = (
    MainIonDensity()(
        impurity_densities=xr.concat(
            [n_high_z, n_zeff_el], dim="element"
        ).assign_coords({"element": impurities}),
        electron_density=ne,
        mean_charge=q.where(q.element != main_ion, drop=True),
    )
    .assign_attrs({"transform": flux_surface})
    .assign_coords({"element": main_ion})
)

# %% remap densities

electron_density = ne.indica.remap_like(emissivity)
main_ion_density = n_main_ion.indica.remap_like(emissivity)
impurity_density = xr.concat(
    [
        n_high_z.indica.remap_like(emissivity),
        n_zeff_el.indica.remap_like(emissivity),
    ],
    dim="element",
).assign_coords({"element": impurities})

# %% calculate the SXR calibration factor

densities = xr.concat(
    [
        impurity_density,
        main_ion_density.expand_dims({"element": [main_ion]}, -1),
    ],
    dim="element",
    coords="minimal",
    compat="override",
)
M = (
    ne.indica.remap_like(emissivity)
    * (
        densities * xr.concat(power_loss.values(), dim="element").sum("ion_charges")
    ).sum("element")
    / emissivity
).mean()

# %% plot

main_ion_density.isel(t=0).plot(x="R")
plt.show()
impurity_density.sel(element=high_z).isel(t=0).plot(x="R")
plt.show()
impurity_density.sel(element=zeff_el).isel(t=0).plot(x="R")
plt.show()
