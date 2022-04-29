Tutorial
========

This tutorial aims to introduce the core operators used in analysis, in a
format that can easily be copied, pasted and ran (we recommend Jupyter
notebooks). It is specific to JET in the interest of producing something that
runs and is testable, however we have done our best to make it clear and easy
to adapt for other machines.

Set Up
------

We start by defining the pulse to analyse, coordinate arrays on which we will
carry out our analysis, the impurities present and the server to request data
from:

.. code-block:: python

   from socket import getfqdn
   import numpy as np
   from indica.utilities import coord_array

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

   server = "https://sal.jetdata.eu" if "jetdata" in getfqdn().lower() else "https://sal.jet.uk"

Reading in the data
-------------------

Next we read in the diagnostic data listed :ref:`Scope, Inputs and Outputs`,
initialise the equilibrium (from EFIT data) and coordinate systems:

.. code-block:: python

   from indica.readers import PPFReader
   from indica.equilibrium import Equilibrium
   from indica.converters import FluxSurfaceCoordinates

   reader = PPFReader(
      pulse=pulse,
      tstart=trange[0],
      tend=trange[1],
      server=server
   )

   diagnostics = {
       "efit": reader.get(uid="jetppf", instrument="eftp", revision=0),
       "hrts": reader.get(uid="jetppf", instrument="hrts", revision=0),
       "sxr": reader.get(uid="jetppf", instrument="sxr", revision=0),
       "ks3": reader.get(uid="jetppf", instrument="ks3", revision=0),
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

Fitting profiles
----------------

Next we fit splines to the electron temperature and density profiles:

.. code-block:: python

   from copy import deepcopy
   from indica.operators import SplineFit

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
   fitter_ne = SplineFit(
      lower_bound=0.0, upper_bound=temp_ne.max() * 1.1, knots=knots_ne
   )
   results_ne = fitter_ne(rho, t, temp_ne)
   ne = results_ne[0]

Fitting soft x-ray profile
--------------------------

Use the soft x-ray camera diagnostic data to estimate the shape of the
emissivity profile:

.. code-block:: python

   from indica.operators import InvertRadiation

   cameras = ["v"]
   n_knots = 7
   inverter = InvertRadiation(
      num_cameras=len(cameras), datatype="sxr", n_knots=n_knots
   )

   emissivity, emiss_fit, *camera_results = inverter(
      R,
      z,
      t,
      *[diagnostics["sxr"][key] for key in cameras],
   )

Read ADAS data
--------------

Read in atomic data from ADAS files and calculate fractional abundance of
ionisation states for elements at different electron temperatures and
densities:

.. code-block:: python

   from indica.readers import ADASReader
   from indica.operators import FractionalAbundance
   from indica.operators import PowerLoss

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
      element: FractionalAbundance(
         SCD=SCD.get(element), ACD=ACD.get(element)
      )
      for element in elements
   }

   #TODO: work out how to distribute these files
   adas = ADASReader("/home/elitherl/Analysis/SXR/indica/sxr_filtered_adf11/")
   PLT = {
      element: adas.get_adf11("plsx", element, year)
      for element, year in zip(impurities, ["5"] * len(impurities))
   }
   PLT[main_ion] = adas.get_adf11("plsx", "h", "5")
   PRB = {
      element: adas.get_adf11("prsx", element, year)
      for element, year in zip(impurities, ["5"] * len(impurities))
   }
   PRB[main_ion] = adas.get_adf11("prsx", "h", "5")
   PL = {
      element: PowerLoss(PLT=PLT.get(element), PRB=PRB.get(element))
      for element in elements
   }

Calculating power loss
----------------------

Interpolate the fractional abundance profile for our elements given their
temperature and density profiles, calculate the power loss for our elements and
the mean charge of each element given the fractional abundancies.

.. code-block:: python

   from xarray import concat
   from indica.operators.mean_charge import MeanCharge

   fzt = {
      elem: concat(
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
      elem: concat(
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
      concat(
          [
              MeanCharge()(FracAbundObj=fzt[elem], element=elem)
              for elem in elements
          ],
          dim="element",
      )
      .assign_coords({"element": elements})
      .assign_attrs(transform=flux_surface)
   )

Calculate the high Z impurity density profile
---------------------------------------------

Next we use the emissivity data calculated from the soft x-ray cameras to
estimate the density profile of the high Z element. We also use the electron
density profile to extrapolate the shape of the high Z density profile outside
of the range of applicability of the SXR diagnostic.

.. code-block:: python

   from indica.operators import ExtrapolateImpurityDensity

   n_high_z_simple = (
      2.5
      * emissivity
      / (
          ne.indica.remap_like(emissivity)
          * power_loss[high_z]
          .indica.remap_like(emissivity)
          .sum("ion_charges")
      )
   )
   extrapolator = ExtrapolateImpurityDensity()
   n_high_z_extrapolated, *high_z_extrapolate_params = extrapolator(
      impurity_density_sxr=n_high_z_simple.where(
          n_high_z_simple > 0.0, other=1.0
      ).fillna(1.0),
      electron_density=ne,
      electron_temperature=te,
      truncation_threshold=1.5e3,
      flux_surfaces=ne.transform,
   )

Estimate low Z density profile
------------------------------

Now we use the effective Z measurement to estimate the low Z element's density
profile by subtracting the profile of the high Z element:

.. code-block:: python

   import xarray as xr
   from indica.operators import ImpurityConcentration

   zeff = diagnostics["ks3"]["zefh"].interp(t=t.values)
   conc_zeff_el, _ = ImpurityConcentration()(
      element=zeff_el,
      Zeff_LoS=zeff,
      impurity_densities=concat(
          [
              n_high_z_extrapolated.fillna(0.0),
              xr.zeros_like(n_high_z_extrapolated),
          ],
          dim="element",
      ).assign_coords({"element": impurities}),
      electron_density=ne.where(ne > 0.0, other=1.0),
      mean_charge=q.fillna(0.0),
      flux_surfaces=flux_surface,
   )
   n_zeff_el = (conc_zeff_el.values * ne).assign_attrs(
      {"transform": flux_surface}
   )

Derive bolometry LOS data
-------------------------

Next we use the data calculated before in order to create an estimator of the
values that the bolometry cameras would read, given our current model:

.. code-block:: python

   from indica.operators import BolometryDerivation

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

   nhz_rho_theta = high_z_extrapolate_params[2]

   bolo_derivation = BolometryDerivation(
      flux_surfs=flux_surface,
      LoS_bolometry_data=bolo_los(diagnostics["bolo"]["kb5v"]),
      t_arr=t,
      impurity_densities=concat([nhz_rho_theta, n_zeff_el], dim="element")
      .assign_coords({"element": [high_z, zeff_el]})
      .transpose("element", "rho_poloidal", "theta", "t"),
      frac_abunds=[fzt.get(high_z), fzt.get(zeff_el)],
      impurity_elements=[high_z, zeff_el],
      electron_density=ne,
      main_ion_power_loss=power_loss.get(main_ion).sum("ion_charges"),
      impurities_power_loss=concat(
          [
              power_loss.get(element).sum("ion_charges")
              for element in impurities
          ],
          dim="element",
      ).assign_coords({"element": impurities}),
   )
   derived_power_los = bolo_derivation(trim=False)

Optimise high Z density profile
-------------------------------

Now we fit a gaussian over-density on the low field side of the plasma using
the actual bolometry measurements and our bolometry predictor:

.. code-block:: python

   nhz_rho_theta = high_z_extrapolate_params[2]
   asymmetry_modifier = high_z_extrapolate_params[3]
   n_high_z = extrapolator.optimize_perturbation(
      extrapolated_smooth_data=nhz_rho_theta,
      orig_bolometry_data=diagnostics["bolo"]["kb5v"],
      bolometry_obj=bolo_derivation,
      impurity_element=high_z,
      asymmetry_modifier=asymmetry_modifier,
   )

   n_high_z.attrs["transform"] = flux_surface

Calculate main ion density
--------------------------

Compute the main ion density given our calculated impurity densities, our
calculated mean charge and the electron density.

.. code-block:: python

   from indica.operators.main_ion_density import MainIonDensity

   n_main_ion = MainIonDensity()(
      impurity_densities=concat(
          [n_high_z, n_zeff_el], dim="element"
      ).assign_coords({"element": impurities}),
      electron_density=ne,
      mean_charge=q.where(q.element != main_ion, drop=True),
   ).assign_attrs({"transform": flux_surface})

Remap densities
---------------

Now we remap the densities ready for plotting:

.. code-block:: python

   electron_density = ne.indica.remap_like(emissivity)
   main_ion_density = n_main_ion.indica.remap_like(emissivity)
   impurity_density = concat(
      [
          n_high_z.indica.remap_like(emissivity),
          n_zeff_el.indica.remap_like(emissivity),
      ],
      dim="element",
   ).assign_coords({"element": impurities})

Plotting
--------

Finally we plot our density profiles:

.. code-block:: python

   import matplotlib.pyplot as plt

   main_ion_density.isel(t=0).plot(x="R")
   plt.show()
   impurity_density.sel(element=high_z).isel(t=0).plot(x="R")
   plt.show()
   impurity_density.sel(element=zeff_el).isel(t=0).plot(x="R")
   plt.show()
