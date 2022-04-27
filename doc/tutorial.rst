Tutorial
========

Set Up
------

We start by defining the pulse to analyse, coordinate arrays on which we will
carry out our analysis and the impurities present:

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
initialise the equilibrium and coordinate systems:

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

Use the soft x-ray camera diagnostic data to estimate the emissivity profile:

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
