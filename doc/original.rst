Original IDL Code
=================

.. warning::

	This section of the documentation describes the original WSX code (up
	to March 2020) written in IDL by `Marco Sertoli
	<marco.sertoli@ukaea.uk>`_ and NOT the current python version.


Workflow
----------------
In this section, the steps of code execution are outlined in detail. The names used here are JET-specific, but the concept will be exactly the same on other fusion devices, whether they are Tokamaks or Stellaroators.

1. **JPN, tr, dt** : choose pulse # (JPN) to analyse, time-range of interest (tr) and time resolution (dt = typically 10-50 ms) for analysis.

2. Create reference time axis **t_unfold** with desired resolution dt in range tr.

3. **Equilibrium UID and INSTRUMENT** : define equilibrium diagnostic and read diagnostic-independent geometric quantities often used throughout the code

	* **rho(nx)**: define reference time-independent normalized radial coordinate array (currently rho-poloidal, radial resolution nx=101 points)
	* **R_2D(nx_Rz), z_2D(nx_Rz)**: define time-independent R and z arrays for mapping quantities in 2D on the poloidal plane (currently spatial resolution nRz = 100)
	* **t_eq(nt_eq)**: read time axis of equilibrium PPF, restrict to chosen time-range tr.
	* **Br(R, z_ax, t_eq), Bz(R, z_ax, t_eq), Bt(R, z_ax, t_eq)**: read magnetic field components on the midplane (from the HFS to the LFS), remap on reference rho
	* **B_tot**: calculate total magnetic field from its components (above)
	* **R_ax(t_eq), z_ax(t_eq)** read magnetic axis position
	* **R_sep(nx_sep, t_eq), z_sep(nx_sep, nt_eq)**: read separatrix position(current spatial resolution nx_sep = 150 points)
	* **R_LFS(rho, ϑ=0, t_eq), R_HFS(rho, ϑ=π, t_eq)**: calculate major radii on LFS & HFS midplane (at z=z_ax) on the reference rho
	* **min_rad_LFS(rho, ϑ=0, t_eq), min_rad_HFS(rho, ϑ=π, t_eq)**: calculate minor radii on the LFS & HFS midplane (z=z_ax) on the reference rho
	* **Vol(rho, t_eq)**: read the plasma volume on the reference rho

4. **Check equilibrium** : for H-mode discharges check that Te at separatrix ~ 100 eV

	* **Read raw HRTS Te(R_hrts, t_hrts)**
	* **Map all measurement positions to reference rho for a set of different R-shifts** for all time-points (currently R_shift = [0, 4.0] cm, dR=0.5 cm, time-independent).
	* **Spline-fit Te** for each R_shift, for each time-point
	* **Find best R-shift** to get Te ~ 100 eV at separatrix for each time-point
	* **Save best R_shift** for remapping of **all diagnostics** used in the analysis

5. **Read SXR data**

	* **Choose which cameras/diagnostics to read**
	* **Read raw data** (all channels)
	* **Downsample** to reference time axis t_unfold
	* **Choose which channels to keep** (some may be faulty and should be discarded from the start)

	*All other diagnostics are currently read later on, before commencing the calculation of the impurity density (step 7.). They could be read at this stage, possibly using a common diagnostic-reading-GUI, callable anytime throughout the code if the user wants to change the data used for the computation. Below is a list of all other physical quantities and respective diagnostics currently included in the code:*

	* **Total radiation**: KB5
	* **Radiation tomographic reconstructions**: BOLT, B5NN, B5ML, B5MF, ...
	* **Electron density**: HRTS, LIDR, KG10, ...
	* **Electron temperature**: HRTS, LIDR, KK3, ...
	* **Ion temperature**: CXRS, ...
	* **Toroidal rotation**: CXRS, ...
	* **Impurity concentration and effective charge (passive spectroscopy)**: KT7/3, KX1, KS6, CXRS, KS3, ...

	*For all diagnostics, the reading steps are similar, with slight differences if measurements are LOS-integrals or local, if the time resolution is higher or lower than the desired one (step 1.):*

	* LOS-integrals (SXR, Bolometry, passive spectroscopy, ...):

		* Read data
		* Downsample / interpolate to t_unfold if diagnostic has higher / lower time-resolution
		* Map LOS coordinate arrays (R,z) to normalized radial coordinate type chosen for reference rho
		* Choose which channels to keep (discard channels that are faulty or viewing the divertor for Bolometry)

	* Local measurements (everything else, ...):

		* Read data
		* Interpolate on t_unfold if diagnostic has higher time-resolution
		* Map measurement positions (R,z) to normalized radial coordinate type chosen for reference rho
		* Choose which channels to keep
		* Spline-fit on reference rho
		* Interpolate on t_unfold if diagnostic has lower time-resolution

	*The raw data of each diagnostic should be stored to variable for future use/reference. Some data with high time resolution (e.g. SXR data) can also be used for studying MHD activity by estimating the oscillation amplitude and phase at the mode frequency. This requires data in the 10-100 kHz range (typical mode frequencies of interest are in the range 1-50 kHz).*

6. **Unfold SXR lines-of-sight** to generate a 2D poloidal map of local emissivities. Depending on the symmetry/asymmetry of the radiation patterns, two different routes are used (*The same methodologies can be applied to the Bolometry diagnostic avoiding LOS viewing the divertor. The functionality of using Bolometry instead of SXR as main driving diagnostic should be added to the new version of the code*).
	* **Symmety**: perform a simple **Abel inversion** of all available LOS
	* **Asymmetry**:
		a) **Choose number of spline knots** for the fit of emissivity and asymmetry parameters (see point c) below). This is calculated as a multiplication factor to the diagnostic's average spatial resolution dρ, defined as the average difference in impact parameters of the neighbouring LOS. Typical values range from **x2** for extremely shaped profiles, to **x6** for cases with only slight asymmetry and/or peaking. *In new version, irregular knot spacing should be tried, with closer-spaced knots in the centre and sparcer in the outer half. Knot spacing must anyway never be higher than the diagnostic's spatial resolution!*
		b) **Perform Abel inversion** assuming poloidal symmetry using a set of lines of sight from one camera only (usually the HFS viewing LOS of camera V) to provide a starting assumption for the emissivity profile shape.
		c) **Fit all LOS using equation 1** of | `M. Sertoli et al. Review of Scientific Instruments 89, 113501 (2018) <https://doi.org/10.1063/1.5046562>`_ searching for the best profiles of **ϵ_SXR(ρ,R_0;t)** and **λ_SXR(ρ;t)**. The local emissivity calculated in b) is used as starting point for ϵ_SXR, while λ_SXR is set to zero across the full radius. More details of the current fitting method can be found in :ref:`computation`

7. **Define parameters to calculate the plasma composition**
	* **Choose electron density and temperature diagnostics** with independent input of UID and INSTRUMENT names for Ne and Te
	* **Force non-hollow Ne or Te profiles** (bool, default = False) to avoid hollow spline fits of Ne and Te data that could arise simply from sparse central data.
	* **SXR detection limit** (float, default = 1500): defined as a minimum Te (eV) roughtly coincident with the photon energy of the filter function edge. This limit depends on the thickness of the Be-filter and on the quality of atomic data, so is machine dependent. (*A default is provided and usually works fine, but the user must have the possibility to choose a different radius or temperature limit*)
	* **Account for Zeff** (bool, default = True): calculate a low-Z impurity density to account for missing contributions to the Zeff measurement (*possible only if a Zeff measurement is available*)
	* **Cross-calibrate to VUV** (default = True): use independent passive-spectroscopy impurity concentration measurement of Z0 to cross-calibrate the impurity density calculated using SXR. *For W this is currently implemented using KT7/3 quasi-continuum or spectral lines measurements*.
	* **Choose impurity elements**:
		* Z0: main radiator (default = W)
		* Z1: time-evolving low-Z (default = Be)
		* Z2: second low-Z element with constant background concentration
		* Z3: second mid-/high-Z element (default = Ni)
	* **Choose extrapolation methods** of impurity density Z0 beyond the SXR detection limit. All extrapolation methods (*choice of user*) proceed separately on the LFS- and HFS-midplane to preserve the measured asymmetry. The asymmetry factor λ_Z0 is re-calculated on the extrapolated profiles and used to estimate the 2D impurity density maps and all quantities that depend on them (e.g. total radiated power, Zeff LOS-integral, etc.).
		* **Constant concentration**: follow shape of electron density profile
		* **Extrapolate derivative**: use derivative at SXR detection limit to extrapolate LFS impurity density until a **rho_max** (user defined) where derivative -> 0; beyond rho_max use electron density shape to extrapolate up to the separatrix. The HFS impurity density is extrapolated using shape of electron density only
		* **Fit to KB5**: extrapolate Z0 impurity density using gaussian shape to fit experimental KB5 LOS-integrals. The fit parameteres  are the gaussian peak, height and width. Beyond the peak, the electron density shape is used up to the separatrix. (*The fit is a delicate point and requires more details...*)

	**The code is often used to test consistency of single diagnostic measurements.** Similarly to the shifts to the equilibrium reconstruction outlined in point 4., this requires the possibility to apply scaling factors to each measurement including:
		* Total magnetic field
		* Zeff
		* Impurity concentrations estimated by passive spectrocopy (independent scaling factors for each measurement e.g. from VUV, X-ray spectrometers, CXRS, etc.)

	*This should be available to the user in the GUI when performing the calculation of the impurity densities.*

8. **Read atomic data**
	* **Read ADAS and/or user-specified files** to build ionization balance and cooling factors for all elements (main ion + Z0-Z3 + minority in new version). 	*The program should automatically set default filenames if data is available for that element, otherwise return an error message. The user should also have the possibility to choose alternative files of the same format.*

		* SCD: ionization rate coefficients
		* ACD: recombination rate coefficients
		* PLT: total radiation loss parameter (spectral lines)
		* PRB: total radiation loss parameter (recombination and bremsstrahlung)
		* PLSX: SXR-filtered radiation loss parameter (spectral lines)
		* PRSX: SXR_filtered radiation loss parameter (recombination and bremsstrahlung)
	* **Interpolate the data on the electron temperature profiles** that will be used for the computation
	* **Build fractional abundance, mean-charge, charge^2** (for Zeff calculation) variables from the ionization and recombination rates assuming local-ionization-equilibrium
	* **Estimate uncertainty of the radiation loss parameters** by using upper and lower bounds of electron temperature data as limits.

	*The SXR files are machine-dependent because they change for varying Be-filters. All other fines MUST be the same for all experiments. It might be worthwhile to install the ADAS files with the program in order not to rely on locally available files and to ensure the data-sets used on different machines are identical. User choice should still be possible if new data-sets were to become available, but information in this regard will anyway be stored in the provenance.*

	*In the new version of the code, there should be the option of evaluating the fractional abundance accounting for transport, by coupling with fast impurity transport codes* (e.g. SANCO, `STRAHL  <https://pure.mpg.de/rest/items/item_2143869/component/file_2143868/content>`_, etc.). *A theory driven estimation of the impurity transport coefficients could also be estimated using neoclassical and turbulence codes (NEO? GKW?) which would also improve the calculation of the peaking factors of the secondary mid-/high-Z impurity Z3 with respect to the main element Z0.*

9. **Computation of plasma composition**. This is iterative and (at present) semi-automatic. It starts with the most basic assumptions and then relies on the user understanding the results of the various consistency checks and taking decisions on the next steps (see steps 1-9 of section :ref:`concept`). *In the new version of the code it should be attempted to make the whole procedure as automatic as possible. The user will anyway have to go through the data consistenty checks, decide if the working assumptions give consistent results or if modifications are needed. The most delicate part is the extrapolation beyond the detection range of the SXR detectors which requires fitting to the total radiated power while still accounting for the contributions of the different elements.*


.. _computation:

Computation details
---------------------------------

1. **Unfolding the SXR lines-of-sight** is relatively strightforward for poloidally symmetric emissivity profiles. For poloidally asymmetric profiles, the assumption is that local emissivity distribution on a flux-surface follows the same physics as the impurity density described by equation 1 of `M. Sertoli et al. Review of Scientific Instruments 89, 113501 (2018) <https://doi.org/10.1063/1.5046562>`_. With this in mind, the fitting must search for optimal **ϵ_SXR(ρ,R_0;t)** and **λ_SXR(ρ;t)** profiles that match the LOS integrals (where ρ = is an array in range [0, 1], R_0 indicates the reference major radius, typically the LFS midplane). 	Complete symmetry means λ_SXR = 0. Below are a few details for the computation:

	a) **Spline knots** in range ρ = [0, 1]. To avoid overfitting, the spatial distance of the knots must be **dρ > dρ _los**, where dρ _los is the "spatial resolution" of the diagnostic, with **ρ _los**  the lines-of-sight impact parameters. Knot distance increases towards the edge.

	b) **Boundary conditions** and **prior assumptions** are:
			* ϵ_SXR(ρ = 1) = 0 (this can be treated as a constant and not a fitting parameter)
			* ϵ_SXR(ρ) >= 0
			* λ_SXR(ρ > 0.5) > 0 *where fast particle contributions are negligible*
			* derivatives at boundaries:
				* ϵ_SXR: 1st derivative(ρ = 0) = 0, 2nd derivative(ρ = 1) = 0
				* λ_SXR: 2nd derivative(ρ = 0) = 0, 2nd derivative(ρ = 1) = 0

	d) **first guesses** of ϵ_SXR and λ_SXR from the second time-point are equal to the results for the previous time-point.

2. **Unfolding the Bolometer lines-of-sight** can be performed using the same methodology as for the SXR, taking care to avoid any LOS viewing the divertor which cannot be described by the poloidal asymmetry formula. The only differences with SXR are:
	* **ϵ_BOLO(ρ = 1) != 0** i.e. the emissivity at the separatrix is a fit parameter.

3. **Spline fitting of profile data** (electron temperature and density, ion temperature, toroidal rotation, etc.) can be performed in a similar fashion using cubic splines. There should be a possibility to combine diagnostic data in a single spline fit, e.g. LIDAR and HRTS for the electron density, or HRTS and ECE for electron temperature).

	a) **Spline knots** in range ρ = [0, 1.05].  The high gradient region at the edge requires higher knot density for the pedestal region ρ = [0.85, 1.0].

	b) **Boundary conditions** and **prior assumptions**:
		* apart from toroidal rotation, values > 0 over the whole radial range
		* 1st derivatives at boundaries (ρ = 0 and ρ = 1.05) = 0
		* value(ρ = 1.05) = 0
