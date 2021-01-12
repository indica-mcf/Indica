Concept and Workflow
==============================

This section of the documentation describes the original WSX code (up to March 2020) written in IDL by `Marco Sertoli <marco.sertoli@ukaea.uk>`_ .


Documentation and Publications
------------------------------------------

| The main features of the code in its various versions since its initial conception are documented in the following papers:
| `M. Sertoli et al. J. Plasma Phys. 85, 905850504 (2019) <https://doi.org/10.1017/S0022377819000618>`_
| `M. Sertoli et al. Review of Scientific Instruments 89, 113501 (2018) <https://doi.org/10.1063/1.5046562>`_
| `M. Sertoli et al. Plasma Phys. Control. Fusion 57 075004 (2015) <https://doi.org/10.1088/0741-3335/57/7/075004>`_

| Other publications in which the results of the method have been used are:
| `F.J. Casson et al.  Nucl. Fusion in press 2020 <https://doi.org/10.1088/1741-4326/ab833f>`_
| `A. Field et al. Plasma Phys. Control. Fusion 62 055010 (2020)  <https://doi.org/10.1088/1361-6587/ab7942>`_
| `S. Gloeggler et al. Nuclear Fusion 59, 126031 (2019) <https://doi.org/10.1088/1741-4326/ab3f7a>`_
| `O. Lindner et al. Nuclear Fusion 59, 016003 (2018) <https://doi.org/10.1088/1741-4326/aae875>`_
| `S. Breton et al. Nuclear Fusion 58, 096003 (2018) <https://doi.org/10.1088/1741-4326/aac780>`_
| `S. Breton et al. Physics of Plasmas 25, 012303 (2018) <https://doi.org/10.1063/1.5019275>`_
| `C. Angioni et al. Nuclear Fusion 57, 056015 (2017) <https://doi.org/10.1088/1741-4326/aa6453>`_
| `C. Angioni et al. Physics of Plasmas 24, 112503 (2017) <https://doi.org/10.1088/1741-4326/aa6453>`_
| `M. Goniche et al. Plasma Physics and Controlled Fusion 59, 055001 (2017) <https://doi.org/10.1088/1361-6587/aa60d2>`_
| `M. Sertoli et al. Nuclear Fusion 55, 113029 (2015) <https://doi.org/10.1088/0029-5515/55/11/113029>`_
| `P. Piovesan et al. Plasma Physics and Controlled Fusion 59, 014027 (2016) <https://doi.org/10.1088/0741-3335/59/1/014027>`_

Scope, Inputs and Outputs
------------------------------------------
The main scope of the code is to evaluate the plasma composition combining a large set of measurements from different diagnostics. The code calculates the time evolution of the density profiles of 3(+1) impurities: one low-Z and two mid-/high-Z elements (+ one additional background low-Z impurity, constant in time). Mid-Z and high-Z elements are resolved in 2D on a poloidal plane to account for poloidal asymmetries.

The main inputs necessary for the computation are:

* equilibrium information and flux-surface mapping libraries
* atomic data (ionization balance and radiative loss parameters)
* diagnostic data for the following quantities:
	* electron temperature and density profiles
	* ion temperature and toroidal rotation
	* SXR radiation
	* total radiation
	* effective charge
	* passive spectroscopy impurity concentration measurements
	* fast magnetic measurement (for MHD investigations)

The output of the code is the time evolution of:

* 2D poloidal maps of the mid-Z and high-Z impurities (e.g. Ni and W, for JET)
* Concentration of the low-Z impurities
* Zeff profile
* Main ion density profile (dilution)
* 2D poloidal map of the total radiation

These resuzlts are checked for consistency against various parameters/measurements including:

* Total radiation vs. bolometry LOS-integrals
* High-Z poloidal asymmetry vs. measured toroidal rotation
* Low-Z impurity concentration vs. CXRS measurements
* High-Z impurity concentration vs. passive spectroscopy estimates

The outputs of the code can be used for further analysis (e.g. the calculation of impurity transport coefficients), comparison with theoretical estimates (e.g. impurity poloidal asymmetries), can be fed into modelling codes (e.g. turbulence modelling, TRANSP modelling), comparison and benchmarking of diagnostic data.


Concept
----------------------------

The concepts behind the code are thoroughly explained in the cited references, but it is useful to summarize here the main assumptions and features:

* The SXR diode diagnostic is the main tool and starting point for the analysis
* The shape of the SXR emissivity profile is dominated by one high-Z impurity (Z0)
* An independent measurement (VUV) of the concentration of the main impurity (Z0) is used to re-scale the first guess of its density (optional)
* A Zeff measurement is used to calculate the concentration of a low-Z impurity (Z1)
* A second low-Z impurity (Z2) with concentration constant in time can be included
* Bolometer measurements (LoS-integrals and tomographic reconstructions) are used to cross-check the results and define extrapolation limits beyond the limit of applicability determined by the SXR diagnostic filter function
* Toroidal rotation measurements or the mode frequency of MHD modes are used to cross-check the poloidal asymmetry of the main high-Z impurity assuming it is governed by centrifugal asymmetries
* A second high-Z impurity (Z3) can be included, with time evolution identical to Z0, but with scaled peaking following simplified neoclassical theory, asymmetry assuming centrifugal effects and scaled of a fixed multiplication factor for the whole time-range of analysis
* Calculation of the impurity transport coefficients of the main impurity Z0 using the Gradient-Flux relation on sawtooth cycles.
* Correlation with MHD activity is performed by analysing:

    * MHD toroidal mode numbers and mode amplitude of core MHD using toroidal magnetic sensors
    * Sawtooth inversion radius and ICRH resonance layer(s)
    * 2D reconstruction of the electron temperature from ECE rotational tomography
    * Profiles of oscillation amplitude and phase of SXR channels due to MHD activity

The inclusion of multiple impurities is performed in a stepwise fashion:

1.	First guess of Z0 re-scaled to the VUV concentration measurement
2.	Inclusion of Z1 using Zeff
3.	Calculation of main ion density from quasi-neutrality
4.	Re-estimation of Z0 using Z1 and main ion contributions to SXR (changes mainly the shape, not much the absolute value because of the rescaling to the VUV Z0 concentration measurement)
5.	Consistency-check of:

 	a.	Final estimate of low-Z makes sense (e.g. 10% of Be is way off…)
	b.	Z0 density asymmetry vs. toroidal rotation or MHD mode frequency
	c.	Total radiated power (Z0 + Z1 + main ion) vs. LoS-integrals of bolometry

6.	If rotation estimate is wrong, start over with different main impurity Z0
7.	If shape of LoS-integrals of bolometry misses features on the LFS-midplane, extrapolate to fit the shape
8.	If missing total radiated power and/or HFS, top/bottom radiation, add Z3 to fill in the gaps
9.	Perform consistency checks again…

Workflow
----------------
In this section, the steps of code execution are outlined in detail. The names used here are JET-specific, but the concept will be exactly the same on other fusion devices, whether they are Tokamaks or Stellaroators.

1. **JPN, tr, dt** : choose pulse # (JPN) to analyse, time-range of interest (tr) and time resolution (dt = typically 10-50 ms) for analysis.

2. Create reference time axis **t_unfold** with desired resolution dt in range tr.

3. **Equilibrium UID and DDA** : define equilibrium diagnostic and read diagnostic-independent geometric quantities often used throughout the code (DTYPES details given in section :ref:`dtype`):

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
	* **Choose electron density and temperature diagnostics** with independent input of UID and DDA names for Ne and Te
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

1. **Unfolding the SXR lines-of-sight** is strightforward for poloidally symmetric emissivity profiles using e.g. Abel inversion. For poloidally asymmetric profiles, the assumption is that local emissivity distribution on a flux-surface follows the same physics as the impurity density described by equation 1 of `M. Sertoli et al. Review of Scientific Instruments 89, 113501 (2018) <https://doi.org/10.1063/1.5046562>`_. With this in mind, the fitting must search for optimal **ϵ_SXR(ρ,R_0;t)** and **λ_SXR(ρ;t)** profiles that match the LOS integrals (where ρ = is an array in range [0, 1], R_0 indicates the reference major radius, typically the LFS midplane). Below are a few details for the computation:

	a) **Spline knots** in range [0,1]. Number of knots must avoid overfitting: the spatial distance between knots **dρ** must be >= the spatial resolution of the diagnostic ~ difference between impact parameters **ρ _los** of the neighbouring LOS **dρ _los**. Typical values of dρ range from **dρ _los x 2** for extremely shaped profiles, to **dρ _los x 6** for cases with lower asymmetry and peaking). Spatial resolution of the knots increases towards outer radii to enable fitting of more shaped profiles.

	b) **Boundery conditions** and **prior assumptions** are:
			* ϵ_SXR(ρ = 1) = 0 
			* ϵ_SXR(ρ) >= 0
			* λ_SXR(ρ > 0.5) > 0 *where fast particle contributions are negligible*
	where complete symmetry means λ_SXR = 0 for all rho. Strong central peaking means λ_SXR(ρ=0) --> 0. Spline boundary conditions for ϵ_SXR and λ_SXR 
			
	c) **Smootheness** parameters take care to avoid extreme gradients of both ϵ_SXR and λ_SXR close to the plasma centre where the diagnostic is less sensitive, as well as outside of the viewing region of most edge LOS where only indirect effects of the emissivity profile shape are detected. The current fitting routine optimizes ϵ_SXR and λ_SXR scanning their values from outer to inner radii. This is because the emissivity in outer flux surfaces affects all lines of sight, while the emissivity from inner surfaces affects only those LOS crossing this space.

	d) the **first guesses** of ϵ_SXR and λ_SXR from the second time-point onwards are the results calculated for the previous time-point. If the chi-sq resulting from this optimisation is not good enough, then the asymmetry parameter is reset to λ_SXR = 0 and a second round of optimisation is performed.
	