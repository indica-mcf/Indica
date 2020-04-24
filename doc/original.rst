Concept and workflow
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

Code concept
--------------

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

5. **Read SXR data**: user to decide which cameras/diagnostics to read:
	
	* **Read raw data** for all channels
	* **Downsample** to reference time axis t_unfold
	* **Choose which channels to keep** (*some channels may be faulty and should be discarded from the start*)

	*All other diagnostics are currently read later on, but could be read at this point or anyway using a common GUI. Below is a list of all other physical quantities and respective diagnostics currently included in the computation:*
	
	* **Total radiation**: KB5 
	* **Radiation tomographic reconstructions**: BOLT, B5NN, B5ML, B5MF, ... 
	* **Electron density**: HRTS, LIDR, KG10, ...
	* **Electron temperature**: HRTS, LIDR, KK3, ...
	* **Ion temperature**: CXRS, ...
	* **Toroidal rotation**: CXRS, ...
	* **Impurity concentration and effective charge (passive spectroscopy)**: KT7/3, KX1, KS6, CXRS, KS3, ...
	
	*For all diagnostics, the reading steps are similar, with slight differences if measurements are LOS-integrals or local:*
		
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
	
6. **Unfold SXR lines-of-sight** to generate a 2D poloidal map of local emissivities. Depending on the symmetry/asymmetry of the radiation patterns, two different routes are used (*The same methodologies can be applied to the Bolometry diagnostic avoiding LOS viewing the divertor. The functionality of using Bolometry instead of SXR as main driving diagnostic should be added to the new version of the code*):

	* **Symmety**: perform a simple **Abel inversion** of all available LOS
	* **Asymmetry**:
		a) Perform **Abel inversion** of a specified set of lines of sight of one camera (HFS viewing LOS of camera V)
		b) **Fit all LOS using equation 1** of | `M. Sertoli et al. Review of Scientific Instruments 89, 113501 (2018) <https://doi.org/10.1063/1.5046562>`_ searching for the best profiles of **ϵ_SXR(ρ,R_0;t)** and **λ_SXR(ρ;t)**. The local emissivity calculated in a) is used as starting point for ϵ_SXR, while λ_SXR is set to zero across the full radius. Quite strict boundery conditions for ϵ_SXR and λ_SXR are specified to avoid problems in the plasma centre and at the boundary (rho = 1) and radial smoothing is performed to avoid excessive gradients. The number of spline knots can be varied between 3-6 depending on gradients in the emissivity pattern and rate of asymmetry.