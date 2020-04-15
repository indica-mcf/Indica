Concept and workflow
==============================

This section of the documentation describes the original WSX code (up to March 2020) written in IDL by `Marco Sertoli <marco.sertoli@ukaea.uk>`_ .


Journal Publications
---------------------

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
* A second low-Z impurity (Z2) with concentration constant in time can be included as well.
* Bolometer measurements (LoS-integrals and tomographic reconstructions) are used to cross-check the results and define extrapolation limits beyond the limit of applicability determined by the SXR diagnostic filter function
* Toroidal rotation measurements or the mode frequency of MHD modes are used to cross-check the poloidal asymmetry of the main high-Z impurity assuming it is governed by centrifugal asymmetries
* There is the possibility of including a second high-Z impurity (Z3) with time evolution identical to that of Z0, but with scaled peaking following simplified neoclassical theory, asymmetry assuming centrifugal effects and scaled of a fixed multiplication factor for the whole time-range of analysis
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
8.	If missing total radiated power and/or HFS, top/bottom radiation, add Z3 (with mass lower than Z0) to fill in the gaps
9.	Perform consistency checks again…
