Input Data Details
==============================

Traceability
-----------------
**Data/parameters read from PPF or file must be traceable to its origin**. Raw data should be stored for future referencing, together with their DDA and UID names and SEQ #, or file name for data contained in ASCII/binary files. This ensures tracking of data origin, the possibility to vary calculation/fitting parameters without re-reading the raw data and enables consistency checks vs. derived data e.g.: 


* Demonstrate the quality of the fits to electron/ion profile data (HRTS, LIDAR, KK3, CXRS, etc.); 

* Demonstrate goodness of inversion algorithm or forward calculation for line-of-sight (LOS) integrated data (SXR, bolometry, passive spectroscopy, etc.).

* Cross-check toroidal rotation calculated from W density asymmetry vs. the experimental CXRS measurement


Time resolution
-----------------
**Time resolution required for the analysis is typically 10-50 ms**. Global time array is built given the time window of analysis [t1, t2] and the desired time resolution. All derived quantities are calculated on this axis, which is different from the time resolution of the raw data of each diagnostic which will have to be interpolated accordingly.


Reference radial array
--------------------------
**Radial array = normalised poloidal flux coordinate (rho), regular grid [0, 1]**. A reference radial array, fixed in time, is used to map all derived and fitted profiles. Currently, a regular grid of 41 points is used across the minor radius, leading to a central resolution of ~ 2 cm which increases at the edge. 

issues: in view of future upgrades, e.g. the inclusion of fast transport codes in the loop to resolve the pedestal region, it might be worthwhile to increase the resolution at the edge for rho > 0.85. 


Equilibrium data
----------------------
As all other PPF data, it is saved to a raw-data variable in its original time resolution. When calling the FLUSH library (currently not working in Python 3) for conversion of equilibrium quantities (e.g. (R,z)↔ψ flux), the data is retrieved for the equilibrium time-point in closest proximity to the requested time-point. This pattern has been kept inside the code as well.


Because of uncertainties in equilibrium-calculated geometrical quantities such as separatrix position (uncertainties in magnetic measurements) or magnetic axis position (uncertainties in the plasma pressure at runtime), a possibility of an (R,z) shift of the whole equilibrium should be included. Since shifting the equilibrium is impossible without re-running the equilibrium code, an alternative solution is to shift ALL diagnostic data of the same amount, but in the opposite direction. The (R,z) shift variables should be time-dependent and included in the equilibrium data since it is a quantity that has to be applied equally to all diagnostics.


Conversion between coordinates is required for most diagnostic data and equilibrium quantities: 


* (R,z)↔(rho,t) diagnostics’ measurement positions. For HRTS, LIDAR, CXRS these are local measurements; for KB5 (bolometry), SXR or other line-of-sight (LOS) integrated measurements these are array of values along the LOS. For LOS coordinates, a fixed number of points along all LOS with a constant linear step between points is used to keep a constant spatial resolution along the LOS. Better ways of doing this?

This transformation is also used to calculate a 2D time-dependent poloidal map of rho over a fixed * (R,z) grid used for the 2D maps.

* (rho,ϑ)↔(R,z,t) where rho is the reference array, ϑ is the poloidal angle (both time-independent). Subset of these transformations that are used very often:


	* (rho,ϑ=0)↔(R_LFS,t) = major radius R, LFS midplane (at z_ax)

	* (rho,ϑ=π)↔(R_HFS,t) = major radius R, HFS midplane (at z_ax)


* rho↔Vol = volume, for the estimation of volumetric integrals of e.g. total radiation, impurity densities, etc.

* (rho,ϑ)→minor radius depending on the application, one may want to use values at the LFS ϑ=π or HFS ϑ=π, HFS-LFS averaged values, flux-surface-averaged values ϑ=[0,2π). All are currently used in the code. 

* rho= rhop↔ rhot(t) i.e. conversion from normalized poloidal flux coordinate to normalized toroidal flux coordinate (modelling codes typically use either one or the other). In the current version there is the possibility of choosing which coordinate (rhop or rhot) to use as main (time-independent) rho coordinate, and which one as subordinate (time-dependent).

* …


Other quantities that should be read from the equilibrium reconstruction are:


* (R_sep,z_sep,t) of the separatrix coordinates (for each time-point R_sep and z_sep are arrays) 

* (R_ax,z_ax,t) = position of the magnetic axis (for each time-point R_ax and z_ax are scalars)

* B_tot (R,t) = total magnetic field (for the calculation of the ICRH resonance position, re-mapping of the ECE data, etc.)

* …


Basic diagnostic data structure
-------------------------------------------
Apart from the measured quantity and its estimated error, specific info varies depending on the diagnostic measurement principle. Since the analysis of the data from all diagnostics is performed in an integrated manner on the same equilibrium reconstruction, the raw data should include the least possible derived quantities relying on other PPFs. As an example, the measurement positions of the ECE diagnostic KK3 given in the PPF rely on a specific equilibrium PPF: instead of using these quantities, the frequency of each channel is read and the radial location self-consistently calculated within the code using the common equilibrium reconstruction data.


Below is an initial list of information ordered depending on the source (database/file) and details of the diagnostics.


* UID, DDA names and SEQ for PPF data e.g. (JETPPF,HRTS,350)

* File name, path and origin (e.g. name of originator, publication details if published) for ASCII (e.g. ADAS ionization and recombination files), NetCDF (TRANSP) or other file types. Origin is especially important to trace data-files present on a private repository and not available on a central database.

* (R,z) of all local measurement positions (e.g. HRTS, LIDAR, CXRS, magnetic probes, etc.) or (R,z) array of lines-of-sight (e.g. KB5, SXR, KT7/3, KX1, etc.)

* f_channel = acquisition frequency of each channel of the resonators (reflectometry or radiometry diagnostics e.g. KK3, KG10)

* n_harm = harmonic of cyclotron frequency (reflectometry or radiometry diagnostics)

* (rho,t) calculated with the methods defined above for the equilibrium

* min_rho for LOS only: it’s called “impact parameter”  and it’s the minimum distance (in rho) of each LOS from the magnetic axis. A different sign is typically assigned to LOS on opposite sides of the magnetic axis to avoid that they have same impact parameter. The current convention is -1 for vertical LOS viewing to the HFS of the magnetic axis and for horizontal LOS viewing below the magnetic axis.

* min_rad as above, but the minimum distance is in physical space (in cm or m)


For JET diagnostics, if the information on the LOS or measurement positions is not available in the PPFs, it is taken from the central database built for SURF (RO is David Taylor). It may not be the most optimized database, but it is a standard and is consistent with what users see on the program SURF. The information therein has been checked for most diagnostics with their ROs.

.. _dtype:

List of DDA and DTYPEs
----------------------------

Grouped by measurement quantity, the diagnostics (identified with their DDA names) currently included in the program are:

* Electron density and temperature diagnostics: **HRTS, LIDR, KK3, KG10**
* Radiation: **SXR, KB5**
* Spectroscopy: **KS3, KT7/3**
* Ion temperature and toroidal rotation: **CXRS**
* Tomographic reconstruction of total radiation: **BOLT, B5NN, B5ML, B5MF**
* Other tools/diagnostics: **analysis of MHD activity** through FFT and toroidal mode analysis of Mirnov coils, oscillation amplitudes of fast KK3 and SXR.

Below are the details of the data-types (DTYPE) that have to be read for each diagnostic (DDA). These DTYPEs can be read durectly from the diagnostic PPF unless otherwised specified (e.g. *Flush* or *SURF*). Flush is currently not available for Python 3, but there are privately developed wrappers that can be used/modified. The SURF database is a publicly available ASCII file */home/flush/surf/input/overlays_db.dat*.

.. list-table:: Title
	:widths: 5 5 10 60
	:header-rows: 1
	
	* 	- DDA
		- DTYPE
		- Axes
		- Description
	* 	- EFIT
		- RMAG
		- t_eq
		- Major radius of magnetic axis (m)

	* 	- 
		- ZMAG
		- t_eq
		- Z of magnetic axis (m)

	* 	- 
		- RSEP
		- n_sep, t_eq
		- Separatrix major radius array (m), n_sep = user defined # of points > *Flush*
	* 	- 
		- ZSEP
		- n_sep, t_eq
		- Separatrix Z array (m), n_sep = user defined # of points  > *Flush*
	* 	- 
		- BR, BZ, BT
		- R, t_eq
		- Components of the magnetic field at the midplane (z = z_ax = ZMAG)  > *Flush*
	* 	- HRTS
		- NE
		- R_hrts, t_hrts
		- Electron density (m^-3)
	* 	- 
		- DNE	
		- R_hrts, t_hrts	
		- Electron density error (m-3)

	*	- 
		- TE	
		- R_hrts, t_hrts
		- Electron temperature (eV)

	* 	- 
		- DTE	
		- R_hrts, t_hrts
		- Electron temperature error (eV)

	* 	- 
		- Z	
		- R_hrts, t_hrts

		- Z positions of measurements (m)
	* 	- LIDR
		- NE	
		- R_lidr, t_lidr
		- Electron density (m-3)

	* 	- 
		- DNE	
		- R_lidr, t_lidr	
		- Electron density error (m-3)

	* 	- 
		- TE	
		- R_lidr, t_lidr
		- Electron temperature (eV)

	* 	- 
		- DTE	
		- R_lidr, t_lidr
		- Electron temperature error (eV)

	*	-
		- Z	
		- R_lidr, t_lidr
		- Z positions of measurements (m)
	
	*	- KK3
		- GEN
		- 20, 96
		- Acquisition parameters including: 
			
			* *channel_index = np.argwhere(GEN[0, :] > 0)*
			* *f_chan = GEN[15, :] (GHz) = resonator frequency to calculate channel R_chan from B_tot*
			* *nharm_chan = GEN[11, :] = measured harmonic of cyclotron frequency, necessary to calculate channel R_chan from B_tot*
			* *cal_chan = GEN[18, :] and GEN[19, :] = channel calibrated if != 0 (if ==0, let user choose whether to use the data)*
	*	- 
		- TE##
		- t_kk3
		- Electron temperature (eV) of each channel (## = channel_number = channel_index + 1)
	*	- 
		- z 
		- 
		- **NOT IN PPF**: z position of the viewing LOS (horizontal view) *> SURF database has slightly different vaues from Datahandbook = 0.1335 (m) for JPN < 80318, 0.2485 (m) for JPN > 80318*
	*	- SXR
		- V##, T##, H##
		- t_sxr
		- Brightness (W m^-2) of the each LOS (## = LOS_number) for cameras V, T, H (user to choose which to read)
	*	-
		- (R, z)
		- 
		- **NOT IN PPF** coordinates (m) of each LOS *> SURF database*
	*	- BOLO
		- KB5V, KB5H
		- channel_index, t_kb5
		- Brightness (W m^-2) of the each LOS (channel_number = channel_index + 1) for cameras V, H (user to choose which to read)
	*	-
		- (R, z)
		- 
		- **NOT IN PPF** radial coordinates (m) of each LOS *> SURF database*
	* 	- KS3
		- ZEFH, ZEFV
		- t_ks3
		- Zeff measurements from horizontal and vertical LOS
	* 	- EDG7
		- LOSH, LOSV
		- 
		- Info on LOS coordinates (mm) for KS3 measurements: R_start = LOSH[1], R_end = LOSH[4], z_start = LOSH[2], z_end = LOSH[5], same for LOSV
	* 	- CXRS *(many DDAs)*
		- TI
		- x_cxrs, t_cxrs
		- Ion temperature (eV), x_cxrs is an *effective* position of measurement in the torus frame (m), but the correct R is RPOS (see below)
	*	-
		- TIHI, TILO
		- x_cxrs, t_cxrs
		- Upper, lower TI limits (eV): TI_ERR = (TIHI - TILO)/2.
	*	-
		- ANGF
		- x_cxrs, t_cxrs
		- Angular rotation frequency (rad)
	*	-
		- AFHI, AFLO
		- x_cxrs, t_cxrs
		- Upper, lower ANGF limits (rad): ANGF_ERR = (AFHI - AFLO)/2.
	*	-
		- CONC
		- x_cxrs, t_cxrs
		- Concentration (%) of measured impurity
	*	-
		- COHI, COLO
		- x_cxrs, t_cxrs
		- Upper, lower CONC limits (rad): CONC_ERR = (COHI - COLO)/2.
	*	-
		- RPOS
		- x_cxrs, t_cxrs
		- R position of measurement (m) *time-dependent since it depends on which PINI is on*
	*	-
		- POS
		- x_cxrs, t_cxrs
		- z position of measurement (m) *time-dependent since it depends on which PINI is on*
	*	-
		- MASS
		- 
		- Atomic mass of measured impurity
	*	-
		- TEXP
		- t_cxrs
		- Exposure time (s)
