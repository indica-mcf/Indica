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

List of Data to Read
--------------------------------------
Grouped by measurement quantity, the diagnostics (identified with their DDA names) currently included in the program are:

* Electron density and temperature diagnostics: **HRTS, LIDR, KK3, KG10**
* Radiation: **SXR, KB5**
* Spectroscopy: **KS3, KT7/3**
* Charge exchange recombination spectroscopy: **CXRS**
* Tomographic reconstruction of total radiation: **BOLT, B5NN, B5ML, B5MF**
* Other tools/diagnostics: **analysis of MHD activity** through FFT and toroidal mode analysis of Mirnov coils, oscillation amplitudes of fast KK3 and SXR.

Below are the details of the data that has to be read for each of these and other quantities that have to be read or computed for a correct functioning of the program. Most of the diagnostic variables are stored in PPFs of which DDA and DTYPE are specified. When this is not the case, the source of this information will be specified in the column DDA:

* **Surf** = external databases to read LOS coordinates
* **Flush** = libraries for reading specific attributes of the equilibrium of mapping between coordinates
* **ASCII** = for quantities stored in ASCII files
* **User** = user-defined quantities

and DTYPE will simply be a variable name.

For Flush there is currently a Python 3 wrapper developed by `Bruno Viola <bruno.viola@ukaea.uk>`_ . The Surf database is a publicly available ASCII file */home/flush/surf/input/overlays_db.dat* and maintained by `David Taylor <David.Taylor@ukaea.uk>`_.


**Equilibrium**

.. list-table::
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	* 	- EFIT
		- RMAG, ZMAG
		- t
		- Major radius and Z of magnetic axis (m)
	*	- User
		- RHO_EQ
		- nrho_eq
		- Reference coordinate for the equilibrium data, nrho_eq = 101 (currently) = user defined # of points
	*	- "
		- THETA
		- ntheta
		- Array of poloidal angles [0, 2 pi) for calculation of flux-surface-averaged quantities etc., ntheta = 10 (currently) = user defined # of poloidal angles
	* 	- Flush
		- RHO_TOR
		- rho_eq, t
		- Conversion of reference radial coordinate rho = rho_poloidal to the normalized toroidal flux coordinate used in many modelling codes (*flush_getftorprofile*)
	* 	- "
		- VOL
		- "
		- Volume within flux surfaces defined by rho_eq (combination of *flush_getflux, flush_getvolume*)
	*	- "
		- MAJR_LFS, MAJR_HFS
		- "
		- Map of rho_eq on major radius R  at LFS and HFS (combination of *flush_getabsoluteflux, flush_getmagaxisflux, flush_getlcfsflux*)
	*	-
		- MINR_LFS, MINR_HFS
		- "
		- Minor radius on LFS and HFS, calculated from MAJR_LFS and MAJR_LFS
	* 	- "
		- RSEP, ZSEP
		- nsep, t
		- Separatrix major radius and z position arrays (m), nsep = 150 (currently) = user defined # of points (*Flush_getLCFSboundary*)
	* 	- Flush
		- BR, BZ, BT, BTOT
		- R, t
		- Radial (BR), vertical (BZ), toroidal (BT) components and total (Btot) magnetic field at the midplane, interpolated on [MAJR_HFS, MAJR_LFS] (*flush_getBr, flush_getBz, flush_getBt*)

|

**Electron density and temperature**

.. list-table:: HRTS and LIDR
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	* 	- HRTS, LIDR
		- NE
		- R, t
		- Electron density (m^-3), R (m) maj-R position of measurement, t (s) time of measurement
	* 	- "
		- DNE
		- "
		- Electron density error (m-3)
	*	- "
		- TE
		- "
		- Electron temperature (eV)
	* 	- "
		- DTE
		- "
		- Electron temperature error (eV)
	* 	- "
		- Z
		-  "
		- Z positions of measurements (m)

.. list-table:: KK3
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	*	- KK3
		- TE##
		- t
		- Electron temperature (eV) of each channel (## = channel_number, *identify from PPF DTYPEs starting with 'TE'*)
	*	- "
		- GEN
		-
		- Acquisition parameters including:

			* channel_index = np.argwhere(GEN[0, :] > 0);
			* f_chan = GEN[15, :] (GHz) = resonator frequency to calculate channel R_chan from B_tot;
			* nharm_chan = GEN[11, :] = measured harmonic of cyclotron frequency, necessary to calculate channel R_chan from B_tot;
			* cal_chan = GEN[18, :] and GEN[19, :] = channel calibrated if != 0 (if ==0, let user choose whether to use the data)

	*	- Surf
		- z
		-
		- z position of the viewing LOS (horizontal view). *Surf has slightly different vaues from Datahandbook = 0.1335 (m) for JPN < 80318, 0.2485 (m) for JPN > 80318*
	* 	- Flush
		- R##
		- Btot, t
		- Radial position of each channel calculated interpolating the total B-field along the LOS of the KK3 antenna with the B-field of cold resonance calculated using the electron cyclotron frequency formula with info from GEN (*flush_getBr, flush_getBz, flush_getBt*)

|

**Radiation**

.. list-table:: SXR
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	*	- SXR
		- V##, T##, H##
		- t
		- Brightness (W m^-2) of the each LOS (## = LOS_number) for cameras V, T, H (user to choose which to read)
	*	- Surf
		- (R, z)
		- channel, nlos
		- Coordinates (m) of all LOS. Identifying string in Surf database is respectively 'KJ3-4 V', 'KJ3-4 T', 'KJ5', nlos = 100 (currently) = number of points along each los, chosen with identical equally spaced steps for all LOS.

.. list-table:: KB5
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	*	- BOLO
		- KB5V, KB5H
		- channel, t
		- Brightness (W m^-2) of all LOS (channel_number = channel_index + 1) for cameras V, H (user to choose which to read)
	*	- Surf
		- (R, z)
		- channel, nlos
		- Coordinates (m) of all LOS. Identifying string in Surf database is  'KB5'

**Spectroscopy**

.. list-table:: KS3
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	* 	- KS3
		- ZEFH, ZEFV
		- t
		- Zeff measurements from horizontal and vertical lines-of-sight
	* 	- EDG7
		- LOSH, LOSV
		-
		- Info on LOS coordinates (mm) for KS3 measurements: R_start = LOSH[1], R_end = LOSH[4], z_start = LOSH[2], z_end = LOSH[5], same for LOSV

|

**Charge exchange recombination spectroscopy**

.. list-table::
	:widths: 5 15 10 60
	:header-rows: 1

	* 	- DDA
		- DTYPE
		- Axes
		- Description
	* 	- Many DDAs, e.g. CXG6
		- TI
		- x, t
		- Ion temperature (eV), x_cxrs is an *effective* position of measurement in the torus frame (m), but the correct R is RPOS (see below)
	*	- "
		- TIHI, TILO
		- "
		- Upper, lower TI limits (eV): TI_ERR = (TIHI - TILO)/2.
	*	- "
		- ANGF
		- "
		- Angular rotation frequency (rad)
	*	- "
		- AFHI, AFLO
		- "
		- Upper, lower ANGF limits (rad): ANGF_ERR = (AFHI - AFLO)/2.
	*	- "
		- CONC
		- "
		- Concentration (%) of measured impurity
	*	- "
		- COHI, COLO
		- "
		- Upper, lower CONC limits (rad): CONC_ERR = (COHI - COLO)/2.
	*	- "
		- RPOS
		- "
		- R position of measurement (m)
	*	- "
		- POS
		- "
		- z position of measurement (m)
	*	- "
		- MASS
		-
		- Atomic mass of measured impurity
	*	- "
		- TEXP
		- t
		- Exposure time (s)

Additionally to these, all measurement coordinates and LOS will have to be converted from (R, z) to rho using Flush (*flush_getabsoluteflux, flush_getmagaxisflux, flush_getlcfsflux*). Both coordinate systems should be saved for future use.

LOS coordinates shouldn't be just the start and end of the LOS, but arrays of values along the LOS which can then be used for performing integrals and other operations, both in (R, z) and rho.

|

**ADAS atomic data files**

For each element included in the analysis, files must be read for: ionization (SCD) and recombination rates (ACD), total radiation loss parameters from spectral lines (PLT) and recombination/Bremsstrahlung (PRB), SXR-filtered radiation loss parameters from spectral lines (PLSX) and recombination/Bremsstrahlung (PRSX), SXR-filter function.

ADAS filenames are standard (e.g. scd96_he.dat) and include a class identifier (e.g. “scd” for ionization rate coefficients), a year identifier (e.g. “96”) and the element, fully in lower-case (e.g. "he") with an underscore in front. All of the files used are stored in the official ADAS repository, apart from SXR-filtered radiation loss parameter files which are often built locally starting from SXR-filter function files (labelled e.g. ‘sxrfil5.dat’). Care must be taken to avoid confusion since on different machines these files can have the same name, but different filter characteristics (e.g. AUG and JET have 75 um and 250 um filters but the files are all labelled plsx5, prsx5, sxrfil5).

The files reported below are the ones currently used. The SXR files are JET-specific for 250 um Be-windows. When using the code on other machines, all files will be the same apart from the SXR-filtered radiation loss parameter files.

.. list-table::
	:widths: 15 55 25
	:header-rows: 1

	* 	- Element
		- Files
		- Comment
	*	- H
		- scd96, acd96, plt96, prb96, plsx5, prsx5
		-
	* 	- He
		- acd96, scd96, plt96, prb96, plsx5, prsx5
		-
	*	- Be
		- acd96, scd96, plt96, prb96, plsx5, prsx5
		-
	* 	- N
		- acd96, scd96, plt96, prb96
		- No SXR-filtered data for 250 um filter
	* 	- Ne
		- acd96, scd96, plt96, prb96, plsx5, prsx5
		-
	* 	- Ar
		- acd85, scd85, plt00, prb00, plsx5, prsx5
		-
	*	- Fe
		- acd00, scd00, plt00, prb00, plsx5, prsx5
		-
	*	- Ni
		- acd89, scd89, plt01, prb00, plsx5, prsx5
		-
	*	- Mo
		- acd89, scd89, plt89, prb89
		- No SXR-filtered data for 250 um filter
	* 	- W
		- acd89, scd89, plt89, prb89, plsx5, prsx5
		- Other files from Thomas Pütterich are currently used. Effort to use official ADAS files is under way.

*For historical reasons, all of these files are currently locally stored in the ../atomdat/ directory of the* `STRAHL program <https://pure.mpg.de/rest/items/item_2143869/component/file_2143868/content>`_ . *In the new version, it will be worthwhile to decouple this from STRAHL and have a local repository within the main program directory.*
