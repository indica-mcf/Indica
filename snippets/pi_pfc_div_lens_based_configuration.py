from matplotlib import pyplot as plt, cm
from matplotlib.colors import LogNorm

import indica
import numpy as np
import scipy as sp  
from scipy.signal import find_peaks
import os
import xarray as xr
import math
import json

from indica import Plasma, PlasmaProfiler
from indica.models.passive_spectrometer import PassiveSpectrometer, read_adf15s, format_pecs
from indica.defaults.load_defaults import load_default_objects
from indica.profilers.profiler_gauss import initialise_gauss_profilers
from indica.converters import LineOfSightTransform
from indica.readers import SOLPSReader, ST40Reader

import sys
sys.path.insert(1, "/home/lorenzo.martinelli/st40_phys_viewer/st40_phys_viewer/standard_plots")
from st40_phys_viewer.standard_plots.plot_elmag import get_vessel_geometry_data


save_dir = "PI_PFC_DV1/lens_configuration"

#############################################################################
######## CHOOSE THE SPECTRAL LINES TO BE INCLUDED IN THE SIMULATION #########
#############################################################################

config = {
    # "c": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 6)},
    "c": {
        ** {str(charge): dict(file_type = "vsu", year = "96") for charge in range(0, 3)},
        ** {str(charge): dict(file_type = "pju", year = "96") for charge in range(4, 6)},
        #**{str(charge): dict(file_type = "vsu", year = "96") for charge in range(0,3)},
        #**{str(charge): dict(file_type = "pjr", year = "93") for charge in range(3,5)},
        #    "5":        dict(file_type = "bnd", year = "96"),
            },    
    "h": {str(charge): dict(file_type="pju", year="12") for charge in range(0, 1)},
    # "he": {
    #      "1": dict(
    #          file_type="bnd",
    #          year="96",
    #      ),
    # },
    # "mo": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},
    # "b": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},
    "li": {str(charge): dict(file_type="pju", year="96") for charge in range(0, 3)},

    # "ar": {
    #     str(charge): dict(file_type="llu", year="transport") for charge in range(16, 18)
    # },
}
# spectrum (nm)
window = np.linspace(360, 675, 630)

adf15  = read_adf15s(elements=config.keys())

pecs   = format_pecs(adf15, wavelength_bounds=slice(window.min(),
                                                     window.max()),)
                    #  electron_density_bounds=slice(1e18, 2e20),
                    #  electron_temperature_bounds=slice(10, 5000), )


#############################################################################
################## TRANSMISSIVITY CALCULATED WITH ZEMAX #####################
#############################################################################

# calculated directly on the last detector, therefore including:
# - the 3 lenses transmission
# - the viewport transmission (N-BK7 glass)

wl_transmissivity  = np.array([   370,    380,    388,    397,    410,    420,    440, 
                                  460,    480,    500,    520,    540,    560,    580, 
                                  600,    620,    640,    660,    680])
val_transmissivity = np.array([0.0727, 0.3446, 0.5488, 0.7023, 0.8053, 0.8383, 0.8517,
                               0.8517, 0.8763, 0.8838, 0.8852, 0.8858, 0.8856, 0.8873,
                               0.8892, 0.8909, 0.8915, 0.8873, 0.8788])


# data scraped with Copilot for Kaya camera quantum efficiency curve
# https://storage.kaya.vision/s/Iron-5514bsi-documentation?dir=undefined&_gl=1*1gl0mxv*_ga*NjM5NDU0NzYuMTc2MDUxNjEyOQ..*_ga_1RKQ68SKPW*czE3NzE2MDExMjAkbzEwJGcwJHQxNzcxNjAxMTIwJGo2MCRsMCRoMA..&openfile=1553860
wl_quantum_efficiency = np.array([360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460,
                                  470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 
                                  580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680])

quantum_efficiency    = np.array([ 45,  50,  55,  58,  60,  63,  66,  69,  72,  74,  76,
                                   78,  79,  80,  81,  82,  82,  82,  81,  80,  79,  77,
                                   75,  72,  70,  67,  64,  60,  56,  52,  48,  44,  40]) * 1e-2

# fiber attenuation curve scraped from https://www.ceramoptec.com/products/optical-fibers/#top (brochure) for Optran UV fibers
fiber_losses_dbkm_wl  = np.array([333, 366, 400, 433, 466, 500, 533, 566, 600, 633, 666,  700])
fiber_losses_dbkm     = np.array([110,  78,  41,  30,  24,  18,  16,  12,  16, 9.8,  9.8,  17])
fiber_transmission_20m= (10 ** (-fiber_losses_dbkm * 0.02 / 10)) # convert from dB/km to transmission over 20m


# interpolate over the wavelength range of interest
optical_transm_interp = np.interp(window, wl_transmissivity, val_transmissivity)
quantum_eff_interp    = np.interp(window, wl_quantum_efficiency, quantum_efficiency)
fiber_transm_interp   = np.interp(window, fiber_losses_dbkm_wl, fiber_transmission_20m)


print("printing transmission curves figure...")
# Plot the transmissivity curve
plt.figure()
plt.plot(wl_transmissivity, val_transmissivity, "o", label="Optical transmission - Zemax data")
plt.plot(window, optical_transm_interp, "-", label="Optical transmission - Interpolated")
plt.plot(wl_quantum_efficiency, quantum_efficiency, "o", label="Quantum efficiency - Kaya data")
plt.plot(window, quantum_eff_interp, "-", label="Quantum efficiency - Interpolated")
plt.plot(fiber_losses_dbkm_wl, fiber_transmission_20m, "o", label="Fibertransmission - CeramOptec Optran UV data")
plt.plot(window, fiber_transm_interp, "-", label="Fiber transmission - Interpolated")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmissivity")
plt.title("Transmission of PI_PFC_DV1 optical chain")
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, "transmissivity_curve.png"), dpi=500)
plt.show()

#############################################################################
########## UTILITY & IDENTIFICATION FUNCTIONS ################################
#############################################################################

def roman_numeral_limited(n):
    """
    Convert an integer to a Roman numeral, limited to common ones.
    Used for ionization stage labels (e.g., I, II, III).

    The algorithm is simple: start from largest allowed value and subtract
    repeatedly, building the string.
    """
    # - list of values and corresponding symbols (VI down to I)
    # - loop through values subtracting as many times as possible
    # - concatenate the matching roman symbol each time
    # This is purely a helper for labelling plots, not a physics calculation.
    val   = [6,      5,    4,     3,    2,   1]
    syms  = ["VI", "V", "IV", "III", "II", "I"]
    roman = ''
    for i in range(len(val)):
        count = int(n / val[i])
        roman += syms[i] * count
        n -= val[i] * count
    return roman

def match_species_verbose(wavelength, adf15_data, tolerance=0.5):
    """
    Match a given wavelength to species in ADAS data.
    Returns the species name if found within tolerance, else None.

    Steps:
    - iterate over each element present in the ADAS dictionary
    - for every ion charge state, check if wavelength data exists
    - compute an absolute difference array and look for matches within
      the specified tolerance (nm)
    - if a match is found, convert the charge to roman numeral for labelling
      and return a human-readable string like "C II" or "LI I"
    - if nothing matches, return None (line is unidentified)
    """
    for element, charges in adf15_data.items():
        for charge, data in charges.items():
            if "wavelength" in data.coords:
                wl_array = data.wavelength.values
                matches = np.abs(wl_array - wavelength) < tolerance
                if np.any(matches):
                    ion_stage = int(charge)
                    species = f"{element.upper()} {roman_numeral_limited(ion_stage + 1)}"
                    return species
    return None


#############################################################################
########## SPECTRAL ANALYSIS FUNCTIONS ######################################
#############################################################################

def find_prominent_spectral_lines(spectra, adf15_data, prominence=1e-6, height = 1e10, top_n=5):
    """
    Detect the most prominent spectral lines in the spectra and identify them.

    Parameters:
    - spectra: xarray.DataArray with dimensions (channel, wavelength)
    - adf15_data: ADAS data for species identification
    - prominence: minimum prominence for peak detection
    - height: minimum height for peak detection
    - top_n: number of top lines to return per channel

    Returns:
    - dict: key=channel, value=list of dicts with 'wavelength', 'intensity', 'species'

    Procedure:
    - loop over each LOS channel in the spectra
    - extract the 1‑D spectrum and wavelength axis
    - call scipy.signal.find_peaks with the specified height/prominence
    - sort the found peaks by their height (intensity) and keep the top N
    - for every selected peak estimate the wavelength & intensity
      and attempt to match it to a species using ADAS data
    - store results in dictionary keyed by channel
    """

    prominent_lines = {}
    for ch in spectra.channel.values:
        spec = spectra.sel(channel=ch)
        y = spec.values.squeeze()  # Ensure y is 1-D
        x = spec.wavelength.values
        peaks, props = find_peaks(y, height = height, prominence=prominence)
        
        # Sort peaks by intensity descending
        sorted_indices = np.argsort(props['peak_heights'])[::-1]
        top_peaks = sorted_indices[:top_n]
        
        lines = []
        for idx in top_peaks:
            wl = x[peaks[idx]]
            intensity = y[peaks[idx]]
            species = match_species_verbose(wl, adf15_data)
            lines.append({
                'wavelength': wl,
                'intensity': intensity,
                'species': species
            })
        prominent_lines[ch] = lines
    return prominent_lines

#############################################################################
############################ PLOTTING FUNCTIONS #############################
#############################################################################

def plot_fz_ion_stages_per_species(Fz, equilibrium, time):
    """
    Plot the fractional abundance (Fz) for each ionization stage of each species.

    Each element is drawn in its own figure with two rows of subplots; the
    number of columns adapts to the number of ion charges.  This visualisation
    helps check if the assumed Li distribution (mimicking C) is reasonable.

    Parameters:
    - Fz: dict of xarray.DataArray, keyed by element symbol (e.g., "c", "h")
    - equilibrium: Equilibrium object from indica
    - time: float, time at which to evaluate equilibrium

    Outline:
    - compute flux surface geometry at the chosen time from the equilibrium
    - for each element in the Fz dictionary:
        * determine how many ionisation stages are present
        * create a grid of subplots (2 rows × ncols)
        * loop over charges and plot a contourf of the 2‑D fractional
          abundance in poloidal plane
        * overlay selected flux surfaces and label with Roman numerals
        * disable any unused axes
    - save figure to disk and display
    """
    rhop = equilibrium.rhop.sel(t=time, method='nearest')

    for element, fz_data in Fz.items():
        ion_charges = fz_data.ion_charge.values
        n_stages    = len(ion_charges)
        ncols       = math.ceil(n_stages / 2)

        fig, axs    = plt.subplots(2, ncols, figsize=(5 * ncols, 10), constrained_layout=True)
        axs         = axs.flatten()

        # plot fractional abundance in the poloidal plane
        for i, charge in enumerate(ion_charges):
            ax       = axs[i]
            fz_stage = fz_data.sel(ion_charge=charge)
            fz_stage.plot.contourf(ax=ax, levels=20, cmap="viridis", alpha=0.7)
            # plot flux surfaces
            rhop.plot.contour(ax=ax, levels=[0.5, 0.7, 0.9, 0.99, 1.01], colors="black", linewidths=1)
            roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
            label = f"{element.upper()} {roman[charge]}" if charge < len(roman) else f"{element.upper()} {charge + 1}"
            ax.set_title(f"Fz: {label}")
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            ax.set_aspect("equal")

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.suptitle(f"Fractional Abundance (Fz) for {element.upper()}", fontsize=16)
        plt.savefig(os.path.join(save_dir, f"Fractional_abundance_{element.upper()}.png"))
        plt.show(block=False)

def plot_target_line_intensities(spectra, target_wavelengths, tolerance=0.5, export=False, export_path='target_line_intensities.json'):
    """
    Extracts and plots the intensity of target spectral lines across LOS channels.

    Parameters:
    - spectra: xarray.DataArray containing synthetic spectra with dimensions (channel, wavelength)
    - target_wavelengths: list of float, target spectral line wavelengths in nm
    - tolerance: float, tolerance for matching wavelengths

    Workflow:
    - determine layout of subplots based on number of lines
    - for each target wavelength:
        * traverse all LOS channels
        * find matches within given tolerance and average if multiple
        * append intensity to list and optionally prepare export record
    - plot each line's intensity versus channel index
    - optionally write results to JSON file
    """
    los_indices = spectra.channel.values
    n_lines     = len(target_wavelengths)
    ncols       = 3 if n_lines > 4 else 2
    nrows       = int(np.ceil(n_lines / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axs      = axs.flatten()

    export_data = []

    for i, target_wl in enumerate(target_wavelengths):
        intensities = []
        for ch in los_indices:
            spec = spectra.sel(channel=ch)
            wl_array = spec.wavelength.values
            intensity_array = spec.values
            match_indices = np.where(np.abs(wl_array - target_wl) < tolerance)[0]
            if match_indices.size > 0:
                intensity = float(np.mean(intensity_array[match_indices]))
            else:
                intensity = 0.0
            intensities.append(intensity)
            export_data.append({
                "los_index"         : int(ch),
                "target_wavelength" : target_wl,
                "intensity"         : intensity
            })

        ax = axs[i]
        ax.plot(los_indices, intensities, marker='o', linestyle='-')
        ax.set_title(f"Intensity at {target_wl} nm")
        ax.set_xlabel("LOS Index")
        ax.set_ylabel("Intensity [W/m³/nm]")
        ax.grid(True)

    # Hide unused subplots if any
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Target Spectral Line Intensities Across LOS", fontsize=16)
    plt.savefig(os.path.join(save_dir, "Li_line_intensity.png"))
    plt.show(block=False)

    

    if export:
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Exported Li line intensities to {export_path}")

#############################################################################

def plot_prominent_lines_vs_los(spectra, prominent_lines, top_n=5, save_dir=None):
    """
    Plot the intensity of the most prominent spectral lines as a function of LOS index.
    LOS index from 0 (top) to N-1 (bottom).

    Parameters:
    - spectra: xarray.DataArray with spectra
    - prominent_lines: output from find_prominent_spectral_lines
    - top_n: number of lines to plot

    Description:
    - determine a colour palette for the top_n lines
    - collect a unique set of (wavelength,label) pairs across all LOS
    - for each unique line compute intensity vs channel by locating the
      nearest wavelength bin in the spectrum
    - draw all curves on a single plot and optionally save
    """

    los_indices = spectra.channel.values
    colors      = cm.tab20(np.linspace(0, 1, top_n))
    
    fig, ax     = plt.subplots(figsize=(10, 6))
    
    # Collect unique lines across all channels
    all_lines   = set()
    for lines in prominent_lines.values():
        for line in lines[:top_n]:
            species = line.get('species', 'Unknown')
            label = f"{species} ({line['wavelength']:.1f} nm)"
            all_lines.add((line['wavelength'], label))
    
    all_lines   = sorted(list(all_lines), key=lambda x: x[0])  # sort by wavelength
    
    for i, (wl, label) in enumerate(all_lines):
        intensities = []
        for ch in los_indices:
            # Find intensity for this wavelength in this channel
            spec            = spectra.sel(channel=ch)
            wl_array        = spec.wavelength.values
            intensity_array = spec.values
            match_idx       = np.argmin(np.abs(wl_array - wl))
            intensities.append(intensity_array[match_idx])
        
        ax.plot(los_indices, intensities, marker='o', label=label, color=colors[i % len(colors)])
    
    ax.set_xlabel("LOS Index (0=top, higher=bottom)")
    ax.set_ylabel("LOS-Integrated Intensity [W/m³/nm]")
    ax.set_title("Intensity of Prominent Spectral Lines vs LOS Index")
    ax.legend()
    ax.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "line_intensity_vs_los.png"), dpi = 300)
    plt.show()

    plt.yscale("log")
    if save_dir:
        plt.savefig(os.path.join(save_dir, "line_intensity_vs_los_logscale.png"), dpi = 300)
    plt.show()

#############################################################################

#############################################################################
#### GET SOLPS AND EQUILIBRIUM DATA FROM ST40 FOR A GIVEN PULSE AND TIME ####
#############################################################################

pulse = 13565 #11890
t     = 0.160 #0.105

print(f"Loading SOLPS and equilibrium data for pulse {pulse} at time {t:.3f} s...")
solps_reader = SOLPSReader(pulse, t)
solps_data   = solps_reader.get()

print("Initialising ST40 Reader for magnetic equilibrium data...")
st40_reader  = ST40Reader(pulse, tstart=0.01, tend=0.17)
equil_data   = st40_reader.get("","efit",)
equilibrium  = indica.Equilibrium(equil_data)
rhop = equilibrium.rhop.sel(t=t, method='nearest')

vessel_data = get_vessel_geometry_data(pulse, "elmag#best")

#############################################################################

#############################################################################
## DEFINE THE TRANSFORM AND THE LINES OF SIGHT FOR THE NEW PI_PFC_DV SYSTEM #
#############################################################################

x_0            = 0.8492 - 0.235 # x_flange - x_viewport
z_0            = 0.7453         # z_flange  

x_viewport     = 0.235


# origin_z_zemax = np.array([7.1,  6.2,  0.2, -0.3,  -2.0,  -2.5, 
#                          -2.9, -3.4, -3.4, -4.5,  -4.9,  -5.3, 
#                           -6.5, -9.2, -9.2, -10.4, -12.6, -16.5]) * 1e-3

# recalculated with Zemax's merit function
origin_z_zemax = 1e-3 * np.array([2.04,  1.34,  0.66, -0.03,  -0.69,  -1.37, 
                          -2.07, -2.79, -3.54, -4.32,  -5.15,  -6.02, 
                          -6.92, -7.93, -9.06, -10.31, -11.74, -13.34], dtype=float) 
# coordinates measured on Zemax at the exit of the viewport
origin_x     = np.array([ x_0, x_0, x_0, x_0, x_0, x_0,
                          x_0, x_0, x_0, x_0, x_0, x_0,
                          x_0, x_0, x_0, x_0, x_0, x_0], dtype=float)
origin_z     = origin_z_zemax + z_0 
origin_y     = 1e-3 * np.array([ 00.00,  00.00,  00.00,  00.00,  00.00,  00.00,
                                 00.00,  00.00,  00.00,  00.00,  00.00,  00.00,
                                 00.00,  00.00,  00.00,  00.00,  00.00,  00.00],  dtype=float)   # assume system is only in the poloidal plane  (2D)  

# coordinates measured on Zemax for the center of the spots on the divertor target
# divertor_x  = 1e-3 * np.array([ 390.97, 395.35,  400.94,  405.31,  410.41,  417.70,
#                                 424.50, 430.81,  438.35,  447.34,  456.08,  466.77,
#                                 477.70, 491.06,  505.63,  523.66,  544.21,  571.16])

# recalculated on a vertical detector placed 100 mm after the window

# divertor_z  = 1e-3 * np.array([ 22.30,   14.53,    8.46,    1.18,   -6.83, -15.57,
#                               -24.31,  -33.05,  -43.49,  -54.65,  -66.07, -80.88,
#                               -94.96, -113.16, -132.35, -156.44, -184.28, -219.87])

# recalculated on a vertical detector placed 100 mm after the window using Zemax's merit function
direction_x  = np.array([ -0.340, -0.340, -0.340, -0.340, -0.340, -0.340,
                          -0.340, -0.340, -0.340, -0.340, -0.340, -0.340,
                          -0.340, -0.340, -0.340, -0.340, -0.340, -0.340], dtype=float)

detector_z   = 1e-3 * np.array([ 42.38,    28.17,   14.06,   -0.01,  -14.07,  -28.18,
                                -42.38,   -56.73,  -71.28,  -86.09, -101.21, -116.69,
                                -132.64, -149.16, -166.28, -183.75, -201.36, -220.11],  dtype=float)
direction_z  = detector_z - origin_z_zemax

direction_y  = 1e-3 * np.array([ 00.00,  00.00,  00.00,  00.00,  00.00,  00.00,
                                 00.00,  00.00,  00.00,  00.00,  00.00,  00.00,
                                 00.00,  00.00,  00.00,  00.00,  00.00,  00.00],  dtype=float)

'''
origin_z_zemax = np.array([7.1,  0.2,  -2.0, 
                          -2.9, -3.4,  -4.9, 
                          -6.5, -9.2, -12.6]) * 1e-3
# coordinates measured on Zemax at the exit of the viewport
origin_x     = np.full(9, x_0)
origin_z     = origin_z_zemax + z_0 
origin_y     = np.zeros(9)   # assume system is only in the poloidal plane  (2D)  

# coordinates measured on Zemax for the center of the spots on the divertor target
divertor_x  = 1e-3 * np.array([ 390.97, 400.94, 410.41, 
                                424.50, 438.35, 456.08,
                                477.70, 505.63, 544.21])
divertor_z  = 1e-3 * np.array([ 22.30,  8.46, -6.83,
                               -24.31, -43.49,  -66.07,
                               -94.96, -132.35, -184.28])
direction_y  = np.zeros(9)
'''

'''
origin_z_zemax = np.array([0.2]) * 1e-3
# coordinates measured on Zemax at the exit of the viewport
origin_x     = np.full(1, x_0)
origin_z     = origin_z_zemax + z_0 
origin_y     = np.zeros(1)   # assume system is only in the poloidal plane  (2D)  

# coordinates measured on Zemax for the center of the spots on the divertor target
# divertor_x  = 1e-3 * np.array([400.94])
# divertor_z  = 1e-3 * np.array([8.46])
# direction_y  = np.zeros(1)

# direction = divertor - origin
# direction_x  = -(divertor_x - x_viewport)
# direction_z  =   divertor_z - origin_z_zemax
'''


machine_dims = ((0.15, 0.85), (0.15, 0.85))
# machine_dims = ((0.15, 0.85), (-0.75, 0.85))
machine_dims = (vessel_data["limiter_trace"].x, vessel_data["limiter_trace"].y)
# transforms   = load_default_objects("st40", "geometry")
# transform    = transforms["blom_dv1"]

# machine_dims  = ((0.15, 0.85), (-0.75, 0.75))
# origin_x      = np.array([1.0, 1.0, 1.0],    dtype=float)
# origin_y      = np.array([0.0, 0.0, 0.0],    dtype=float)
# origin_z      = np.array([0.7, 0.7, 0.7],    dtype=float)
# direction_x   = np.array([-0.8, -0.8, -0.8], dtype=float)
# direction_y   = np.array([0.0, 0.0, 0.0],    dtype=float)
# direction_z   = np.array([0.4, 0.1, 0.0],    dtype=float)
#name          = "dummy_los"

#origin_x      = np.array([x_0, x_0, x_0],    dtype=float)
#origin_y      = np.array([0.0, 0.0, 0.0],    dtype=float)
#origin_z      = np.array([0.7, 0.7, 0.7],    dtype=float)
#direction_x   = np.array([-0.8, -0.8, -0.8], dtype=float)
#direction_y   = np.array([0.0, 0.0, 0.0],    dtype=float)
#direction_z   = np.array([0.4, 0.1, 0.0],    dtype=float)

print("Defining Line of Sight transform for the new PI_PFC_DV lens-based configuration...")
print("Origin (m):")
print("  x:", origin_x)
print("  y:", origin_y)
print("  z:", origin_z)

print("Direction (m):")
print("  x:", direction_x)
print("  y:", direction_y)
print("  z:", direction_z)

transform = LineOfSightTransform(
        origin_x           = origin_x,   
        origin_y           = origin_y,   
        origin_z           = origin_z,   
        direction_x        = direction_x,
        direction_y        = direction_y,
        direction_z        = direction_z,
        name               = "PI_PFC_DV_18LOS_600µm_core_diameter",
        dl                 = 5.00 * 1e-3,
        spot_width         = 5.26 * 1e-3, # measured on Zemax for central LOS (2*sqrt(2)*RMS(X)) 
        spot_height        = 5.26 * 1e-3, # measured on Zemax for central LOS (2*sqrt(2)*RMS(Y))
        spot_shape         = "round",
        beamlets_method    = "adaptive",
        n_beamlets         = 25,
        focal_length       = 336.32 * 1e-3, #867*1e-3, # calculated from central LOS spot sizes using a simple slope function
        machine_dimensions = machine_dims,
        passes             = 1,
        plot_beamlets      = True,
    ) 

transform.set_equilibrium(equilibrium=equilibrium)
divspec = PassiveSpectrometer(name="test", pecs=pecs, window=window)

divspec.set_transform(transform)

### plot the LOS and the flux surfaces to check they are correctly defined in the machine geometry
divspec.transform.plot(t, orientation="Rz", )
for trace in vessel_data["tile_trace"]:
    plt.plot(trace.x, trace.y, color=trace.line.color)
plt.plot(vessel_data["limiter_trace"].x, vessel_data["limiter_trace"].y, color="black", linestyle="--")
rhop.plot.contour(levels=[0.5, 0.7, 0.9, 0.99, 1.01], colors="black", linewidths=1)
rhop.plot.contour(levels=[1], colors="black", linewidths=2)
plt.xlim(0.15, 0.85)
plt.ylim(0.15, 0.85)

plt.grid()
plt.savefig(os.path.join(save_dir, "transform_LOSs.png"), dpi=900)
plt.show(block=False)

fig = plt.figure()
divspec.transform.plot(t, orientation="all", )
plt.show(block=False)
#############################################################################

# Add Li mimicking C density
total_li_density = solps_data["nion"].sel(element="c", t=t)

# Expand Ni to include Li
Ni_expanded = solps_data["nion"].sel(t=t)
li_density  = Ni_expanded.sel(element="c").assign_coords(element="li")
Ni_expanded = xr.concat([Ni_expanded, li_density], dim="element")

# Expand Fz to include Li
Fz_expanded = {k: v.sel(t=t) for k, v in solps_data["fz"].items()}

# Compute Fz for Li mimicking C's ionization distribution
Fz_li_da = Fz_expanded["c"].sel(ion_charge=slice(0, 2))
Fz_expanded["li"] = Fz_li_da

#############################################################################



Nh = solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0)
Nh = Nh.drop_vars(("element", "ion_charge", ))

bckc = divspec(Te=solps_data["te"], Ne=solps_data["ne"], Nimp=Ni_expanded,
               Fz=Fz_expanded, Nh=Nh, Ti=solps_data["te"], t=[t], )

print("done")
emissivity = 0
for element in divspec.intensity.keys():
    emissivity += divspec.intensity[element].sum("wavelength").sum("t")

# nd0 = (solps_data["nion"].sel(element="h") * solps_data["fz"]["h"].sel(ion_charge=0)).sel(t=t)
extent = [emissivity.R.min(), emissivity.R.max(), emissivity.z.min(), emissivity.z.max()]

plt.figure()
plt.title("emissivity (photons/m^3)")
h = plt.imshow(emissivity.values, extent=extent, norm=LogNorm())
plt.colorbar(label="Emissivity (photons/m³)")
h.set_clim(emissivity.values.max()*1e-4, emissivity.values.max())  # Adjust color limits for better contrast
rhop.plot.contour(levels=[0.5, 0.7, 0.9, 0.99, 1.01], colors="black", linewidths=1)
rhop.plot.contour(levels=[1], colors="black", linewidths=2)
plt.axis("equal")
plt.xlim(0.1, 0.9)
plt.grid()
a = os.path.join(save_dir, "emissivity.png")
print(f"Saving emissivity plot to: {a}") 
plt.savefig(a)

plt.figure()
plt.grid()
divspec.transform.plot(t, orientation="Rz", )
h2 = plt.imshow(emissivity.values, extent=extent, norm=LogNorm())
h2.set_clim(emissivity.values.max()*1e-4, emissivity.values.max())  # Adjust color limits for better contrast
vessel_data = get_vessel_geometry_data(pulse, "elmag#best")
for trace in vessel_data["tile_trace"]:
    plt.plot(trace.x, trace.y, color=trace.line.color)
plt.plot(vessel_data["limiter_trace"].x, vessel_data["limiter_trace"].y, color="black", linestyle="--")
rhop.plot.contour(levels=[0.5, 0.7, 0.9, 0.99, 1.01], colors="black", linewidths=1)
rhop.plot.contour(levels=[1], colors="black", linewidths=2)
plt.colorbar(label="Emissivity (photons/m³)")
plt.xlim(0.15, 0.85)
plt.ylim(0.15, 0.85)

plt.savefig(os.path.join(save_dir, "transform_emissivity.png"), dpi=900)
plt.show(block=False)


plt.figure()
spectra = divspec.bckc["spectra"]

# Apply transmission factors to the spectra
transmissivity_da = xr.DataArray(optical_transm_interp, dims=['wavelength'], coords={'wavelength': window})
fiber_transm_da   = xr.DataArray(fiber_transm_interp,   dims=['wavelength'], coords={'wavelength': window})
quantum_eff_da    = xr.DataArray(quantum_eff_interp,    dims=['wavelength'], coords={'wavelength': window})
spectra           = spectra * transmissivity_da * fiber_transm_da * quantum_eff_da

cols_chan = cm.gnuplot2(np.linspace(0.1, 0.75, len(spectra.channel), dtype=float))
for idx, chan_num in enumerate(spectra.channel.values):
    plt.plot(
        spectra.wavelength,
        spectra.sel(t=t, channel=chan_num),
        label=f"channel={chan_num}",
        color=cols_chan[idx],
        alpha=0.8,
    )
plt.ylabel("Emissivity (photon/m^2/nm/s)")
plt.xlabel("Wavelength (nm)")
plt.legend()
plt.savefig(os.path.join(save_dir, "spectra.png"))
plt.show(block=True)

#############################################################################
############################## ANALYZE SPECTRA ##############################
#############################################################################


os.makedirs(save_dir, exist_ok=True)

# Find prominent spectral lines
prominent_lines = find_prominent_spectral_lines(spectra, adf15, prominence=1e-6, height = 1e14, top_n=10)

# Print summary
for ch, lines in prominent_lines.items():
    print(f"\nLOS Channel {ch}:")
    for line in lines:
        species = line['species'] if line['species'] else "Unknown"
        print(f"  {line['wavelength']:.2f} nm ({species}): {line['intensity']:.2e}")

# Plot intensities vs LOS index
# plot_prominent_lines_vs_los(spectra, prominent_lines, top_n=5, save_dir=save_dir)

plot_prominent_lines_vs_los(spectra, prominent_lines, top_n=10, save_dir=save_dir)

# Plot Fz for Li
plot_fz_ion_stages_per_species({"li": Fz_expanded["li"]}, equilibrium, t)

# Define target Li spectral lines (Li II at 548.5 nm, Li I at 670.8 nm)
target_lines_li = [548.5, 670.8]

# Plot the intensities of Li lines across LOS
plot_target_line_intensities(spectra, target_lines_li, export=True, export_path=os.path.join(save_dir, "Li_lines.json"))